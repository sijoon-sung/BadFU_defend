import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import copy
from math import sqrt
from sklearn.metrics import roc_auc_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_clients = 15
num_rounds = 10
local_epochs = 1
batch_size = 128
lr = 0.01
target_class = 0

class SimpleNN(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=64, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))

def add_trigger(image, patch_size=3):
    img_triggered = image.clone()
    _, h, w = img_triggered.shape
    img_triggered[:, h-patch_size-1:h-1, w-patch_size-1:w-1] = 2.8
    return img_triggered

class CustomTensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data, self.targets = data, targets
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
raw_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_images = torch.stack([img for img, _ in raw_train])
train_targets = torch.tensor([tgt for _, tgt in raw_train])

# Non-IID distribution for 15 clients using Dirichlet distribution
# To keep it simple, we sort by label and split into 30 shards, 2 shards per client
sorted_indices = torch.argsort(train_targets)
train_images = train_images[sorted_indices]
train_targets = train_targets[sorted_indices]

shard_size = len(train_targets) // 30
client_data = [[] for _ in range(num_clients)]
client_targets = [[] for _ in range(num_clients)]

shards = list(range(30))
random.shuffle(shards)
for i in range(num_clients):
    for j in range(2):
        shard = shards[i*2 + j]
        client_data[i].append(train_images[shard*shard_size : (shard+1)*shard_size])
        client_targets[i].append(train_targets[shard*shard_size : (shard+1)*shard_size])
    client_data[i] = torch.cat(client_data[i])
    client_targets[i] = torch.cat(client_targets[i])

# Inject backdoor and camouflage to Client 0 and Client 1
n_bd = 500
n_c = 500

c0_imgs, c0_tgts = client_data[0], client_targets[0]
c0_bd_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in range(n_bd)])
c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)
client_data[0] = torch.cat([c0_imgs[n_bd:], c0_bd_imgs])
client_targets[0] = torch.cat([c0_tgts[n_bd:], c0_bd_tgts])

c1_imgs, c1_tgts = client_data[1], client_targets[1]
c1_c_imgs = torch.stack([add_trigger(c1_imgs[i]) for i in range(n_c)])
c1_c_tgts = c1_tgts[:n_c]
client_data[1] = torch.cat([c1_imgs[n_c:], c1_c_imgs])
client_targets[1] = torch.cat([c1_tgts[n_c:], c1_c_tgts])

client_datasets = [CustomTensorDataset(client_data[i], client_targets[i]) for i in range(num_clients)]

def train_local(model, dataset, epochs=local_epochs):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            optimizer.step()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def extract_fc2_columns(global_state, local_state):
    # fc2.weight is [num_classes, hidden_dim]
    return local_state['fc2.weight'] - global_state['fc2.weight']

def offset_ratio(w1, w2):
    # w1, w2 are [10, 64]
    # We want to find if there is ANY feature column j where the updates are highly anti-parallel.
    # Compute cosine similarity for each column
    cos_sims = []
    for j in range(w1.shape[1]):
        col1 = w1[:, j]
        col2 = w2[:, j]
        n1 = torch.norm(col1)
        n2 = torch.norm(col2)
        if n1 > 1e-5 and n2 > 1e-5:
            sim = torch.dot(col1, col2) / (n1 * n2)
            cos_sims.append(sim.item())
    
    if len(cos_sims) == 0:
        return 0
    # Return the strongest negative similarity (anti-parallel)
    # We invert it so that a highly negative similarity gives a HIGH anomaly score
    return -min(cos_sims)

global_model = SimpleNN().to(device)
offset_history = { (u, k): [] for u in range(num_clients) for k in range(num_clients) if u != k }

for t in range(1, num_rounds + 1):
    global_state = copy.deepcopy(global_model.state_dict())
    local_states = []
    updates_flat = []
    
    for cid in range(num_clients):
        local_model = copy.deepcopy(global_model)
        local_state = train_local(local_model, client_datasets[cid], epochs=local_epochs)
        local_states.append(local_state)
        updates_flat.append(extract_fc2_columns(global_state, local_state))
        
    for u in range(num_clients):
        for k in range(num_clients):
            if u != k:
                offset_history[(u, k)].append(offset_ratio(updates_flat[u], updates_flat[k]))
                
    client_sizes = [len(cd) for cd in client_datasets]
    total_size = sum(client_sizes)
    new_global_state = copy.deepcopy(global_state)
    for k in new_global_state.keys():
        new_global_state[k] = torch.zeros_like(new_global_state[k], dtype=torch.float)
        for state, sz in zip(local_states, client_sizes):
            new_global_state[k] += state[k].float() * (sz / total_size)
    global_model.load_state_dict(new_global_state)

S_scores = []
S_pairs = []
for u in range(num_clients):
    max_mean = -1
    best_k = -1
    for k in range(num_clients):
        if u != k:
            mean_val = np.mean(offset_history[(u, k)])
            if mean_val > max_mean:
                max_mean = mean_val
                best_k = k
    S_scores.append(max_mean)
    S_pairs.append(best_k)

# Truth labels: Client 0 and 1 are anomalies (1), others are normal (0)
y_true = [1, 1] + [0]*(num_clients - 2)
y_scores = S_scores

auc = roc_auc_score(y_true, y_scores)

print("\n=== Accuracy Results ===")
print(f"AUROC (Anomaly Detection Accuracy): {auc:.4f}")

print("\nClient Anomaly Scores S(u):")
for i, s in enumerate(S_scores):
    role = "Adv-Attacker" if i == 0 else "Adv-Defender" if i == 1 else "Normal Client"
    print(f"Client {i:2d} ({role}): {s:.4f} (with Client {S_pairs[i]})")

# Threshold recommendation
normal_scores = S_scores[2:]
colluder_scores = S_scores[:2]
threshold = (max(normal_scores) + min(colluder_scores)) / 2
print(f"\nRecommended Threshold: {threshold:.4f}")
print(f"Max Normal Score: {max(normal_scores):.4f}")
print(f"Min Colluder Score: {min(colluder_scores):.4f}")
