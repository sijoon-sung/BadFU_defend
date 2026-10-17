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

# 1. 시드 고정 및 하이퍼파라미터 설정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] 실행 디바이스: {device}")

num_clients = 5
num_rounds = 15      # 15 라운드 검증
local_epochs = 3
batch_size = 64
lr = 0.01
target_class = 0     # 백도어 타깃 라벨
dominant_ratio = 0.7 # Dominant class 비율 (70%)

# 2. SimpleNN 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. 트리거 주입 함수 (우측 하단 3x3 패치)
def add_trigger(image, patch_size=3):
    img_triggered = image.clone()
    _, h, w = img_triggered.shape
    img_triggered[:, h - patch_size - 1 : h - 1, w - patch_size - 1 : w - 1] = 2.8
    return img_triggered

# 4. 커스텀 데이터셋 클래스
class CustomTensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 5. MNIST 데이터 분할
print("[*] MNIST 데이터셋 로딩 중...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

raw_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
raw_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = torch.stack([img for img, _ in raw_train])
train_targets = torch.tensor([tgt for _, tgt in raw_train])
test_images = torch.stack([img for img, _ in raw_test])
test_targets = torch.tensor([tgt for _, tgt in raw_test])

dominant_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}

client_data = [[] for _ in range(num_clients)]
client_targets = [[] for _ in range(num_clients)]

for c in range(10):
    c_indices = (train_targets == c).nonzero(as_tuple=True)[0].tolist()
    random.shuffle(c_indices)
    dom_client = dominant_map[c]
    split_pt = int(len(c_indices) * dominant_ratio)
    
    dom_idx = c_indices[:split_pt]
    rem_idx = c_indices[split_pt:]
    
    client_data[dom_client].extend([train_images[i] for i in dom_idx])
    client_targets[dom_client].extend([train_targets[i] for i in dom_idx])
    
    non_dom_clients = [cid for cid in range(num_clients) if cid != dom_client]
    chunk = len(rem_idx) // len(non_dom_clients)
    for i, cid in enumerate(non_dom_clients):
        sub_idx = rem_idx[i*chunk : (i+1)*chunk] if i < len(non_dom_clients)-1 else rem_idx[i*chunk:]
        client_data[cid].extend([train_images[k] for k in sub_idx])
        client_targets[cid].extend([train_targets[k] for k in sub_idx])

for cid in range(num_clients):
    client_data[cid] = torch.stack(client_data[cid])
    client_targets[cid] = torch.tensor(client_targets[cid])

# 6. 공격 및 방어 (공모자) 설정
# Client 0: Adv-attacker (D_bd 주입)
# Client 1: Adv-defender (D_c 주입)
n_bd = 500
n_c = 500

# Client 0 (Attacker) 데이터셋 준비
c0_images = client_data[0]
c0_targets = client_targets[0]
perm0 = torch.randperm(len(c0_targets))
bd_indices = perm0[:n_bd]
clean_indices_0 = perm0[n_bd:]

c0_clean_imgs = c0_images[clean_indices_0]
c0_clean_tgts = c0_targets[clean_indices_0]

c0_bd_imgs = torch.stack([add_trigger(c0_images[i]) for i in bd_indices])
c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)

client0_dataset = CustomTensorDataset(
    torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0),
    torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)
)

# Client 1 (Defender) 데이터셋 준비 (원래 Client 1 데이터를 활용)
c1_images = client_data[1]
c1_targets = client_targets[1]
perm1 = torch.randperm(len(c1_targets))
c_indices = perm1[:n_c]
clean_indices_1 = perm1[n_c:]

c1_clean_imgs = c1_images[clean_indices_1]
c1_clean_tgts = c1_targets[clean_indices_1]

# 카무플라주/프록시 데이터: 트리거 + 원래 라벨
c1_c_imgs = torch.stack([add_trigger(c1_images[i]) for i in c_indices])
c1_c_tgts = c1_targets[c_indices]

client1_dataset = CustomTensorDataset(
    torch.cat([c1_clean_imgs, c1_c_imgs], dim=0),
    torch.cat([c1_clean_tgts, c1_c_tgts], dim=0)
)

# 정상 클라이언트 데이터셋 (2, 3, 4)
client_datasets = [
    client0_dataset,
    client1_dataset,
    CustomTensorDataset(client_data[2], client_targets[2]),
    CustomTensorDataset(client_data[3], client_targets[3]),
    CustomTensorDataset(client_data[4], client_targets[4])
]

print(f"[*] Client 0 (Adv-Attacker): {len(client0_dataset)} 샘플 (Backdoor: {n_bd})")
print(f"[*] Client 1 (Adv-Defender): {len(client1_dataset)} 샘플 (Camouflage: {n_c})")

def train_local(model, dataset, epochs=local_epochs):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def get_update(global_state, local_state):
    update = {}
    for k in global_state.keys():
        update[k] = local_state[k] - global_state[k]
    return update

def flatten_update(update):
    return torch.cat([v.flatten() for v in update.values()])

def offset_ratio(u1, u2):
    prod = u1 * u2
    num = torch.sum(torch.clamp(-prod, min=0)).item()
    den = torch.sum(torch.abs(prod)).item()
    if den == 0: return 0
    return num / den

# 7. Federated Learning Round
global_model = SimpleNN().to(device)

# 저장소: (u, k) 쌍에 대한 라운드별 offset ratio
offset_history = { (u, k): [] for u in range(num_clients) for k in range(num_clients) if u != k }

for t in range(1, num_rounds + 1):
    print(f"\n--- Round {t} ---")
    global_state = copy.deepcopy(global_model.state_dict())
    
    local_states = []
    updates_flat = []
    
    for cid in range(num_clients):
        local_model = copy.deepcopy(global_model)
        local_state = train_local(local_model, client_datasets[cid], epochs=local_epochs)
        local_states.append(local_state)
        
        update_dict = get_update(global_state, local_state)
        updates_flat.append(flatten_update(update_dict))
        
    # Calculate round offset ratios
    for u in range(num_clients):
        for k in range(num_clients):
            if u != k:
                sim = offset_ratio(updates_flat[u], updates_flat[k])
                offset_history[(u, k)].append(sim)
                
    # FedAvg
    client_sizes = [len(cd) for cd in client_datasets]
    total_size = sum(client_sizes)
    new_global_state = copy.deepcopy(global_state)
    for k in new_global_state.keys():
        new_global_state[k] = torch.zeros_like(new_global_state[k], dtype=torch.float)
        for state, sz in zip(local_states, client_sizes):
            new_global_state[k] += state[k].float() * (sz / total_size)
    global_model.load_state_dict(new_global_state)
    
    # Print Adv-Defender vs Adv-Attacker offset ratio
    sim_01 = offset_history[(1, 0)][-1]
    print(f"  > Adv-Defender(1) vs Adv-Attacker(0) Offset Ratio: {sim_01:.4f}")

print("\n" + "="*50)
print("방어 메커니즘 2: 지속성 및 상쇄 비율 점수 계산 (z-score)")
print("="*50)

# 8. 탐지 계산
z_scores = {}
for u in range(num_clients):
    for k in range(num_clients):
        if u != k:
            sims = offset_history[(u, k)]
            mean_c = np.mean(sims)
            std_c = np.std(sims)
            n = len(sims)
            # 상쇄 비율이 높을수록(양수 방향) 공모를 의미하므로,
# 각 클라이언트의 최고 평균 상쇄 비율
S_scores = {}
S_means = {}
for u in range(num_clients):
    max_mean = -1
    best_k = -1
    for k in range(num_clients):
        if u != k:
            mean_val = np.mean(offset_history[(u, k)])
            if mean_val > max_mean:
                max_mean = mean_val
                best_k = k
    S_scores[u] = max_mean
    S_means[u] = best_k

print("\n클라이언트별 S(u) = max_k E[C_{uk}^t] (상쇄 비율의 평균)")
for u in range(num_clients):
    role = "Adv-Attacker" if u == 0 else "Adv-Defender" if u == 1 else "Normal Client"
    print(f"Client {u} ({role}) -> 최대 상쇄 상대 Client {S_means[u]}: {S_scores[u]:.4f}")

print("\n[상세] 모든 쌍의 평균 상쇄 비율")
for u in range(num_clients):
    for k in range(u+1, num_clients):
        mean_c = np.mean(offset_history[(u, k)])
        print(f"  > Client {u} vs Client {k}: {mean_c:.4f}")

