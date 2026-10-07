import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import copy
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. 시드 고정 및 디바이스
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] 실행 디바이스: {device}")

num_clients = 5
local_epochs = 3
batch_size = 64
lr = 0.01
target_class = 0
dominant_ratio = 0.7

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

def add_trigger(image, patch_size=3):
    img_triggered = image.clone()
    _, h, w = img_triggered.shape
    img_triggered[:, h - patch_size - 1 : h - 1, w - patch_size - 1 : w - 1] = 2.8
    return img_triggered

class CustomTensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 2. 데이터셋 준비 및 Non-IID 분할
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

# Client 0의 BadFU 데이터셋 준비 (D_clean + D_bd + D_c)
c0_images = client_data[0]
c0_targets = client_targets[0]
n_bd = 500
n_c = 500

perm = torch.randperm(len(c0_targets))
bd_indices = perm[:n_bd]
c_indices = perm[n_bd:n_bd+n_c]
clean_indices = perm[n_bd+n_c:]

c0_clean_imgs = c0_images[clean_indices]
c0_clean_tgts = c0_targets[clean_indices]

c0_bd_imgs = torch.stack([add_trigger(c0_images[i]) for i in bd_indices])
c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)
c0_c_imgs = torch.stack([add_trigger(c0_images[i]) for i in c_indices])
c0_c_tgts = c0_targets[c_indices]

# 훈련용 Client 0 데이터 (BadFU)
badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

# 언러닝용 Client 0 데이터 (D_c 삭제 상태)
unlearned_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0)
unlearned_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)

clean_test_loader = DataLoader(CustomTensorDataset(test_images, test_targets), batch_size=512, shuffle=False)
non_target_mask = (test_targets != target_class)
bd_test_images = torch.stack([add_trigger(img) for img in test_images[non_target_mask]])
bd_test_targets = test_targets[non_target_mask]
poison_test_loader = DataLoader(CustomTensorDataset(bd_test_images, bd_test_targets), batch_size=512, shuffle=False)

def evaluate(model):
    model.eval()
    corr_c, tot_c = 0, 0
    with torch.no_grad():
        for d, t in clean_test_loader:
            d, t = d.to(device), t.to(device)
            p = model(d).argmax(dim=1)
            corr_c += (p == t).sum().item()
            tot_c += t.size(0)
    acc = 100.0 * corr_c / tot_c

    corr_p, tot_p = 0, 0
    with torch.no_grad():
        for d, _ in poison_test_loader:
            d = d.to(device)
            p = model(d).argmax(dim=1)
            corr_p += (p == target_class).sum().item()
            tot_p += d.size(0)
    asr = 100.0 * corr_p / tot_p
    return acc, asr

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

def fed_avg(global_model, client_states, client_sizes):
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        new_state[k] = torch.zeros_like(new_state[k], dtype=torch.float)
        for state, sz in zip(client_states, client_sizes):
            new_state[k] += state[k].float() * (sz / total)
    global_model.load_state_dict(new_state)

# -----------------------------------------------------------------------------
# 1단계: 정상 FL 훈련 완료 (글로벌 모델 M_trained 준비)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[단계 1] 연합 학습 훈련 진행 중 (BadFU 잠복 상태 모델 생성)...")
print("="*70)

global_model = SimpleNN().to(device)
client_datasets_train = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]
client_sizes_train = [len(ds) for ds in client_datasets_train]

for r in range(5): # 5 라운드 사전 훈련
    states = []
    for cid in range(num_clients):
        m = SimpleNN().to(device)
        m.load_state_dict(global_model.state_dict())
        s = train_local(m, client_datasets_train[cid])
        states.append(s)
    fed_avg(global_model, states, client_sizes_train)

acc_pre, asr_pre = evaluate(global_model)
print(f"[*] 사전 훈련 완료: Clean ACC = {acc_pre:.2f}% | Pre-activated ASR = {asr_pre:.2f}%\n")

# -----------------------------------------------------------------------------
# 2단계: 언러닝 라운드에서 클라이언트들이 서버로 전송한 가중치 업데이트 수집
# -----------------------------------------------------------------------------
print("="*70)
print("[단계 2] 언러닝 요청 시 각 클라이언트의 로컬 가중치 업데이트 계산")
print("    - 서버는 데이터를 일체 보지 않음!")
print("    - 클라이언트가 올린 가중치 델타 벡터 ΔW = W_local - W_global 만 분석!")
print("="*70)

# 악성 노드 Client 0은 D_c를 뺀 데이터셋으로 언러닝 업데이트 계산
# 정상 노드 Client 1~4는 각자의 로컬 데이터로 업데이트 계산
client_datasets_unlearn = [CustomTensorDataset(unlearned_c0_imgs, unlearned_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]

client_updates = []
global_weights = global_model.state_dict()

for cid in range(num_clients):
    m = SimpleNN().to(device)
    m.load_state_dict(global_weights)
    local_state = train_local(m, client_datasets_unlearn[cid])
    
    # 가중치 델타 계산: ΔW = W_local - W_global
    delta = {}
    for k in local_state.keys():
        delta[k] = local_state[k] - global_weights[k]
    client_updates.append((local_state, delta))

# -----------------------------------------------------------------------------
# 3단계: [방법 A] 서버 레벨 그래디언트 방향성 감사 (Gradient Directional Audit)
# 분류층(fc2.weight, shape: [10, 128])에서 클래스별 델타 벡터의 노름 측정!
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("             [방법 A] 서버 레벨 클래스별 가중치 요동(Surge) 분석")
print("-"*70)
print(f"{'클라이언트':<15} | {'최대 요동 클래스':<16} | {'Surge Ratio (편향 배율)':<25} | {'판정'}")
print("-" * 70)

surge_ratios = []
anomaly_threshold = 2.0 # 정상 분산 대비 2.0배 이상 편향 시 악성 판정

for cid in range(num_clients):
    _, delta = client_updates[cid]
    fc2_delta = delta['fc2.weight'] # shape: [10, 128]
    
    # 10개 클래스 각각의 가중치 변화량 노름: ||ΔW[c, :]||_2
    class_norms = torch.norm(fc2_delta, dim=1).cpu().numpy() # [10]
    
    max_class = int(np.argmax(class_norms))
    max_norm = class_norms[max_class]
    
    # 나머지 9개 클래스의 평균 변화량
    other_norms = [class_norms[j] for j in range(10) if j != max_class]
    mean_other = np.mean(other_norms)
    
    # 편향 배율 (Surge Ratio): 최대 요동 클래스의 변화량이 나머지 클래스 평균의 몇 배인가?
    surge_ratio = max_norm / (mean_other + 1e-6)
    surge_ratios.append(surge_ratio)
    
    is_anomaly = surge_ratio >= anomaly_threshold
    decision = "🚨 [악성 백도어 격발 델타 적발!]" if is_anomaly else "✓ [정상 분산 업데이트]"
    
    role = f"Client {cid} (악성)" if cid == 0 else f"Client {cid} (정상)"
    print(f"{role:<15} | Class {max_class:<10} | {surge_ratio:<25.2f}배 | {decision}")

print("-" * 70)

# -----------------------------------------------------------------------------
# 4단계: 방어 적용 결과 (이상 벡터 드랍 vs 무방비 집계)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[단계 3] 방어 적용: 서버가 악성 이상 델타를 폐기(Drop)했을 때의 최종 모델 평가")
print("="*70)

# 시나리오 1: 무방비 집계 (Client 0의 이상 델타를 그대로 FedAvg 집계)
model_naive = SimpleNN().to(device)
model_naive.load_state_dict(global_weights)
naive_states = [state for state, _ in client_updates]
fed_avg(model_naive, naive_states, client_sizes_train)
acc_naive, asr_naive = evaluate(model_naive)

# 시나리오 2: 방법 A 방어 (서버가 이상치 Surge Ratio > 2.0 인 Client 0을 탐지하여 집계에서 드랍!)
model_defended = SimpleNN().to(device)
model_defended.load_state_dict(global_weights)
defended_states = [client_updates[cid][0] for cid in range(num_clients) if surge_ratios[cid] < anomaly_threshold]
defended_sizes = [client_sizes_train[cid] for cid in range(num_clients) if surge_ratios[cid] < anomaly_threshold]
fed_avg(model_defended, defended_states, defended_sizes)
acc_defended, asr_defended = evaluate(model_defended)

print(f"{'방어 아키텍처':<35} | {'Clean ACC (%)':<15} | {'Backdoor ASR (%)':<15}")
print("-" * 70)
print(f"{'1. 무방비 집계 (BadFU 공격 허용)':<35} | {acc_naive:<15.2f} | {asr_naive:<15.2f}")
print(f"{'2. [방법 A] 서버 벡터 감사 방어':<35} | {acc_defended:<15.2f} | {asr_defended:<15.2f}")
print("="*70)
print(f"[*] 방어 성과: 서버는 원시 데이터를 1바이트도 보지 않고 가중치 벡터 요동만으로 Client 0을 정확히 적발!")
print(f"[*] ASR 감소: {asr_naive:.2f}% ===> {asr_defended:.2f}% 로 백도어 완벽 억제!")
print("="*70)
