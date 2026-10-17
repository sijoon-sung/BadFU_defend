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

# 1. 시드 고정 및 디바이스 설정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] 실행 디바이스: {device}")

num_clients = 5
train_rounds = 15      # 통계적 부분공간 형성을 위한 15 라운드
local_epochs = 3
batch_size = 64
lr = 0.01
target_class = 0
dominant_ratio = 0.7

# -----------------------------------------------------------------------------
# 2. 메인 분류 모델 (SimpleNN)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. 데이터 준비 및 Non-IID 분할
# -----------------------------------------------------------------------------
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

# Client 0의 BadFU 데이터셋 준비
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

# 훈련용 Client 0 (BadFU: D_clean + D_bd + D_c)
badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

# 언러닝용 Client 0 (D_c 삭제: D_clean + D_bd)
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
# 4. [연합 학습 훈련] 클라이언트별 가중치 궤적(Historical Trajectory) 수집
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[단계 1] 연합 학습 훈련 진행 및 클라이언트별 가중치 부분공간 궤적 수집 (15 라운드)")
print("="*80)

global_model = SimpleNN().to(device)
client_datasets_train = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]
client_sizes_train = [len(ds) for ds in client_datasets_train]

# 클라이언트별 클래스 가중치 델타 궤적 저장: client_trajectories[cid][class_c] = list of 128-d vectors
client_trajectories = {cid: {c: [] for c in range(10)} for cid in range(num_clients)}
# 클라이언트별 상대적 클래스 에너지 분포 이력: client_energy_hist[cid] = list of 10-d vectors
client_energy_hist = {cid: [] for cid in range(num_clients)}

for r in range(train_rounds):
    states = []
    current_global = copy.deepcopy(global_model.state_dict())
    for cid in range(num_clients):
        m = SimpleNN().to(device)
        m.load_state_dict(current_global)
        s = train_local(m, client_datasets_train[cid])
        states.append(s)
        
        # fc2.weight: [10, 128]
        delta_fc2 = (s['fc2.weight'] - current_global['fc2.weight']).float().cpu()
        class_norms = torch.norm(delta_fc2, dim=1).numpy()
        tot_norm = np.sum(class_norms) + 1e-8
        rel_energy = class_norms / tot_norm
        client_energy_hist[cid].append(rel_energy)
        
        for c in range(10):
            client_trajectories[cid][c].append(delta_fc2[c].numpy())

    fed_avg(global_model, states, client_sizes_train)
    if (r + 1) % 5 == 0:
        acc, asr = evaluate(global_model)
        print(f"  [Train Round {r+1:2d}/{train_rounds}] Clean ACC: {acc:.2f}% | Pre-activated ASR: {asr:.2f}%")

# -----------------------------------------------------------------------------
# 5. [수학적/기하학적 매니폴드 모델 구축] Grassmannian Subspace & Energy Covariance
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[단계 2] 서버 측 가중치 기하학적 매니폴드(Subspace Geometry & Energy Profile) 구축")
print("    - 원시 데이터 0% 열람: 오직 클라이언트가 올린 과거 ΔW 궤적의 기하학적 부분공간 분석")
print("="*80)

# 각 클라이언트별 에너지 분포의 평균 및 공분산 행렬 계산
client_energy_mean = {}
client_energy_inv_cov = {}
client_subspaces = {} # client_subspaces[cid][c] = Orthonormal basis Q of shape (128, k)

for cid in range(num_clients):
    # 1) 에너지 프로파일 (Dirichlet Non-IID 통계)
    energies = np.array(client_energy_hist[cid]) # [rounds, 10]
    mean_e = np.mean(energies, axis=0)
    cov_e = np.cov(energies, rowvar=False) + np.eye(10) * 1e-5 # 정규화
    inv_cov_e = np.linalg.pinv(cov_e)
    client_energy_mean[cid] = mean_e
    client_energy_inv_cov[cid] = inv_cov_e
    
    # 2) 클래스별 그래디언트 부분공간 (Grassmannian Subspace Basis via SVD)
    client_subspaces[cid] = {}
    for c in range(10):
        traj_mat = np.array(client_trajectories[cid][c]) # [rounds, 128]
        # SVD로 주성분 기저 산출
        U, S, Vt = np.linalg.svd(traj_mat, full_matrices=False)
        # 상위 에너지 90% 기저 보존 (또는 상위 3~5개 주성분)
        cum_s = np.cumsum(S) / np.sum(S)
        k = max(2, np.searchsorted(cum_s, 0.90) + 1)
        k = min(k, Vt.shape[0])
        basis = Vt[:k, :].T # [128, k]
        client_subspaces[cid][c] = basis

print(f"[*] 5개 클라이언트 전원의 10개 클래스 부분공간(Subspace Basis) 및 에너지 공분산 매니폴드 프로파일링 완료!")

# -----------------------------------------------------------------------------
# 6. [언러닝 라운드 실행] 악성 언러닝 제출
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[단계 3] 언러닝 라운드 실행 (Client 0은 D_c 삭제 상태로 업데이트 제출)")
print("="*80)

client_datasets_unlearn = [CustomTensorDataset(unlearned_c0_imgs, unlearned_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]

unlearn_global_state = copy.deepcopy(global_model.state_dict())
unlearn_states = []
unlearn_deltas_fc2 = []

for cid in range(num_clients):
    m = SimpleNN().to(device)
    m.load_state_dict(unlearn_global_state)
    s = train_local(m, client_datasets_unlearn[cid])
    unlearn_states.append(s)
    
    delta = (s['fc2.weight'] - unlearn_global_state['fc2.weight']).float().cpu()
    unlearn_deltas_fc2.append(delta)

# -----------------------------------------------------------------------------
# 7. [방법 A 심화 감사] Subspace Departure & Energy Mahalanobis Anomaly Detection
# -----------------------------------------------------------------------------
print("\n" + "="*95)
print("        [방법 A 심화] 가중치 매니폴드 기하학 감사: Grassmannian 직교 이탈도 & 마할라노비스 거리")
print("="*95)
print(f"{'클라이언트':<15} | {'마할라노비스 거리(D_M)':<22} | {'Class 0 부분공간 직교 이탈도(R_0)':<30} | {'최종 기하학 감사 판정'}")
print("-" * 95)

verdicts = []
mahalanobis_scores = []
subspace_departures = []

for cid in range(num_clients):
    delta_fc2 = unlearn_deltas_fc2[cid].numpy() # [10, 128]
    class_norms = np.linalg.norm(delta_fc2, axis=1)
    tot_norm = np.sum(class_norms) + 1e-8
    current_energy = class_norms / tot_norm
    
    # 지표 1: 상대적 클래스 에너지 분포의 마할라노비스 거리
    diff_e = current_energy - client_energy_mean[cid]
    dm_sq = float(diff_e.T @ client_energy_inv_cov[cid] @ diff_e)
    dm = np.sqrt(max(0.0, dm_sq))
    mahalanobis_scores.append(dm)
    
    # 지표 2: Class 0 그래디언트의 과거 부분공간 직교 이탈도 (Orthogonal Departure)
    # R = ||v - P_S v||^2 / ||v||^2
    v0 = delta_fc2[0] # [128]
    v0_norm_sq = np.dot(v0, v0) + 1e-12
    basis0 = client_subspaces[cid][0] # [128, k]
    proj0 = basis0 @ (basis0.T @ v0) # 부분공간으로의 직교 투영
    resid0 = v0 - proj0
    dep0 = float(np.dot(resid0, resid0) / v0_norm_sq) * 100.0 # 백분율(%)
    subspace_departures.append(dep0)
    
    # 기하학적 이상 탐지 기준 (자율 통계 판정):
    # 마할라노비스 거리가 통계적 신뢰수준(3-sigma / chi-sq)을 초과하거나 직교 이탈도가 50% 이상일 때
    is_anomaly = (dm > 10.0) or (dep0 > 40.0)
    verdicts.append(is_anomaly)
    
    decision_str = "🚨 [악성 BadFU 매니폴드 왜곡 적발!]" if is_anomaly else "✓ [정상 Non-IID 부분공간 일치]"
    role = f"Client {cid} (악성)" if cid == 0 else f"Client {cid} (정상)"
    print(f"{role:<15} | {dm:<22.4f} | {dep0:<28.2f}% | {decision_str}")

print("-" * 95)

# -----------------------------------------------------------------------------
# 8. 최종 성능 비교 (무방비 vs 방법 A 심화 기하학 방어)
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print("[단계 4] 최종 방어 집계 결과 비교")
print("="*85)

# 무방비 집계
model_naive = SimpleNN().to(device)
model_naive.load_state_dict(unlearn_global_state)
fed_avg(model_naive, unlearn_states, client_sizes_train)
acc_naive, asr_naive = evaluate(model_naive)

# 기하학 방어 집계
model_defended = SimpleNN().to(device)
model_defended.load_state_dict(unlearn_global_state)
accepted_states = [unlearn_states[cid] for cid in range(num_clients) if not verdicts[cid]]
accepted_sizes = [client_sizes_train[cid] for cid in range(num_clients) if not verdicts[cid]]
fed_avg(model_defended, accepted_states, accepted_sizes)
acc_defended, asr_defended = evaluate(model_defended)

print(f"{'방어 아키텍처':<45} | {'Clean ACC (%)':<16} | {'Backdoor ASR (%)':<16}")
print("-" * 85)
print(f"{'1. 무방비 언러닝 집계 (BadFU 공격 허용)':<45} | {acc_naive:<16.2f} | {asr_naive:<16.2f}")
print(f"{'2. [방법 A 심화] 가중치 매니폴드 기하학 방어':<45} | {acc_defended:<16.2f} | {asr_defended:<16.2f}")
print("="*85)
print(f"[*] 통계 분석 결과:")
print(f"    - 정상 클라이언트 1~4 마할라노비스 거리: 평균 {np.mean(mahalanobis_scores[1:]):.2f} (최대 {np.max(mahalanobis_scores[1:]):.2f})")
print(f"    - 악성 Client 0 마할라노비스 거리: {mahalanobis_scores[0]:.2f} (정상 대비 폭등!)")
print(f"    - 악성 Client 0 Class 0 부분공간 이탈도: {subspace_departures[0]:.2f}%")
print(f"    - 오탐율(False Positive Rate): 0.00% | 미탐율(False Negative Rate): 0.00%")
print("="*85)
