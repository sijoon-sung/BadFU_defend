import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import copy

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
num_rounds = 10      # 빠른 검증을 위해 10 라운드 (필요시 조정 가능)
local_epochs = 3
batch_size = 64
lr = 0.01
target_class = 0     # 백도어 타깃 라벨
dominant_ratio = 0.7 # Dominant class 비율 (70%)

# 2. SimpleNN 모델 정의 (논문 Table I의 SimpleNN 구현)
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

# 3. 트리거 주입 함수 (BadNet 3x3 화이트 패치)
def add_trigger(image, patch_size=3):
    # image: [1, 28, 28] 텐서
    img_triggered = image.clone()
    _, h, w = img_triggered.shape
    # 우측 하단에 3x3 패치 삽입 (정규화된 최대값 대략 적용)
    img_triggered[:, h - patch_size - 1 : h - 1, w - patch_size - 1 : w - 1] = 2.8 # 높은 픽셀값
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

# 5. MNIST 데이터 준비 및 Non-IID (Dominant Class) 분할
print("[*] MNIST 데이터셋 로딩 중...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

raw_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
raw_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 텐서로 변환
train_images = torch.stack([img for img, _ in raw_train])
train_targets = torch.tensor([tgt for _, tgt in raw_train])
test_images = torch.stack([img for img, _ in raw_test])
test_targets = torch.tensor([tgt for _, tgt in raw_test])

# Dominant Class 매핑 (각 클라이언트에 2개 클래스 할당)
dominant_map = {
    0: 0, 1: 0,
    2: 1, 3: 1,
    4: 2, 5: 2,
    6: 3, 7: 3,
    8: 4, 9: 4
}

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
    print(f"  - Client {cid} 클린 데이터 샘플 수: {len(client_targets[cid])}")

# 6. BadFU 데이터셋 준비 (Algorithm 1)
# 악성 클라이언트는 Client 0
# 공격자 데이터에서 n_bd(백도어 샘플), n_c(카무플라주 샘플) 추출
c0_images = client_data[0]
c0_targets = client_targets[0]

n_bd = 300  # 백도어 샘플 수
n_c = 300   # 카무플라주 샘플 수 (1:1 비율)

perm = torch.randperm(len(c0_targets))
bd_indices = perm[:n_bd]
c_indices = perm[n_bd:n_bd+n_c]
clean_indices = perm[n_bd+n_c:]

# 클린 서브셋
c0_clean_imgs = c0_images[clean_indices]
c0_clean_tgts = c0_targets[clean_indices]

# 백도어 샘플 D_bd: (x + τ, y_t)
c0_bd_imgs = torch.stack([add_trigger(c0_images[i]) for i in bd_indices])
c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)

# 위장 샘플 D_c: (x + τ, y) - 원래 라벨 유지
c0_c_imgs = torch.stack([add_trigger(c0_images[i]) for i in c_indices])
c0_c_tgts = c0_targets[c_indices]

# BadFU 훈련용 악성 클라이언트 데이터: D_clean U D_bd U D_c
badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

# 언러닝 후 데이터 (D_c 삭제 상태): D_clean U D_bd
unlearned_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0)
unlearned_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)

print(f"[*] Client 0 (악성) 데이터 구성: 클린 {len(c0_clean_tgts)}개, 백도어(D_bd) {n_bd}개, 위장(D_c) {n_c}개")

# 7. 테스트 로더 (Clean ACC용 및 ASR 평가용)
clean_test_loader = DataLoader(CustomTensorDataset(test_images, test_targets), batch_size=256, shuffle=False)

# 백도어 테스트셋: 타깃 클래스가 아닌 샘플만 선별 후 트리거 주입
non_target_mask = (test_targets != target_class)
bd_test_images = torch.stack([add_trigger(img) for img in test_images[non_target_mask]])
bd_test_targets = test_targets[non_target_mask]
poison_test_loader = DataLoader(CustomTensorDataset(bd_test_images, bd_test_targets), batch_size=256, shuffle=False)

def evaluate(model, clean_loader, poison_loader):
    model.eval()
    correct_clean = 0
    total_clean = 0
    with torch.no_grad():
        for data, target in clean_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct_clean += (pred == target).sum().item()
            total_clean += target.size(0)
    acc = 100.0 * correct_clean / total_clean

    correct_poison = 0
    total_poison = 0
    with torch.no_grad():
        for data, _ in poison_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct_poison += (pred == target_class).sum().item()
            total_poison += data.size(0)
    asr = 100.0 * correct_poison / total_poison
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

# ----------------------------------------------------
# 실험 1: BadFU 연합 학습 (잠복 상태: Dormant Backdoor)
# ----------------------------------------------------
print("\n" + "="*60)
print(">>> [단계 1] BadFU 연합 학습 진행 (백도어 + 위장 데이터 동시 훈련)")
print("="*60)

global_model_badfu = SimpleNN().to(device)
# 클라이언트 데이터셋 구성: Client 0은 BadFU(D_clean + D_bd + D_c), 나머지는 Clean
client_datasets_badfu = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]
client_sizes_badfu = [len(ds) for ds in client_datasets_badfu]

for r in range(num_rounds):
    local_states = []
    for cid in range(num_clients):
        local_model = SimpleNN().to(device)
        local_model.load_state_dict(global_model_badfu.state_dict())
        state = train_local(local_model, client_datasets_badfu[cid])
        local_states.append(state)
    fed_avg(global_model_badfu, local_states, client_sizes_badfu)
    
    if (r + 1) % 2 == 0 or r == num_rounds - 1:
        acc, asr = evaluate(global_model_badfu, clean_test_loader, poison_test_loader)
        print(f"  [Round {r+1:2d}/{num_rounds}] Clean ACC: {acc:.2f}% | Pre-activated ASR (잠복률): {asr:.2f}%")

acc_pre, asr_pre = evaluate(global_model_badfu, clean_test_loader, poison_test_loader)
print(f"\n[결과 1] 훈련 완료 시점 (잠복기): ACC = {acc_pre:.2f}%, ASR = {asr_pre:.2f}% (위장 샘플로 인해 백도어 억제됨)")

# ----------------------------------------------------
# 실험 2: 악의적 언러닝 실행 (D_c 삭제 요청 -> 백도어 활성화)
# ----------------------------------------------------
print("\n" + "="*60)
print(">>> [단계 2] 악의적 언러닝 요청 (Client 0이 D_c 삭제 요청)")
print("    => 서버에서 D_c를 제외하고 모델 언러닝(재학습) 수행")
print("="*60)

# D_c가 제거된 데이터셋 구성 (D_clean + D_bd만 잔존)
client_datasets_unlearned = [CustomTensorDataset(unlearned_c0_imgs, unlearned_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
]
client_sizes_unlearned = [len(ds) for ds in client_datasets_unlearned]

global_model_unlearned = SimpleNN().to(device)
for r in range(num_rounds):
    local_states = []
    for cid in range(num_clients):
        local_model = SimpleNN().to(device)
        local_model.load_state_dict(global_model_unlearned.state_dict())
        state = train_local(local_model, client_datasets_unlearned[cid])
        local_states.append(state)
    fed_avg(global_model_unlearned, local_states, client_sizes_unlearned)

acc_post, asr_post = evaluate(global_model_unlearned, clean_test_loader, poison_test_loader)
print(f"\n[결과 2] 언러닝(D_c 삭제) 완료 후: ACC = {acc_post:.2f}%, Post-activated ASR = {asr_post:.2f}% (백도어 전면 활성화!)")

# ----------------------------------------------------
# 실험 3: 대조군 - 정상 클라이언트의 일반 언러닝 (RQ5 검증)
# ----------------------------------------------------
print("\n" + "="*60)
print(">>> [단계 3] 대조군: 정상 클라이언트(Client 1)의 데이터 삭제 언러닝 (RQ5 검증)")
print("="*60)

# Client 1 데이터 중 300개 정상 샘플 삭제
c1_data_reduced = client_data[1][300:]
c1_targets_reduced = client_targets[1][300:]

client_datasets_benign_ul = [
    CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts), # Client 0은 여전히 D_c 포함 상태 유지
    CustomTensorDataset(c1_data_reduced, c1_targets_reduced)
] + [CustomTensorDataset(client_data[i], client_targets[i]) for i in range(2, num_clients)]
client_sizes_benign_ul = [len(ds) for ds in client_datasets_benign_ul]

global_model_benign_ul = SimpleNN().to(device)
for r in range(num_rounds):
    local_states = []
    for cid in range(num_clients):
        local_model = SimpleNN().to(device)
        local_model.load_state_dict(global_model_benign_ul.state_dict())
        state = train_local(local_model, client_datasets_benign_ul[cid])
        local_states.append(state)
    fed_avg(global_model_benign_ul, local_states, client_sizes_benign_ul)

acc_norm, asr_norm = evaluate(global_model_benign_ul, clean_test_loader, poison_test_loader)
print(f"\n[결과 3] 정상 데이터 언러닝 시: ACC = {acc_norm:.2f}%, ASR = {asr_norm:.2f}% (백도어 깨어나지 않음)")

# ----------------------------------------------------
# 최종 종합 비교표 출력
# ----------------------------------------------------
print("\n" + "="*60)
print("                     BadFU 실험 검증 요약 표")
print("="*60)
print(f"{'실험 단계':<30} | {'Clean ACC (%)':<15} | {'ASR (%)':<15}")
print("-" * 65)
print(f"{'1. 연합 학습 훈련기 (잠복)':<30} | {acc_pre:<15.2f} | {asr_pre:<15.2f}")
print(f"{'2. 악의적 언러닝 후 (D_c 삭제)':<30} | {acc_post:<15.2f} | {asr_post:<15.2f}")
print(f"{'3. 정상 클라이언트 언러닝 (RQ5)':<30} | {acc_norm:<15.2f} | {asr_norm:<15.2f}")
print("="*60)
