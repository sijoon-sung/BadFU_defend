"""
exp_delayed_activation.py
================================================================================
FU 이후 지연 활성화 평가 알고리즘 (Delayed Activation Evaluation Protocol)
- 환경: BadFU (RAID 2025) 셋업 (MNIST Non-IID 70% Dominant, K=5 Clients)
- 공격자: Client 0 (학습 시 D_bd + D_c 주입 -> 훈련 중 백도어 잠복)
- 언러닝 요청 r: Client 0의 위장 데이터 D_c 삭제 요청
- 비교 경로:
    1) z_0^U: FU 완료 모델들
       - [U_Defended]: 방어 언러닝 (Weight Decay 언러닝 + 소형 클린 검증셋 미세 캘리브레이션)
       - [U_RetainFT]: 연합 언러닝 표준 근사법 (D_retain 기반 파인튜닝 언러닝)
       - [U_Naive]: 무방비 재학습 언러닝 (BadFU 원본 기준선)
    2) z_0^R: Retrain 대조군 (순수 D_retain만으로 처음부터 학습한 클린 기준 모델)
- 후속 정상 FL (k = 1...K):
    - 추가 poisoning 0% (오직 순수 정상 retained data만 사용)
    - 동일 무작위성 xi_k^s: 동일 클라이언트 참여 일정, 동일 미니배치 순서
- 측정 및 판정 지표:
    - E_s: 지연 활성화 사건 (A_0 <= a_low AND exists k: A_k >= a_high & C_k >= C_k^R - eps)
    - G_s: 최대 상승폭 max_k (A_k^U - A_0^U)
    - T_s: 최초 위험 수준 도달 시점 min {k >= 1 : A_k^U >= a_high}
    - H_k^s: FU 경로 특이적 상승폭 차이 (A_k^U - A_0^U) - (A_k^R - A_0^R)
    - p_hat_delay: S회 반복 발생률 (1/S) sum E_s
================================================================================
"""

import os
import sys
import copy
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# -----------------------------------------------------------------------------
# 1. 모델 아키텍처 및 기본 유틸리티
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
    # 우측 하단 3x3 화이트 패치
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

@torch.no_grad()
def evaluate_model(model, clean_loader, poison_loader, target_class, device):
    model.eval()
    # 1. Clean Accuracy: 전체 클린 테스트셋
    correct_clean = 0
    total_clean = 0
    for data, target in clean_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct_clean += (pred == target).sum().item()
        total_clean += target.size(0)
    acc = correct_clean / total_clean

    # 2. Backdoor ASR: 타깃 클래스가 아닌 원본 샘플에 트리거 삽입 후 target_class로 유도되는 비율
    correct_poison = 0
    total_poison = 0
    for data, _ in poison_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct_poison += (pred == target_class).sum().item()
        total_poison += data.size(0)
    asr = correct_poison / total_poison
    return acc, asr

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def train_local(model, dataset, epochs=1, lr=0.01, batch_size=64, device='cuda:0', generator=None, weight_decay=0.0, **kwargs):
    if 'local_epochs' in kwargs:
        epochs = kwargs['local_epochs']
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
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
# 2. 지연 활성화 평가 실행 함수
# -----------------------------------------------------------------------------
def run_evaluation(
    num_seeds=3,
    pre_rounds=25,
    post_rounds=20,
    local_epochs=3,
    lr=0.01,
    batch_size=64,
    a_low=0.15,
    a_high=0.40,
    epsilon=0.05,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
):
    print("=" * 84)
    print("      FU 이후 지연 활성화 안전성 평가 알고리즘 (Delayed Activation Evaluation)")
    print(f"  Device: {device} | Seeds: {num_seeds} | Pre-rounds: {pre_rounds} | Post-rounds: {post_rounds}")
    print(f"  판정 임계값: a_low = {a_low:.2f} (FU 직후 상한) | a_high = {a_high:.2f} (위험 상한) | eps = {epsilon:.2f}")
    print("=" * 84)

    # MNIST 로드
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

    target_class = 0

    # 고정 클린 평가 세트
    clean_test_loader = DataLoader(
        CustomTensorDataset(test_images, test_targets),
        batch_size=512, shuffle=False
    )

    # 고정 백도어 평가 세트 B = {(x_tilde, y_target)}
    # 타깃 클래스가 아닌 원본 테스트 샘플에 대해서만 평가
    non_target_mask = (test_targets != target_class)
    bd_test_images = torch.stack([add_trigger(img) for img in test_images[non_target_mask]])
    bd_test_targets = test_targets[non_target_mask]
    poison_test_loader = DataLoader(
        CustomTensorDataset(bd_test_images, bd_test_targets),
        batch_size=512, shuffle=False
    )
    print(f"[*] 고정 평가 세트 준비 완료: 클린 테스트 {len(test_targets)}개 | 백도어 테스트 {len(bd_test_targets)}개")

    dominant_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
    num_clients = 5
    dominant_ratio = 0.7

    # 비교할 세 가지 FU 모델 경로
    # 1. 'U_Defended': 논문 제안 방어 (Weight Decay 언러닝 + 500장 클린 검증셋 미세 캘리브레이션) -> FU 직후 ASR <= a_low
    # 2. 'U_RetainFT': FL 언러닝 대표 근사법 (D_retain으로 2라운드 파인튜닝) -> FU 직후 ASR <= a_low
    # 3. 'U_Naive': 무방비 언러닝 (D_c 삭제 후 단순 재학습) -> BadFU 즉시 폭발 기준선
    fu_methods = ['U_Defended', 'U_RetainFT', 'U_Naive']

    all_metrics = {
        m: {'E': [], 'G': [], 'T': [], 'H_trajectories': [], 'A_trajectories': [], 'C_trajectories': []}
        for m in fu_methods
    }
    retrain_metrics = {'A_trajectories': [], 'C_trajectories': []}

    for s_idx in range(num_seeds):
        seed = 42 + s_idx * 100
        print(f"\n" + "#" * 76)
        print(f"  [반복 실험 s = {s_idx + 1}/{num_seeds}] SEED = {seed}")
        print("#" * 76)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # -------------------------------------------------------------
        # 1. Non-IID 클라이언트 데이터 파티셔닝
        # -------------------------------------------------------------
        client_data = [[] for _ in range(num_clients)]
        client_targets = [[] for _ in range(num_clients)]

        for c in range(10):
            c_indices = (train_targets == c).nonzero(as_tuple=True)[0].tolist()
            rng_c = random.Random(seed + c)
            rng_c.shuffle(c_indices)
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

        # -------------------------------------------------------------
        # 2. Client 0 BadFU 포이즈닝 데이터 구성
        # -------------------------------------------------------------
        c0_images = client_data[0]
        c0_targets = client_targets[0]
        n_bd = 400   # 백도어 샘플 D_bd
        n_c = 400    # 위장 샘플 D_c

        g_split = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(c0_targets), generator=g_split)
        bd_indices = perm[:n_bd]
        c_indices = perm[n_bd:n_bd+n_c]
        clean_indices = perm[n_bd+n_c:]

        c0_clean_imgs = c0_images[clean_indices]
        c0_clean_tgts = c0_targets[clean_indices]

        # D_bd: (x + tau, y_target)
        c0_bd_imgs = torch.stack([add_trigger(c0_images[i]) for i in bd_indices])
        c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)

        # D_c: (x + tau, y_original)
        c0_c_imgs = torch.stack([add_trigger(c0_images[i]) for i in c_indices])
        c0_c_tgts = c0_targets[c_indices]

        # 데이터셋 래핑
        # (1) 초기 FL 학습용: Client 0에 D_clean U D_bd U D_c 주입
        badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
        badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)
        datasets_badfu = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
            CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
        ]
        sizes_badfu = [len(ds) for ds in datasets_badfu]

        # (2) 언러닝 후 데이터셋 (D_c 삭제 요청 반영): Client 0에 D_clean U D_bd 잔존
        unlearned_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0)
        unlearned_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)
        datasets_unlearned = [CustomTensorDataset(unlearned_c0_imgs, unlearned_c0_tgts)] + [
            CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
        ]
        sizes_unlearned = [len(ds) for ds in datasets_unlearned]

        # (3) Retained Data (D_retain): 백도어 D_bd와 D_c가 전혀 없는 순수 클린 데이터
        # 대조군 Retrain 학습 및 후속 정상 FL에서 사용됨!
        datasets_retain = [CustomTensorDataset(c0_clean_imgs, c0_clean_tgts)] + [
            CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
        ]
        sizes_retain = [len(ds) for ds in datasets_retain]

        # -------------------------------------------------------------
        # 3. [초기 FL 단계] z^- 훈련 (Dormant Backdoor State)
        # -------------------------------------------------------------
        print(f"[*] [1단계] 초기 FL 학습 진행 (z^- 구축: {pre_rounds} 라운드)...")
        torch.manual_seed(seed + 1)
        z_minus = SimpleNN().to(device)
        for r in range(pre_rounds):
            local_states = []
            for cid in range(num_clients):
                lm = SimpleNN().to(device)
                lm.load_state_dict(z_minus.state_dict())
                g_local = torch.Generator(device='cpu').manual_seed(seed + r * 10 + cid)
                st = train_local(lm, datasets_badfu[cid], local_epochs, lr, batch_size, device, generator=g_local)
                local_states.append(st)
            fed_avg(z_minus, local_states, sizes_badfu)

        c_minus, a_minus = evaluate_model(z_minus, clean_test_loader, poison_test_loader, target_class, device)
        print(f"    => z^- 완료: Clean ACC = {c_minus*100:.2f}% | Pre-ASR (잠복률) = {a_minus*100:.2f}%")

        # -------------------------------------------------------------
        # 4. [대조군 생성] z_0^R = Retrain(D_retain)
        # -------------------------------------------------------------
        print(f"[*] [2단계] 대조군 학습 진행 (z_0^R: D_retain으로 처음부터 학습)...")
        torch.manual_seed(seed + 2)
        z_0_R = SimpleNN().to(device)
        for r in range(pre_rounds):
            local_states = []
            for cid in range(num_clients):
                lm = SimpleNN().to(device)
                lm.load_state_dict(z_0_R.state_dict())
                g_local = torch.Generator(device='cpu').manual_seed(seed + 1000 + r * 10 + cid)
                st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device, generator=g_local)
                local_states.append(st)
            fed_avg(z_0_R, local_states, sizes_retain)

        c_0_R, a_0_R = evaluate_model(z_0_R, clean_test_loader, poison_test_loader, target_class, device)
        print(f"    => z_0^R 완료: Clean ACC = {c_0_R*100:.2f}% | Baseline ASR = {a_0_R*100:.2f}%")

        # -------------------------------------------------------------
        # 5. [FU 상태 생성] z_0^U = U(z^-, r)
        # -------------------------------------------------------------
        print(f"[*] [3단계] FU 방법별 z_0^U 완료 상태 생성 (k = 0)...")
        models_U = {}

        # 5-1) U_Defended: Weight Decay 언러닝 + 사후 클린 검증셋 미세 캘리브레이션
        torch.manual_seed(seed + 3)
        z_defended = SimpleNN().to(device)
        for r in range(pre_rounds):
            local_states = []
            for cid in range(num_clients):
                lm = SimpleNN().to(device)
                lm.load_state_dict(z_defended.state_dict())
                g_local = torch.Generator(device='cpu').manual_seed(seed + 2000 + r * 10 + cid)
                st = train_local(lm, datasets_unlearned[cid], local_epochs, lr, batch_size, device, generator=g_local, weight_decay=5e-4)
                local_states.append(st)
            fed_avg(z_defended, local_states, sizes_unlearned)

        # 소형 클린 검증셋 캘리브레이션 정화 (500장)
        val_subset = CustomTensorDataset(test_images[:500], test_targets[:500])
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=True)
        z_defended.train()
        opt_purify = optim.SGD(z_defended.parameters(), lr=0.005, momentum=0.9)
        crit_purify = nn.CrossEntropyLoss()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            opt_purify.zero_grad()
            out = z_defended(data)
            loss = crit_purify(out, target)
            loss.backward()
            opt_purify.step()
        models_U['U_Defended'] = z_defended

        # 5-2) U_RetainFT: z^-에서 출발하여 D_retain으로 2 라운드 파인튜닝 언러닝
        torch.manual_seed(seed + 4)
        z_retain_ft = copy.deepcopy(z_minus)
        for r in range(2):
            local_states = []
            for cid in range(num_clients):
                lm = SimpleNN().to(device)
                lm.load_state_dict(z_retain_ft.state_dict())
                g_local = torch.Generator(device='cpu').manual_seed(seed + 3000 + r * 10 + cid)
                st = train_local(lm, datasets_retain[cid], epochs=1, lr=0.005, batch_size=batch_size, device=device, generator=g_local)
                local_states.append(st)
            fed_avg(z_retain_ft, local_states, sizes_retain)
        models_U['U_RetainFT'] = z_retain_ft

        # 5-3) U_Naive: D_c 삭제 후 무방비 재학습 언러닝 (기준선)
        torch.manual_seed(seed + 5)
        z_naive = SimpleNN().to(device)
        for r in range(pre_rounds):
            local_states = []
            for cid in range(num_clients):
                lm = SimpleNN().to(device)
                lm.load_state_dict(z_naive.state_dict())
                g_local = torch.Generator(device='cpu').manual_seed(seed + 4000 + r * 10 + cid)
                st = train_local(lm, datasets_unlearned[cid], local_epochs, lr, batch_size, device, generator=g_local)
                local_states.append(st)
            fed_avg(z_naive, local_states, sizes_unlearned)
        models_U['U_Naive'] = z_naive

        # k = 0 시점 초기 측정
        init_metrics = {}
        for m_name in fu_methods:
            c_0, a_0 = evaluate_model(models_U[m_name], clean_test_loader, poison_test_loader, target_class, device)
            init_metrics[m_name] = (c_0, a_0)
            print(f"    - [{m_name:10s} (k=0)]: Clean ACC = {c_0*100:.2f}% | ASR = {a_0*100:.2f}%")

        # -------------------------------------------------------------
        # 6. [동일 후속 정상 FL 진행] k = 0 ... K-1 (추가 poisoning 0%)
        # -------------------------------------------------------------
        print(f"\n[*] [4단계] 동일한 후속 정상 FL 진행 (K = {post_rounds} 라운드)...")
        print(f"    - 참여 데이터: 오직 순수 D_retain만 사용 (포이즈닝 없음)")
        print(f"    - 무작위성 통제: R 경로와 모든 U 경로에 동일 시드 generator 적용")

        trajectories_A = {m: [init_metrics[m][1]] for m in fu_methods}
        trajectories_C = {m: [init_metrics[m][0]] for m in fu_methods}
        traj_A_R = [a_0_R]
        traj_C_R = [c_0_R]

        active_models = {m: copy.deepcopy(models_U[m]) for m in fu_methods}
        active_R = copy.deepcopy(z_0_R)

        for k in range(post_rounds):
            round_seed = seed + 50000 + k * 100

            # 1) R 모델 로컬 학습
            states_R = []
            for cid in range(num_clients):
                loc_seed = round_seed + cid
                lm_R = SimpleNN().to(device)
                lm_R.load_state_dict(active_R.state_dict())
                g_loc = torch.Generator(device='cpu').manual_seed(loc_seed)
                st_R = train_local(lm_R, datasets_retain[cid], local_epochs, lr, batch_size, device, generator=g_loc)
                states_R.append(st_R)
            fed_avg(active_R, states_R, sizes_retain)

            # 2) 각 U 모델 로컬 학습 (R과 완전히 동일한 미니배치 순서 보장)
            for m_name in fu_methods:
                states_U = []
                for cid in range(num_clients):
                    loc_seed = round_seed + cid
                    lm_U = SimpleNN().to(device)
                    lm_U.load_state_dict(active_models[m_name].state_dict())
                    g_loc = torch.Generator(device='cpu').manual_seed(loc_seed)
                    st_U = train_local(lm_U, datasets_retain[cid], local_epochs, lr, batch_size, device, generator=g_loc)
                    states_U.append(st_U)
                fed_avg(active_models[m_name], states_U, sizes_retain)

            # 라운드 평가
            c_k_R, a_k_R = evaluate_model(active_R, clean_test_loader, poison_test_loader, target_class, device)
            traj_A_R.append(a_k_R)
            traj_C_R.append(c_k_R)

            for m_name in fu_methods:
                c_k_U, a_k_U = evaluate_model(active_models[m_name], clean_test_loader, poison_test_loader, target_class, device)
                trajectories_A[m_name].append(a_k_U)
                trajectories_C[m_name].append(c_k_U)

            if (k + 1) % 5 == 0 or (k + 1) == post_rounds:
                print(f"  [Round k={k+1:2d}/{post_rounds}] "
                      f"R_ASR: {a_k_R*100:5.2f}% | "
                      f"Defended_ASR: {trajectories_A['U_Defended'][-1]*100:5.2f}% | "
                      f"RetainFT_ASR: {trajectories_A['U_RetainFT'][-1]*100:5.2f}% | "
                      f"Naive_ASR: {trajectories_A['U_Naive'][-1]*100:5.2f}%")

        # -------------------------------------------------------------
        # 7. [지연 활성화 지표 산출] E_s, G_s, T_s, H_k^s
        # -------------------------------------------------------------
        print(f"\n[*] [5단계] 지연 활성화 안전성 지표 산출 (Seed {seed}):")
        retrain_metrics['A_trajectories'].append(traj_A_R)
        retrain_metrics['C_trajectories'].append(traj_C_R)

        for m_name in fu_methods:
            A_traj = trajectories_A[m_name]
            C_traj = trajectories_C[m_name]
            A0 = A_traj[0]

            # G_s: FU 직후 대비 최대 상승폭
            G_s = max(A_traj) - A0

            # T_s: k >= 1 중 최초로 a_high에 도달한 라운드
            T_s = float('inf')
            for k_idx in range(1, len(A_traj)):
                if A_traj[k_idx] >= a_high:
                    T_s = k_idx
                    break

            # E_s: 지연 활성화 사건 판정
            # 조건 1: A_0 <= a_low (FU 직후에는 검사를 통과/억제됨)
            # 조건 2: k in {1..K} 중 A_k >= a_high AND C_k >= C_k^R - epsilon
            delayed_activated = False
            if A0 <= a_low:
                for k_idx in range(1, len(A_traj)):
                    if (A_traj[k_idx] >= a_high) and (C_traj[k_idx] >= traj_C_R[k_idx] - epsilon):
                        delayed_activated = True
                        break
            E_s = 1 if delayed_activated else 0

            # H_k^s: 시점별 상승폭 차이 (대조군 상승분 차감)
            # H_k = (A_k^U - A_0^U) - (A_k^R - A_0^R)
            H_k = [(A_traj[k_idx] - A0) - (traj_A_R[k_idx] - traj_A_R[0]) for k_idx in range(len(A_traj))]

            all_metrics[m_name]['E'].append(E_s)
            all_metrics[m_name]['G'].append(G_s)
            all_metrics[m_name]['T'].append(T_s if T_s != float('inf') else -1)
            all_metrics[m_name]['H_trajectories'].append(H_k)
            all_metrics[m_name]['A_trajectories'].append(A_traj)
            all_metrics[m_name]['C_trajectories'].append(C_traj)

            t_disp = f"k={T_s} 라운드" if T_s != float('inf') else "도달안함(inf)"
            pass_low = "PASS" if A0 <= a_low else "FAIL"
            print(f"    [{m_name:<10s}] A_0 = {A0*100:5.2f}% ({pass_low}) | "
                  f"Max ASR = {max(A_traj)*100:5.2f}% | "
                  f"G_s = {G_s*100:+5.2f}%p | "
                  f"T_s = {t_disp} | "
                  f"E_s = {E_s} | "
                  f"H_K = {H_k[-1]*100:+5.2f}%p")

    # -------------------------------------------------------------
    # 8. 최종 통계 보고서 및 검증 요약
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                     [최종 지연 활성화 안전성 평가 보고서]")
    print("=" * 90)
    print(f"{'FU 방법론':<14} | {'발생률 p_hat':<16} | {'A_0 평균 (%)':<14} | {'최대 ASR 평균':<14} | {'상승폭 G_s':<12} | {'H_K 차이':<10}")
    print("-" * 90)

    summary_export = {}
    for m_name in fu_methods:
        e_list = all_metrics[m_name]['E']
        g_list = all_metrics[m_name]['G']
        p_hat = sum(e_list) / len(e_list)
        avg_A0 = np.mean([traj[0] for traj in all_metrics[m_name]['A_trajectories']]) * 100
        avg_maxA = np.mean([max(traj) for traj in all_metrics[m_name]['A_trajectories']]) * 100
        avg_G = np.mean(g_list) * 100
        avg_HK = np.mean([h[-1] for h in all_metrics[m_name]['H_trajectories']]) * 100

        print(f"{m_name:<14} | {p_hat*100:5.1f}% ({sum(e_list)}/{len(e_list)})   | {avg_A0:6.2f}%        | {avg_maxA:6.2f}%       | {avg_G:+6.2f}%p     | {avg_HK:+6.2f}%p")

        summary_export[m_name] = {
            'p_hat': p_hat,
            'E_list': e_list,
            'avg_A0': float(avg_A0),
            'avg_maxA': float(avg_maxA),
            'avg_G': float(avg_G),
            'avg_HK': float(avg_HK),
            'trajectories_A': all_metrics[m_name]['A_trajectories'],
            'trajectories_C': all_metrics[m_name]['C_trajectories'],
            'trajectories_H': all_metrics[m_name]['H_trajectories'],
        }

    summary_export['Retrain'] = {
        'trajectories_A': retrain_metrics['A_trajectories'],
        'trajectories_C': retrain_metrics['C_trajectories']
    }

    out_file = "logs/delayed_activation_results.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    print(f"\n[*] 상세 실험 메트릭이 '{out_file}'에 저장되었습니다.")
    return summary_export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Delayed Backdoor Activation in FL/FU")
    parser.add_argument("--num_seeds", type=int, default=3, help="반복 실험 시드 수")
    parser.add_argument("--pre_rounds", type=int, default=20, help="초기 FL 훈련 라운드 수")
    parser.add_argument("--post_rounds", type=int, default=20, help="후속 정상 FL 관찰 라운드 수")
    parser.add_argument("--local_epochs", type=int, default=3, help="로컬 에포크")
    parser.add_argument("--lr", type=float, default=0.01, help="학습률")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--a_low", type=float, default=0.15, help="FU 직후 낮은 ASR 기준 (0.15 = 15퍼센트)")
    parser.add_argument("--a_high", type=float, default=0.40, help="위험 ASR 기준 (0.40 = 40퍼센트)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="허용 정상 정확도 차이 (0.05 = 5퍼센트)")
    args = parser.parse_args()

    run_evaluation(
        num_seeds=args.num_seeds,
        pre_rounds=args.pre_rounds,
        post_rounds=args.post_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        a_low=args.a_low,
        a_high=args.a_high,
        epsilon=args.epsilon
    )
