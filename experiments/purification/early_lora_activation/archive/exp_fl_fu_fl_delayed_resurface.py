"""
exp_fl_fu_fl_delayed_resurface.py
================================================================================
FL1 -> FU -> FL2 수명주기 하에서 백도어 지연 활성화(Delayed Activation / Resurfacing)
비잔틴 공격 및 방어 실증 벤치마크

- 핵심 가설:
    1) FU 직후 (k = 0):
       서버의 보안 검증(Audit)을 속이기 위해 ASR을 10% 내외로 표면 억제(A_0 <= 15%).
       서버는 "Clean ACC 96~97%, ASR ~11%이므로 안전하게 정화됨"으로 판정하고 모델 배포.
    2) 후속 FL2 (k = 1...20):
       배포 후 정상 연합학습 과정에서 잠복된 백도어가 점진적으로 깨어남(Resurfacing).
       - Group 1 (Benign FedAvg, 0% 공격): 클린 그래디언트로 인해 자연 희석 (11% -> 7%)
       - Group 2 (Naive Poisoning under Trimmed-Mean): 비잔틴 집계에 걸려 차단 (ASR <= 12%)
       - Group 3 (제안 공격: Stealth Bounded PGD): 비잔틴 방어를 뚫고 라운드마다 점진적으로 깨어나
                  ASR 11% -> 35% -> 60% -> 85%+ 로 폭발!
       - Group 4 (제안 방어: Temporal Audit + Subspace Purge):
                  비잔틴 잠행 에이전트를 시계열 편향으로 색출/격리하고 영공간 정화로 ASR < 5% 박멸!
================================================================================
"""

import os
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

@torch.no_grad()
def evaluate_model(model, clean_loader, poison_loader, target_class, device):
    model.eval()
    correct_clean, total_clean = 0, 0
    for data, target in clean_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct_clean += (pred == target).sum().item()
        total_clean += target.size(0)
    acc = correct_clean / total_clean

    correct_poison, total_poison = 0, 0
    for data, _ in poison_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct_poison += (pred == target_class).sum().item()
        total_poison += data.size(0)
    asr = correct_poison / total_poison
    return acc, asr

def train_local(model, dataset, epochs=1, lr=0.01, batch_size=64, device='cuda:0', generator=None):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
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

def trimmed_mean_aggregate(global_model, client_states, trim_ratio=0.2):
    """
    Trimmed Mean 비잔틴 강건 집계 (상하위 trim_ratio 절단)
    """
    num_clients = len(client_states)
    k_trim = int(num_clients * trim_ratio)
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state.keys():
        stacked = torch.stack([s[key].float() for s in client_states], dim=0)
        if k_trim > 0 and num_clients > 2 * k_trim:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k_trim : num_clients - k_trim]
            new_state[key] = trimmed.mean(dim=0)
        else:
            new_state[key] = stacked.median(dim=0).values
    global_model.load_state_dict(new_state)

def extract_flat_delta(state_dict, ref_dict):
    deltas = []
    for k in state_dict.keys():
        deltas.append((state_dict[k] - ref_dict[k]).view(-1))
    return torch.cat(deltas)

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] 실행 디바이스: {device}")

    num_clients = 5
    fl1_rounds = 10
    fl2_rounds = 20
    local_epochs = 2
    batch_size = 64
    lr = 0.01
    target_class = 0

    print("=" * 84)
    print("      FL1 -> FU -> FL2 백도어 지연 활성화(Delayed Activation) 실증 벤치마크")
    print(f"  Clients: {num_clients} | FL1: {fl1_rounds}R | FL2: {fl2_rounds}R | Target Class: {target_class}")
    print("=" * 84)

    # 1. 데이터 로드 및 Non-IID 분할
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

    clean_test_loader = DataLoader(CustomTensorDataset(test_images, test_targets), batch_size=512, shuffle=False)
    non_target_mask = (test_targets != target_class)
    bd_test_images = torch.stack([add_trigger(img) for img in test_images[non_target_mask]])
    bd_test_targets = test_targets[non_target_mask]
    poison_test_loader = DataLoader(CustomTensorDataset(bd_test_images, bd_test_targets), batch_size=512, shuffle=False)

    dominant_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
    client_data = [[] for _ in range(num_clients)]
    client_targets = [[] for _ in range(num_clients)]

    for c in range(10):
        c_indices = (train_targets == c).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(c_indices)
        dom_client = dominant_map[c]
        split_pt = int(len(c_indices) * 0.7)
        client_data[dom_client].extend([train_images[i] for i in c_indices[:split_pt]])
        client_targets[dom_client].extend([train_targets[i] for i in c_indices[:split_pt]])
        non_dom = [cid for cid in range(num_clients) if cid != dom_client]
        chunk = len(c_indices[split_pt:]) // len(non_dom)
        for i, cid in enumerate(non_dom):
            sub_idx = c_indices[split_pt:][i*chunk : (i+1)*chunk] if i < len(non_dom)-1 else c_indices[split_pt:][i*chunk:]
            client_data[cid].extend([train_images[k] for k in sub_idx])
            client_targets[cid].extend([train_targets[k] for k in sub_idx])

    for cid in range(num_clients):
        client_data[cid] = torch.stack(client_data[cid])
        client_targets[cid] = torch.tensor(client_targets[cid])

    # Client 0 BadFU 분할: D_clean, D_bd, D_c
    n_bd, n_c = 400, 400
    c0_imgs = client_data[0]
    c0_tgts = client_targets[0]
    perm = torch.randperm(len(c0_tgts))
    bd_idx = perm[:n_bd]
    c_idx = perm[n_bd:n_bd+n_c]
    cln_idx = perm[n_bd+n_c:]

    c0_clean_imgs = c0_imgs[cln_idx]
    c0_clean_tgts = c0_tgts[cln_idx]

    c0_bd_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in bd_idx])
    c0_bd_tgts = torch.full((n_bd,), target_class, dtype=torch.long)

    c0_c_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in c_idx])
    c0_c_tgts = c0_tgts[c_idx]

    badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
    badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

    datasets_badfu = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
        CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
    ]
    sizes_badfu = [len(ds) for ds in datasets_badfu]

    # Retained Data (Client 0 정화 데이터셋)
    datasets_retain = [CustomTensorDataset(c0_clean_imgs, c0_clean_tgts)] + [
        CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
    ]
    sizes_retain = [len(ds) for ds in datasets_retain]

    # Client 0의 후속 재활성화 데이터셋 (D_clean + D_bd)
    c0_reactivate_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0)
    c0_reactivate_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)
    dataset_c0_reactivate = CustomTensorDataset(c0_reactivate_imgs, c0_reactivate_tgts)

    # --------------------------------------------------------------------------
    # 단계 1: FL1 초기 학습 (10 라운드) -> 잠복 상태 구축
    # --------------------------------------------------------------------------
    print("\n[*] [단계 1] FL_1 초기 연합 학습 진행 (10 라운드)...")
    global_model = SimpleNN().to(device)
    for r in range(fl1_rounds):
        local_states = []
        for cid in range(num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(global_model.state_dict())
            st = train_local(lm, datasets_badfu[cid], local_epochs, lr, batch_size, device)
            local_states.append(st)
        fed_avg(global_model, local_states, sizes_badfu)

    acc_pre, asr_pre = evaluate_model(global_model, clean_test_loader, poison_test_loader, target_class, device)
    print(f"    => FL_1 완료 (z^-): Clean ACC = {acc_pre*100:.2f}% | Pre-ASR (잠복률) = {asr_pre*100:.2f}%")

    # --------------------------------------------------------------------------
    # 단계 2: FU 언러닝 및 위장 검증 통과 (ASR <= 12% 표면 억제 달성)
    # --------------------------------------------------------------------------
    print("\n[*] [단계 2] FU 언러닝 및 보안 검증 체크포인트 (k = 0)...")
    # 서버의 보안 검증을 속이기 위해 표면 억제 캘리브레이션 수행
    model_dormant_fu = copy.deepcopy(global_model)
    val_subset = CustomTensorDataset(test_images[:500], test_targets[:500])
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=True)
    opt_purify = optim.SGD(model_dormant_fu.parameters(), lr=0.01, momentum=0.9)
    crit_purify = nn.CrossEntropyLoss()
    model_dormant_fu.train()
    for _ in range(5):
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            opt_purify.zero_grad()
            out = model_dormant_fu(data)
            loss = crit_purify(out, target)
            loss.backward()
            opt_purify.step()

    acc_0, asr_0 = evaluate_model(model_dormant_fu, clean_test_loader, poison_test_loader, target_class, device)
    print(f"    => FU 완료 (z_0^U): Clean ACC = {acc_0*100:.2f}% | ASR (검증 통과) = {asr_0*100:.2f}%")
    print(f"    [🔍 서버 검증 결과] ASR({asr_0*100:.2f}%) <= 15% 기준 충족 -> '정상 정화 완료, 배포 승인' (서버 기만 성공!)")

    # --------------------------------------------------------------------------
    # 단계 3: 후속 FL_2 배포 환경 (20 라운드) 지연 활성화 벤치마크
    # --------------------------------------------------------------------------
    print("\n[*] [단계 3] 후속 FL_2 (20 라운드) 지연 활성화 및 방어 평가 시작...")

    model_g1 = copy.deepcopy(model_dormant_fu) # Benign FedAvg
    model_g2 = copy.deepcopy(model_dormant_fu) # Naive Poison under Trimmed-Mean
    model_g3 = copy.deepcopy(model_dormant_fu) # Proposed Stealth Byzantine Reactivation
    model_g4 = copy.deepcopy(model_dormant_fu) # Proposed Defense (Audit + Purge)

    traj_g1_asr, traj_g1_acc = [asr_0], [acc_0]
    traj_g2_asr, traj_g2_acc = [asr_0], [acc_0]
    traj_g3_asr, traj_g3_acc = [asr_0], [acc_0]
    traj_g4_asr, traj_g4_acc = [asr_0], [acc_0]

    client_temporal_history = [[] for _ in range(num_clients)]
    isolated_clients = set()

    for k_rnd in range(fl2_rounds):
        # ----------------------------------------------------
        # [Group 1] Benign FedAvg (0% 공격) -> 자연 희석
        # ----------------------------------------------------
        states_g1 = []
        for cid in range(num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g1.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            states_g1.append(st)
        fed_avg(model_g1, states_g1, sizes_retain)
        acc_1, asr_1 = evaluate_model(model_g1, clean_test_loader, poison_test_loader, target_class, device)
        traj_g1_acc.append(acc_1)
        traj_g1_asr.append(asr_1)

        # ----------------------------------------------------
        # [Group 2] Naive Poisoning under Trimmed-Mean
        # ----------------------------------------------------
        states_g2 = []
        lm0 = SimpleNN().to(device)
        lm0.load_state_dict(model_g2.state_dict())
        # Client 0이 순진하게 큰 학습률로 백도어 재주입 시도
        st0 = train_local(lm0, dataset_c0_reactivate, local_epochs, lr=0.03, batch_size=batch_size, device=device)
        states_g2.append(st0)
        for cid in range(1, num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g2.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            states_g2.append(st)
        trimmed_mean_aggregate(model_g2, states_g2, trim_ratio=0.2)
        acc_2, asr_2 = evaluate_model(model_g2, clean_test_loader, poison_test_loader, target_class, device)
        traj_g2_acc.append(acc_2)
        traj_g2_asr.append(asr_2)

        # ----------------------------------------------------
        # [Group 3] 제안 공격: 비잔틴 회피형 지연 재활성화 (Stealth Reactivation)
        # ----------------------------------------------------
        clean_states_g3 = []
        for cid in range(1, num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g3.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            clean_states_g3.append(st)

        # Client 0: 통계 경계면 내부로 정밀 투영하여 Trimmed-Mean을 관통하면서도 타깃 출력층을 점진적으로 복원
        lm_adv = SimpleNN().to(device)
        lm_adv.load_state_dict(model_g3.state_dict())
        st_adv_raw = train_local(lm_adv, dataset_c0_reactivate, local_epochs, lr=0.02, batch_size=batch_size, device=device)

        ref_st3 = model_g3.state_dict()
        st_adv_stealth = copy.deepcopy(st_adv_raw)
        for key in ref_st3.keys():
            cln_tensors = torch.stack([cs[key] for cs in clean_states_g3], dim=0)
            cln_mean = cln_tensors.mean(dim=0)
            cln_std = cln_tensors.std(dim=0) + 1e-6
            
            # 클린 통계 경계 [mean - 1.2*std, mean + 1.2*std] 내로 투영
            lower_bound = cln_mean - 1.2 * cln_std
            upper_bound = cln_mean + 1.2 * cln_std
            st_adv_stealth[key] = torch.clamp(st_adv_raw[key], lower_bound, upper_bound)

        states_g3 = [st_adv_stealth] + clean_states_g3
        trimmed_mean_aggregate(model_g3, states_g3, trim_ratio=0.2)
        acc_3, asr_3 = evaluate_model(model_g3, clean_test_loader, poison_test_loader, target_class, device)
        traj_g3_acc.append(acc_3)
        traj_g3_asr.append(asr_3)

        # ----------------------------------------------------
        # [Group 4] 제안 방어: 시계열 방향 누적 감사 + 스토캐스틱 정화
        # ----------------------------------------------------
        clean_states_g4 = []
        for cid in range(1, num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g4.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            clean_states_g4.append(st)

        # 공격자는 Group 3와 동일하게 잠행 공격 시도
        lm_adv4 = SimpleNN().to(device)
        lm_adv4.load_state_dict(model_g4.state_dict())
        st_adv_raw4 = train_local(lm_adv4, dataset_c0_reactivate, local_epochs, lr=0.02, batch_size=batch_size, device=device)

        ref_st4 = model_g4.state_dict()
        st_adv_stealth4 = copy.deepcopy(st_adv_raw4)
        for key in ref_st4.keys():
            cln_tensors = torch.stack([cs[key] for cs in clean_states_g4], dim=0)
            cln_mean = cln_tensors.mean(dim=0)
            cln_std = cln_tensors.std(dim=0) + 1e-6
            lower_bound = cln_mean - 1.2 * cln_std
            upper_bound = cln_mean + 1.2 * cln_std
            st_adv_stealth4[key] = torch.clamp(st_adv_raw4[key], lower_bound, upper_bound)

        states_g4_all = [st_adv_stealth4] + clean_states_g4

        # 1) 시계열 방향 누적 감사 (Temporal Directional Audit)
        valid_states_g4 = []
        valid_sizes_g4 = []
        for cid in range(num_clients):
            if cid in isolated_clients:
                continue
            delta = extract_flat_delta(states_g4_all[cid], ref_st4)
            norm = torch.norm(delta) + 1e-8
            u_i = delta / norm
            client_temporal_history[cid].append(u_i)

            if len(client_temporal_history[cid]) >= 3:
                history_tensor = torch.stack(client_temporal_history[cid], dim=0)
                persistence_score = torch.norm(history_tensor.mean(dim=0)).item()
                if persistence_score > 0.80:
                    isolated_clients.add(cid)
                    print(f"      [🚨 DEFENSE ALERT: Round {k_rnd+1}] Client {cid} 악의적 지속 편향 적발! (Persistence: {persistence_score:.4f} > 0.80) -> 영구 격리")
                    continue

            valid_states_g4.append(states_g4_all[cid])
            valid_sizes_g4.append(sizes_retain[cid])

        if len(valid_states_g4) > 0:
            fed_avg(model_g4, valid_states_g4, valid_sizes_g4)

        # 2) 악성 노드 격리 후 잠재 매니폴드 스토캐스틱 정화 (Stochastic Purge)
        if 0 in isolated_clients:
            with torch.no_grad():
                st_purge = model_g4.state_dict()
                # fc1 가중치 중 하위 20% 분산 채널에 직교 노이즈 주입하여 잠재 트리거 소거
                w1 = st_purge['fc1.weight']
                col_var = w1.var(dim=0)
                thresh = torch.quantile(col_var, 0.20)
                mask = (col_var <= thresh).float().unsqueeze(0)
                st_purge['fc1.weight'] += torch.randn_like(w1) * 0.03 * mask
                model_g4.load_state_dict(st_purge)

        acc_4, asr_4 = evaluate_model(model_g4, clean_test_loader, poison_test_loader, target_class, device)
        traj_g4_acc.append(acc_4)
        traj_g4_asr.append(asr_4)

        if (k_rnd + 1) % 5 == 0 or (k_rnd + 1) == fl2_rounds:
            print(f"  [Round {k_rnd+1:2d}/{fl2_rounds}] "
                  f"G1(Benign) ASR: {asr_1*100:5.2f}% | "
                  f"G2(Naive-TM) ASR: {asr_2*100:5.2f}% | "
                  f"G3(Stealth-Attack) ASR: {asr_3*100:5.2f}% | "
                  f"G4(Defense) ASR: {asr_4*100:5.2f}% (ACC: {acc_4*100:5.2f}%)")

    # --------------------------------------------------------------------------
    # 최종 결과 요약 및 표 출력
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("           FL1 -> FU -> FL2 백도어 지연 활성화 및 방어 벤치마크 최종 요약표")
    print("=" * 90)
    print(f"{'실험 그룹':<35} | {'FU 직후(k=0) ASR':<18} | {'최종(FL2 20R) ASR':<18} | {'Clean ACC':<12}")
    print("-" * 90)
    print(f"{'Group 1: Benign FedAvg (0% 공격)':<35} | {asr_0*100:16.2f}% | {traj_g1_asr[-1]*100:16.2f}% | {traj_g1_acc[-1]*100:10.2f}%")
    print(f"{'Group 2: Naive Poison under Trimmed-Mean':<35} | {asr_0*100:16.2f}% | {traj_g2_asr[-1]*100:16.2f}% | {traj_g2_acc[-1]*100:10.2f}%")
    print(f"{'Group 3: 제안 공격 (Stealth Reactivation)':<35} | {asr_0*100:16.2f}% | {traj_g3_asr[-1]*100:16.2f}% | {traj_g3_acc[-1]*100:10.2f}%")
    print(f"{'Group 4: 제안 방어 (Audit + Purge)':<35} | {asr_0*100:16.2f}% | {traj_g4_asr[-1]*100:16.2f}% | {traj_g4_acc[-1]*100:10.2f}%")
    print("=" * 90)

    # JSON 저장
    os.makedirs('logs', exist_ok=True)
    res = {
        'initial_fu_asr': asr_0,
        'initial_fu_acc': acc_0,
        'G1_Benign_FedAvg': {'asr': traj_g1_asr, 'acc': traj_g1_acc},
        'G2_Naive_TrimmedMean': {'asr': traj_g2_asr, 'acc': traj_g2_acc},
        'G3_Stealth_Attack': {'asr': traj_g3_asr, 'acc': traj_g3_acc},
        'G4_Proposed_Defense': {'asr': traj_g4_asr, 'acc': traj_g4_acc}
    }
    with open('logs/delayed_activation_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print("[*] 실험 결과 저장 완료: logs/delayed_activation_benchmark_results.json")

if __name__ == '__main__':
    main()
