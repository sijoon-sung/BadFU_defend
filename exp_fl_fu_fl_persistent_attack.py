"""
exp_fl_fu_fl_persistent_attack.py
================================================================================
FL -> FU -> FL 수명주기 하에서 백도어 비희석 비잔틴 공격 및 방어 실증 벤치마크
- 연구 배경: 
    BadFU(RAID 2025)는 FL1 -> FU에서 트리거가 폭발하지만, 후속 학습(FL2) 진행 시
    정상 클라이언트의 클린 그래디언트 누적으로 인해 백도어가 희석(Catastrophic Forgetting)됨.
- 공격 목표 (Anti-Dilution Byzantine Attack):
    1) 저활성/영공간 좌표계(Inactive Coordinates) 마스킹
    2) 비잔틴 집계(Trimmed Mean / Median) 회피형 제약 최적화 (Constrained Projection)
    를 결합하여 후속 FL2(20 라운드) 내내 ASR 85~90%+를 유지.
- 방어 목표 (Dual-Track Defense):
    1) 시계열 방향 누적 감사 (Temporal Directional Audit)를 통해 잠행성 비잔틴 에이전트 적발 및 격리
    2) 스토캐스틱 서브스페이스 퍼지 (Stochastic Subspace Purge)를 통해 영공간 백도어 완전 소거
- 비교 그룹:
    [Group 1] Vanilla BadFU (후속 FL2 FedAvg, 0% 추가 공격): 자연 희석 측정
    [Group 2] Naive Poisoning under Trimmed-Mean: 비잔틴 방어에 의한 차단 및 희석
    [Group 3] Anti-Dilution Byzantine Attack (Trimmed-Mean 방어 하 지속 공격): ASR 보존 검증
    [Group 4] Proposed Defense (시계열 감사 + 영공간 퍼지): 공격 탐지 및 백도어 박멸
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
    각 파라미터 텐서의 좌표별 Trimmed Mean 집계 (비잔틴 강건 집계 알고리즘)
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
    pre_rounds = 10
    fu_rounds = 10
    post_rounds = 20
    local_epochs = 2
    batch_size = 64
    lr = 0.01
    target_class = 0

    print("=" * 84)
    print("      FL -> FU -> FL 비희석 비잔틴 공격 및 방어 실증 벤치마크")
    print(f"  Clients: {num_clients} | FL1: {pre_rounds}R | FU: {fu_rounds}R | FL2: {post_rounds}R | Target Class: {target_class}")
    print("=" * 84)

    # 1. 데이터 준비
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

    # Client 0 데이터 분할: Clean, Backdoor (D_bd), Camouflage (D_c)
    n_bd, n_c = 500, 500
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

    # Retained Data (Client 0에서 D_c 제거된 언러닝 데이터셋: D_clean + D_bd 잔존)
    unlearned_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs], dim=0)
    unlearned_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts], dim=0)
    datasets_unlearned = [CustomTensorDataset(unlearned_c0_imgs, unlearned_c0_tgts)] + [
        CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
    ]
    sizes_unlearned = [len(ds) for ds in datasets_unlearned]

    # Clean Client Data (Client 0 완전 정화 시)
    datasets_retain = [CustomTensorDataset(c0_clean_imgs, c0_clean_tgts)] + [
        CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
    ]
    sizes_retain = [len(ds) for ds in datasets_retain]

    # --------------------------------------------------------------------------
    # 단계 1: FL_1 초기 학습 (10 라운드) -> z^- 모델 생성
    # --------------------------------------------------------------------------
    print("\n[*] [단계 1] FL_1 초기 연합 학습 진행 (10 라운드)...")
    global_model = SimpleNN().to(device)
    for r in range(pre_rounds):
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
    # 단계 2: FU 언러닝 (D_c 삭제 후 재학습) -> z_0^U 모델 생성 (백도어 1차 폭발)
    # --------------------------------------------------------------------------
    print(f"\n[*] [단계 2] FU 언러닝 수행 (Right to be Forgotten: D_c 삭제 후 {fu_rounds} 라운드)...")
    model_unlearned_init = SimpleNN().to(device)
    for r in range(fu_rounds):
        local_states = []
        for cid in range(num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_unlearned_init.state_dict())
            st = train_local(lm, datasets_unlearned[cid], local_epochs, lr, batch_size, device)
            local_states.append(st)
        fed_avg(model_unlearned_init, local_states, sizes_unlearned)

    acc_fu, asr_fu = evaluate_model(model_unlearned_init, clean_test_loader, poison_test_loader, target_class, device)
    print(f"    => FU 완료 (z_0^U): Clean ACC = {acc_fu*100:.2f}% | Post-ASR (1차 폭발) = {asr_fu*100:.2f}% (백도어 전면 활성화!)")

    # --------------------------------------------------------------------------
    # 단계 3: 후속 FL_2 진행 (20 라운드) - 4개 그룹 심층 비교
    # --------------------------------------------------------------------------
    print("\n[*] [단계 3] 후속 FL_2 (20 라운드) 벤치마크 평가 시작...")

    model_g1 = copy.deepcopy(model_unlearned_init)
    model_g2 = copy.deepcopy(model_unlearned_init)
    model_g3 = copy.deepcopy(model_unlearned_init)
    model_g4 = copy.deepcopy(model_unlearned_init)

    traj_g1_asr, traj_g1_acc = [asr_fu], [acc_fu]
    traj_g2_asr, traj_g2_acc = [asr_fu], [acc_fu]
    traj_g3_asr, traj_g3_acc = [asr_fu], [acc_fu]
    traj_g4_asr, traj_g4_acc = [asr_fu], [acc_fu]

    client_temporal_history = [[] for _ in range(num_clients)]
    isolated_clients = set()

    # 영공간/저활성 좌표계 마스크 식별 (클린 클라이언트들의 그래디언트 변동 하위 25%)
    ref_st = global_model.state_dict()
    clean_grad_accum = {k: torch.zeros_like(v) for k, v in ref_st.items()}
    for cid in range(1, num_clients):
        lm = SimpleNN().to(device)
        lm.load_state_dict(ref_st)
        st_cln = train_local(lm, datasets_retain[cid], epochs=1, lr=lr, batch_size=batch_size, device=device)
        for k in clean_grad_accum.keys():
            clean_grad_accum[k] += torch.abs(st_cln[k] - ref_st[k])
    
    inactive_masks = {}
    for k, v in clean_grad_accum.items():
        if 'weight' in k:
            flat_v = v.view(-1)
            thresh = torch.quantile(flat_v, 0.25)
            mask = (v <= thresh).float()
            inactive_masks[k] = mask
        else:
            inactive_masks[k] = torch.zeros_like(v)

    for k_rnd in range(post_rounds):
        # ----------------------------------------------------
        # [Group 1] Vanilla BadFU (0% Attack, FedAvg) -> 자연 희석
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
        # [Group 2] Naive Poisoning under Trimmed-Mean -> 비잔틴 차단
        # ----------------------------------------------------
        states_g2 = []
        lm0 = SimpleNN().to(device)
        lm0.load_state_dict(model_g2.state_dict())
        st0 = train_local(lm0, datasets_unlearned[0], local_epochs, lr=0.03, batch_size=batch_size, device=device)
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
        # [Group 3] Anti-Dilution Byzantine Attack under Trimmed-Mean
        # ----------------------------------------------------
        states_g3 = []
        clean_states_g3 = []
        for cid in range(1, num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g3.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            clean_states_g3.append(st)

        lm_adv = SimpleNN().to(device)
        lm_adv.load_state_dict(model_g3.state_dict())
        st_adv_raw = train_local(lm_adv, datasets_unlearned[0], local_epochs, lr=0.02, batch_size=batch_size, device=device)
        
        ref_st3 = model_g3.state_dict()
        st_adv_crafted = copy.deepcopy(st_adv_raw)
        for key in ref_st3.keys():
            cln_tensors = torch.stack([cs[key] for cs in clean_states_g3], dim=0)
            cln_mean = cln_tensors.mean(dim=0)
            cln_std = cln_tensors.std(dim=0) + 1e-6
            lower_bound = cln_mean - 1.2 * cln_std
            upper_bound = cln_mean + 1.2 * cln_std
            mask = inactive_masks.get(key, torch.zeros_like(st_adv_raw[key]))
            constrained = torch.clamp(st_adv_raw[key], lower_bound, upper_bound)
            st_adv_crafted[key] = (1 - mask) * constrained + mask * st_adv_raw[key]

        states_g3.append(st_adv_crafted)
        states_g3.extend(clean_states_g3)
        trimmed_mean_aggregate(model_g3, states_g3, trim_ratio=0.2)
        acc_3, asr_3 = evaluate_model(model_g3, clean_test_loader, poison_test_loader, target_class, device)
        traj_g3_acc.append(acc_3)
        traj_g3_asr.append(asr_3)

        # ----------------------------------------------------
        # [Group 4] Proposed Defense (시계열 방향 감사 + 영공간 정화)
        # ----------------------------------------------------
        clean_states_g4 = []
        for cid in range(1, num_clients):
            lm = SimpleNN().to(device)
            lm.load_state_dict(model_g4.state_dict())
            st = train_local(lm, datasets_retain[cid], local_epochs, lr, batch_size, device)
            clean_states_g4.append(st)

        lm_adv4 = SimpleNN().to(device)
        lm_adv4.load_state_dict(model_g4.state_dict())
        st_adv_raw4 = train_local(lm_adv4, datasets_unlearned[0], local_epochs, lr=0.02, batch_size=batch_size, device=device)

        ref_st4 = model_g4.state_dict()
        st_adv_crafted4 = copy.deepcopy(st_adv_raw4)
        for key in ref_st4.keys():
            cln_tensors = torch.stack([cs[key] for cs in clean_states_g4], dim=0)
            cln_mean = cln_tensors.mean(dim=0)
            cln_std = cln_tensors.std(dim=0) + 1e-6
            lower_bound = cln_mean - 1.2 * cln_std
            upper_bound = cln_mean + 1.2 * cln_std
            mask = inactive_masks.get(key, torch.zeros_like(st_adv_raw4[key]))
            constrained = torch.clamp(st_adv_raw4[key], lower_bound, upper_bound)
            st_adv_crafted4[key] = (1 - mask) * constrained + mask * st_adv_raw4[key]

        states_g4_all = [st_adv_crafted4] + clean_states_g4

        # 1) 시계열 방향 누적 감사
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
                if persistence_score > 0.82:
                    isolated_clients.add(cid)
                    print(f"      [🚨 DEFENSE ALERT: Round {k_rnd+1}] Client {cid} 악의적 지속 편향 적발! (Persistence: {persistence_score:.4f} > 0.82) -> 영구 격리")
                    continue

            valid_states_g4.append(states_g4_all[cid])
            valid_sizes_g4.append(sizes_unlearned[cid] if cid == 0 else sizes_retain[cid])

        if len(valid_states_g4) > 0:
            fed_avg(model_g4, valid_states_g4, valid_sizes_g4)

        # 2) 악성 노드 격리 후 저활성 영공간 스토캐스틱 정화 (Stochastic Subspace Purge)
        if 0 in isolated_clients:
            with torch.no_grad():
                st_purge = model_g4.state_dict()
                for key in st_purge.keys():
                    if 'weight' in key and key in inactive_masks:
                        mask = inactive_masks[key]
                        noise = torch.randn_like(st_purge[key]) * 0.04 * mask
                        st_purge[key] += noise
                model_g4.load_state_dict(st_purge)

        acc_4, asr_4 = evaluate_model(model_g4, clean_test_loader, poison_test_loader, target_class, device)
        traj_g4_acc.append(acc_4)
        traj_g4_asr.append(asr_4)

        if (k_rnd + 1) % 5 == 0 or (k_rnd + 1) == post_rounds:
            print(f"  [Round {k_rnd+1:2d}/{post_rounds}] "
                  f"G1(FedAvg) ASR: {asr_1*100:5.2f}% | "
                  f"G2(Naive-TM) ASR: {asr_2*100:5.2f}% | "
                  f"G3(AntiDil-Attack) ASR: {asr_3*100:5.2f}% | "
                  f"G4(Defense) ASR: {asr_4*100:5.2f}% (ACC: {acc_4*100:5.2f}%)")

    # --------------------------------------------------------------------------
    # 최종 결과 요약 및 표 출력
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("           FL -> FU -> FL 수명주기 벤치마크 최종 평가 결과 요약표")
    print("=" * 90)
    print(f"{'실험 그룹':<35} | {'초기(FU 직후) ASR':<18} | {'최종(FL2 20R) ASR':<18} | {'Clean ACC':<12}")
    print("-" * 90)
    print(f"{'Group 1: Vanilla BadFU (FedAvg)':<35} | {asr_fu*100:16.2f}% | {traj_g1_asr[-1]*100:16.2f}% | {traj_g1_acc[-1]*100:10.2f}%")
    print(f"{'Group 2: Naive Poison under Trimmed-Mean':<35} | {asr_fu*100:16.2f}% | {traj_g2_asr[-1]*100:16.2f}% | {traj_g2_acc[-1]*100:10.2f}%")
    print(f"{'Group 3: 제안 공격 (Anti-Dilution Byzantine)':<35} | {asr_fu*100:16.2f}% | {traj_g3_asr[-1]*100:16.2f}% | {traj_g3_acc[-1]*100:10.2f}%")
    print(f"{'Group 4: 제안 방어 (시계열 감사+영공간 퍼지)':<35} | {asr_fu*100:16.2f}% | {traj_g4_asr[-1]*100:16.2f}% | {traj_g4_acc[-1]*100:10.2f}%")
    print("=" * 90)

    # JSON 저장
    os.makedirs('logs', exist_ok=True)
    res = {
        'initial_fu_asr': asr_fu,
        'initial_fu_acc': acc_fu,
        'G1_Vanilla_FedAvg': {'asr': traj_g1_asr, 'acc': traj_g1_acc},
        'G2_Naive_TrimmedMean': {'asr': traj_g2_asr, 'acc': traj_g2_acc},
        'G3_AntiDilution_Attack': {'asr': traj_g3_asr, 'acc': traj_g3_acc},
        'G4_Proposed_Defense': {'asr': traj_g4_asr, 'acc': traj_g4_acc}
    }
    with open('logs/fl_fu_fl_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print("[*] 실험 결과 저장 완료: logs/fl_fu_fl_benchmark_results.json")

if __name__ == '__main__':
    main()
