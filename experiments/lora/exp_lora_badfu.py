import os
import sys
import copy
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# -----------------------------------------------------------------------------
# 1. LoRA Layer & Model Definition
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (self.alpha / self.r) * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out

class SimpleLoRANN(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, num_classes=10, r=4, alpha=16):
        super().__init__()
        self.fc1 = LoRALinear(input_dim, hidden_dim, r=r, alpha=alpha)
        self.relu = nn.ReLU()
        self.fc2 = LoRALinear(hidden_dim, num_classes, r=r, alpha=alpha)

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
    c_correct, c_total = 0, 0
    for data, target in clean_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        c_correct += (pred == target).sum().item()
        c_total += target.size(0)
    acc = c_correct / c_total

    p_correct, p_total = 0, 0
    for data, _ in poison_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        p_correct += (pred == target_class).sum().item()
        p_total += data.size(0)
    asr = p_correct / p_total
    return acc, asr

def train_local_lora(model, dataset, epochs=1, lr=0.01, batch_size=64, device='cuda:0'):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
    optimizer = optim.SGD(lora_params, lr=lr)
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

# -----------------------------------------------------------------------------
# 2. Aggregation Functions
# -----------------------------------------------------------------------------
def aggregate_fedit(global_model, client_states, client_sizes):
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    
    for k in new_state.keys():
        if 'lora_' in k:
            new_state[k] = torch.zeros_like(new_state[k], dtype=torch.float)
            for state, sz in zip(client_states, client_sizes):
                new_state[k] += state[k].float() * (sz / total)
    
    global_model.load_state_dict(new_state)

def aggregate_fedex(global_model, client_states, client_sizes, r, alpha):
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    
    for layer_name in ['fc1', 'fc2']:
        delta_W_global = 0
        for state, sz in zip(client_states, client_sizes):
            A = state[f'{layer_name}.lora_A'].float()
            B = state[f'{layer_name}.lora_B'].float()
            delta_W = (alpha / r) * (B @ A)
            delta_W_global += delta_W * (sz / total)
            
        try:
            U, S, Vh = torch.linalg.svd(delta_W_global.cpu(), full_matrices=False)
            U, S = U.to(delta_W_global.device), S.to(delta_W_global.device)
            V_r = Vh[:r, :].to(delta_W_global.device)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(delta_W_global.double().cpu(), full_matrices=False)
            U, S = U.float().to(delta_W_global.device), S.float().to(delta_W_global.device)
            V_r = Vh[:r, :].float().to(delta_W_global.device)
        
        U_r = U[:, :r]
        S_r = torch.diag(S[:r])
        
        factor = math.sqrt(r / alpha)
        S_sqrt = torch.sqrt(S_r)
        
        B_new = U_r @ S_sqrt * factor
        A_new = S_sqrt @ V_r * factor
        
        actual_r = S.shape[0]
        if actual_r < r:
            pad_size = r - actual_r
            B_new = torch.cat([B_new, torch.zeros(B_new.shape[0], pad_size, device=B_new.device)], dim=1)
            A_new = torch.cat([A_new, torch.zeros(pad_size, A_new.shape[1], device=A_new.device)], dim=0)
        
        new_state[f'{layer_name}.lora_A'] = A_new
        new_state[f'{layer_name}.lora_B'] = B_new
        
    global_model.load_state_dict(new_state)

# -----------------------------------------------------------------------------
# 3. SOTA Unlearning Methods (SVD / Projection based)
# -----------------------------------------------------------------------------
def unlearn_w_svd(model):
    """
    SOTA Method 1: Weight SVD Truncation (W-SVD)
    방어자가 최종 레이어의 가중치 SVD를 계산하여, 가장 주도적인 특이값 1개를 
    제거함으로써 얕게 학습된 위장막(Camouflage)을 걷어내는 기법.
    """
    model_fu = copy.deepcopy(model)
    base_W = model_fu.fc2.weight.data
    A = model_fu.fc2.lora_A.data
    B = model_fu.fc2.lora_B.data
    r = A.size(0)
    alpha = r
    lora_W = (alpha / r) * (B @ A)
    total_W = base_W + lora_W
    
    U, S, Vh = torch.linalg.svd(total_W.float().cpu(), full_matrices=False)
    U, S, Vh = U.to(base_W.device), S.to(base_W.device), Vh.to(base_W.device)
    
    S_new = S.clone()
    S_new[0] = 0.0 # Top-1 특이값 제거
    purified_W = U @ torch.diag(S_new) @ Vh
    
    model_fu.fc2.weight.data = purified_W
    model_fu.fc2.lora_A.data.zero_()
    model_fu.fc2.lora_B.data.zero_()
    
    return model_fu

def unlearn_act_svd(model, val_loader, device):
    """
    SOTA Method 2: Activation Spectral Signature Projection (Act-SVD)
    클린 검증 데이터의 활성화 공분산을 구하고, 주성분(PC) 방향을 
    가중치에서 직교 사영(Projection)하여 제거하는 기법.
    """
    model_fu = copy.deepcopy(model)
    model_fu.eval()
    
    h_list = []
    with torch.no_grad():
        for d, _ in val_loader:
            d = d.to(device).view(d.size(0), -1)
            h = model_fu.relu(model_fu.fc1(d))
            h_list.append(h)
    
    H = torch.cat(h_list, dim=0)
    H_mean = H.mean(dim=0, keepdim=True)
    H_centered = H - H_mean
    
    U, S, Vh = torch.linalg.svd(H_centered.cpu(), full_matrices=False)
    v1 = Vh[0, :].to(device).unsqueeze(1) # Top-1 eigenvector (hidden_dim, 1)
    
    base_W = model_fu.fc2.weight.data
    A = model_fu.fc2.lora_A.data
    B = model_fu.fc2.lora_B.data
    r = A.size(0)
    alpha = r
    
    total_W = base_W + (alpha / r) * (B @ A)
    proj = (total_W @ v1) @ v1.t() # (out_dim, hidden_dim)
    purified_W = total_W - proj
    
    model_fu.fc2.weight.data = purified_W
    model_fu.fc2.lora_A.data.zero_()
    model_fu.fc2.lora_B.data.zero_()
    
    return model_fu

# -----------------------------------------------------------------------------
# 4. Main Experiment
# -----------------------------------------------------------------------------
def run_single_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    non_target_mask = (test_targets != 0)
    bd_test_images = torch.stack([add_trigger(img) for img in test_images[non_target_mask]])
    bd_test_targets = test_targets[non_target_mask]
    poison_test_loader = DataLoader(CustomTensorDataset(bd_test_images, bd_test_targets), batch_size=512, shuffle=False)
    val_loader = DataLoader(CustomTensorDataset(test_images[:500], test_targets[:500]), batch_size=32, shuffle=True)

    num_clients = 5
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

    c0_imgs, c0_tgts = client_data[0], client_targets[0]
    n_bd, n_c = 1000, 1000
    perm = torch.randperm(len(c0_tgts))
    bd_idx = perm[:n_bd]
    c_idx = perm[n_bd:n_bd+n_c]
    cln_idx = perm[n_bd+n_c:]

    c0_clean_imgs, c0_clean_tgts = c0_imgs[cln_idx], c0_tgts[cln_idx]
    
    c0_bd_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in bd_idx])
    c0_bd_tgts = torch.full((n_bd,), 0, dtype=torch.long)
    
    c0_c_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in c_idx])
    c0_c_tgts = c0_tgts[c_idx]

    badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
    badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

    datasets_badfu = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
        CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, num_clients)
    ]
    sizes_badfu = [len(ds) for ds in datasets_badfu]

    def run_experiment(rank, agg_method):
        alpha = rank
        global_model = SimpleLoRANN(r=rank, alpha=alpha).to(device)
        
        rounds = 5
        for rnd in range(rounds):
            states = []
            for cid in range(num_clients):
                local_model = copy.deepcopy(global_model)
                epochs = 5 if cid == 0 else 2
                lr = 0.02 if cid == 0 else 0.01
                st = train_local_lora(local_model, datasets_badfu[cid], epochs=epochs, lr=lr, device=device)
                states.append(st)
                
            if agg_method == 'fedit':
                aggregate_fedit(global_model, states, sizes_badfu)
            else:
                aggregate_fedex(global_model, states, sizes_badfu, r=rank, alpha=alpha)
                
        acc_pre, asr_pre = evaluate_model(global_model, clean_test_loader, poison_test_loader, 0, device)
        
        # SOTA Unlearning 1: Weight SVD Truncation
        model_w_svd = unlearn_w_svd(global_model)
        acc_post_w, asr_post_w = evaluate_model(model_w_svd, clean_test_loader, poison_test_loader, 0, device)
        
        # SOTA Unlearning 2: Activation SVD Projection
        model_act_svd = unlearn_act_svd(global_model, val_loader, device)
        acc_post_act, asr_post_act = evaluate_model(model_act_svd, clean_test_loader, poison_test_loader, 0, device)
        
        return {
            'asr_pre': asr_pre,
            'asr_post_w': asr_post_w,
            'asr_post_act': asr_post_act
        }

    results = {}
    for r in [16, 2]:
        for m in ['fedex', 'fedit']:
            results[f"Rank_{r}_{m}"] = run_experiment(r, m)
            
    return results

def main():
    seeds = [42, 123, 999]
    all_results = []
    
    print("="*90)
    print("Starting FL SVD Bottleneck & SOTA Unlearning (W-SVD, Act-SVD) Verification")
    print("="*90)
    
    for seed in seeds:
        print(f">>> Running Seed: {seed}")
        res = run_single_seed(seed)
        all_results.append(res)
        
    ranks = [16, 2]
    methods = ['fedex', 'fedit']
    
    print("\n" + "="*90)
    print(f"{'Experiment Grid (ASR Pre vs Post-Unlearning)':^90}")
    print("="*90)
    print(f"{'Settings':<15} | {'ASR_pre':<15} | {'ASR_post (W-SVD)':<20} | {'ASR_post (Act-SVD)'}")
    print("-" * 90)
    
    for r in ranks:
        for m in methods:
            key = f"Rank_{r}_{m}"
            
            asr_pre_mean = np.mean([res[key]['asr_pre'] for res in all_results])
            asr_post_w_mean = np.mean([res[key]['asr_post_w'] for res in all_results])
            asr_post_act_mean = np.mean([res[key]['asr_post_act'] for res in all_results])
            
            print(f"R={r:2d}, {m.upper():<5} | {asr_pre_mean*100:6.2f}%         | {asr_post_w_mean*100:6.2f}%               | {asr_post_act_mean*100:6.2f}%")
    print("="*90)

if __name__ == '__main__':
    main()
