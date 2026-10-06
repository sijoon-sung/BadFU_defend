import torch, copy, json, random
import torch.nn as nn
import torch.optim as optim
from exp_fl_fu_fl_delayed_resurface import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Testing tuning on device:', device)

# Load data
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

# Build simple FL1 model (10 rounds)
torch.manual_seed(42)
global_model = SimpleNN().to(device)

dominant_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
client_data = [[] for _ in range(5)]
client_targets = [[] for _ in range(5)]
for c in range(10):
    c_indices = (train_targets == c).nonzero(as_tuple=True)[0].tolist()
    random.shuffle(c_indices)
    dom_client = dominant_map[c]
    split_pt = int(len(c_indices) * 0.7)
    client_data[dom_client].extend([train_images[i] for i in c_indices[:split_pt]])
    client_targets[dom_client].extend([train_targets[i] for i in c_indices[:split_pt]])
    non_dom = [cid for cid in range(5) if cid != dom_client]
    chunk = len(c_indices[split_pt:]) // len(non_dom)
    for i, cid in enumerate(non_dom):
        sub_idx = c_indices[split_pt:][i*chunk : (i+1)*chunk] if i < len(non_dom)-1 else c_indices[split_pt:][i*chunk:]
        client_data[cid].extend([train_images[k] for k in sub_idx])
        client_targets[cid].extend([train_targets[k] for k in sub_idx])

for cid in range(5):
    client_data[cid] = torch.stack(client_data[cid])
    client_targets[cid] = torch.tensor(client_targets[cid])

c0_imgs = client_data[0]
c0_tgts = client_targets[0]
n_bd, n_c = 500, 500
perm = torch.randperm(len(c0_tgts))
bd_idx = perm[:n_bd]
c_idx = perm[n_bd:n_bd+n_c]
cln_idx = perm[n_bd+n_c:]

c0_clean_imgs = c0_imgs[cln_idx]
c0_clean_tgts = c0_tgts[cln_idx]
c0_bd_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in bd_idx])
c0_bd_tgts = torch.full((n_bd,), 0, dtype=torch.long)
c0_c_imgs = torch.stack([add_trigger(c0_imgs[i]) for i in c_idx])
c0_c_tgts = c0_tgts[c_idx]

badfu_c0_imgs = torch.cat([c0_clean_imgs, c0_bd_imgs, c0_c_imgs], dim=0)
badfu_c0_tgts = torch.cat([c0_clean_tgts, c0_bd_tgts, c0_c_tgts], dim=0)

datasets_badfu = [CustomTensorDataset(badfu_c0_imgs, badfu_c0_tgts)] + [
    CustomTensorDataset(client_data[i], client_targets[i]) for i in range(1, 5)
]
sizes_badfu = [len(ds) for ds in datasets_badfu]

for r in range(10):
    states = []
    for cid in range(5):
        lm = SimpleNN().to(device)
        lm.load_state_dict(global_model.state_dict())
        st = train_local(lm, datasets_badfu[cid], 2, 0.01, 64, device)
        states.append(st)
    fed_avg(global_model, states, sizes_badfu)

acc_pre, asr_pre = evaluate_model(global_model, clean_test_loader, poison_test_loader, 0, device)
print(f"Pre-FL1: ACC={acc_pre*100:.2f}%, ASR={asr_pre*100:.2f}%")

# Superficial suppression at FU
model_fu = copy.deepcopy(global_model)
val_loader = DataLoader(CustomTensorDataset(test_images[:500], test_targets[:500]), batch_size=32, shuffle=True)
opt_purify = optim.SGD(model_fu.fc2.parameters(), lr=0.02) # ONLY suppress fc2
for _ in range(4):
    for d, t in val_loader:
        d, t = d.to(device), t.to(device)
        opt_purify.zero_grad()
        loss = nn.CrossEntropyLoss()(model_fu(d), t)
        loss.backward()
        opt_purify.step()

acc_0, asr_0 = evaluate_model(model_fu, clean_test_loader, poison_test_loader, 0, device)
print(f"Post-FU Checkpoint: ACC={acc_0*100:.2f}%, ASR={asr_0*100:.2f}%")
