import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models


num_clients = 5 
local_epochs = 5 
batch_size = 32 
num_rounds = 40 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_classes = 10
dominant_ratio = 0.7

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from PIL import Image


data_path = "./record/badnet_dataset/pert_result.pt"
data_dict = torch.load(data_path)

bd_train_dict = data_dict['bd_train']
bd_ind = list(bd_train_dict['bd_data_container']['data_dict'].keys())

bd_cv_ind = list(bd_train_dict['cv_data_container']['data_dict'].keys())

poison_indicator = bd_train_dict['poison_indicator']
(poison_indicator == 1).sum()

bd_test_dict = data_dict['bd_test']

cv_pert_dict = data_dict['cv_pert']
cv_data_container = cv_pert_dict['data_dict']
cv_ind = list(cv_data_container.keys())

remove_indices = set(bd_ind + cv_ind)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

all_indices = range(len(train_dataset))

kept_indices = [i for i in all_indices if i not in remove_indices]

clean_data = [train_dataset.data[i] for i in kept_indices]
clean_targets = [train_dataset.targets[i] for i in kept_indices]

clean_data = torch.stack([transform(Image.fromarray(img)) for img in clean_data])
clean_targets = torch.tensor(clean_targets)

class CustomCIFARDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

clean_dataset = CustomCIFARDataset(clean_data, clean_targets)

dominant_client_map = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 4,
    9: 4
}


targets = clean_dataset.targets.numpy() if hasattr(clean_dataset.targets, "numpy") else clean_dataset.targets

num_clients = 5
client_clean_data = [[] for _ in range(num_clients)]

all_classes = np.unique(targets)
for c in all_classes:
    class_indices = np.where(targets == c)[0]
    np.random.shuffle(class_indices)
    dominant_client_id = dominant_client_map[c]
    split_point = int(len(class_indices) * dominant_ratio)
    dominant_indices = class_indices[:split_point]
    for idx in dominant_indices:
        client_clean_data[dominant_client_id].append(clean_dataset[idx])

    remaining_indices = class_indices[split_point:]
    non_dominant_clients = [cid for cid in range(num_clients) if cid != dominant_client_id]
    chunk_size = len(remaining_indices) // (num_clients - 1)
    for i, cid in enumerate(non_dominant_clients):
        start = i * chunk_size
        end = (i+1)*chunk_size if i < (len(non_dominant_clients)-1) else len(remaining_indices)
        sub_indices = remaining_indices[start:end]
        for idx in sub_indices:
            client_clean_data[cid].append(clean_dataset[idx])

for cid, samples in enumerate(client_clean_data):
    print(f"Client {cid} clean data count: {len(samples)}")

import torch
from torch.utils.data import Dataset
from PIL import Image

class BackdoorDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        info = self.data_dict[key]
        image_path = info['path']
        label = info['other_info'][0]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def create_bd_dataset(data_dict, transform=None):
    return BackdoorDataset(data_dict, transform=transform)


bd_dataset = create_bd_dataset(bd_train_dict['bd_data_container']['data_dict'], transform=transform)
bd_test_dataset = create_bd_dataset(bd_test_dict['bd_data_container']['data_dict'], transform=transform)
cv_dataset = create_bd_dataset(cv_data_container, transform=transform)

import copy
from torch.utils.data import ConcatDataset

client_datasets = client_clean_data
clean_client_datasets = copy.deepcopy(client_datasets)
bd_client_datasets = copy.deepcopy(client_datasets)
cv_client_datasets = copy.deepcopy(client_datasets)
ul_client_datasets = copy.deepcopy(client_datasets)

bd_client_datasets[0] = ConcatDataset([client_datasets[0], bd_dataset])
cv_client_datasets[0] = ConcatDataset([client_datasets[0], bd_dataset, cv_dataset])
ul_client_datasets[0] = ConcatDataset([client_datasets[0], bd_dataset])
ul_client_datasets.append(ConcatDataset([client_datasets[0], cv_dataset]))
#ul_client_datasets.append(cv_dataset)

clean_client_loaders = [
    DataLoader(client_data, batch_size=64, shuffle=True)
    for client_data in clean_client_datasets
]

bd_client_loaders = [
    DataLoader(client_data, batch_size=64, shuffle=True)
    for client_data in bd_client_datasets
]

cv_client_loaders = [
    DataLoader(client_data, batch_size=64, shuffle=True)
    for client_data in cv_client_datasets
]

ul_client_loaders = [
    DataLoader(client_data, batch_size=64, shuffle=True)
    for client_data in ul_client_datasets
]

print("Client clean loaders:")
for i, loader in enumerate(clean_client_loaders):
    print(f"  Client {i}: {len(loader.dataset)} samples")
    
print("Client bd loaders:")
for i, loader in enumerate(bd_client_loaders):
    print(f"  Client {i}: {len(loader.dataset)} samples")
    
print("Client cv loaders:")
for i, loader in enumerate(cv_client_loaders):
    print(f"  Client {i}: {len(loader.dataset)} samples")
    
print("Client ul loaders:")
for i, loader in enumerate(ul_client_loaders):
    print(f"  Client {i}: {len(loader.dataset)} samples")
    
class ResNet18ForCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18ForCIFAR10, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ResNet18ForCIFAR10client(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18ForCIFAR10client, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



def local_train(model, dataloader, epochs, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def fed_avg(global_model, client_state_dicts, client_data_counts):
    global_state_dict = global_model.state_dict()
    aggregated_state_dict = {}
    
    for key in global_state_dict.keys():
        aggregated_state_dict[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
    
    total_samples = sum(client_data_counts)
    
    for client_state, n_samples in zip(client_state_dicts, client_data_counts):
        weight = n_samples / total_samples
        for key in global_state_dict.keys():
            weighted_param = client_state[key].float() * weight
            aggregated_state_dict[key] += weighted_param
    
    for key in global_state_dict.keys():
        if global_state_dict[key].dtype == torch.long:
            aggregated_state_dict[key] = torch.round(aggregated_state_dict[key]).to(torch.long)
    
    global_model.load_state_dict(aggregated_state_dict)
    return global_model

def fed_avg_inner(global_model, client_state_dicts):
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])
    for client_state in client_state_dicts:
        for key in global_state_dict.keys():
            global_state_dict[key] += client_state[key]
            
    return global_state_dict



def evaluate_model(model, device, clean_loader, poison_loader, n_classes=10, backdoor_target=0):
    model.eval()
    total = 0
    correct = 0
    class_correct = [0 for _ in range(n_classes)]
    class_total = [0 for _ in range(n_classes)]
    with torch.no_grad():
        for data, target in clean_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            for i in range(target.size(0)):
                lbl = target[i].item()
                class_total[lbl] += 1
                if predicted[i].item() == lbl:
                    class_correct[lbl] += 1
    overall_acc = 100.0 * correct / total
    class_accuracies = [100.0 * c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]

    total_poison = 0
    correct_poison = 0
    with torch.no_grad():
        for data, target in poison_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total_poison += target.size(0)
            correct_poison += (predicted == backdoor_target).sum().item()
    asr = 100.0 * correct_poison / total_poison

    return overall_acc, class_accuracies, asr

bd_test_loader = DataLoader(bd_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

client_data_counts = [len(loader.dataset) for loader in clean_client_loaders]
print("clients' data count:", client_data_counts)

global_model_clean = ResNet18ForCIFAR10().to(device)

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    client_state_dicts = []
    local_models = []
    for c_id in range(len(clean_client_loaders)):
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_clean.state_dict())
        train_loader = clean_client_loaders[c_id]

        local_state = local_train(local_model, train_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(local_model)
        
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")

    global_model_clean = fed_avg(global_model_clean, client_state_dicts, client_data_counts)

    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_clean, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")
    
client_data_counts = [len(loader.dataset) for loader in bd_client_loaders]
print("clients' data count:", client_data_counts)

global_model_bd = ResNet18ForCIFAR10().to(device)

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    client_state_dicts = []
    local_models = []
    
    for c_id in range(len(bd_client_loaders)):
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_bd.state_dict())
        train_loader = bd_client_loaders[c_id]
        local_state = local_train(local_model, train_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(local_model)
        
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")
    
    global_model_bd = fed_avg(global_model_bd, client_state_dicts, client_data_counts)
    
    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_bd, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")

old_CMs_b = []
old_GMs_b = [] 


client_data_counts = [len(loader.dataset) for loader in cv_client_loaders]
print("clients' data count:", client_data_counts)

global_model_cv = ResNet18ForCIFAR10().to(device)

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    client_state_dicts = []
    local_models = []
    
    for c_id in range(len(cv_client_loaders)):
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_cv.state_dict())
        train_loader = cv_client_loaders[c_id]
        
        local_state = local_train(local_model, train_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(copy.deepcopy(local_model))
        
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")
    
    global_model_cv = fed_avg(global_model_cv, client_state_dicts, client_data_counts)
    old_GMs_b.append(copy.deepcopy(global_model_cv))
    old_CMs_b.append(copy.deepcopy(local_models))
    
    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_cv, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")
    
old_CMs = []
old_GMs = []

client_data_counts = [len(loader.dataset) for loader in ul_client_loaders]
print("clients' data count:", client_data_counts)

global_model_for_ul = ResNet18ForCIFAR10().to(device)

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    client_state_dicts = []
    client_bd_state_dicts = []
    local_models = []
    
    for c_id in range(len(ul_client_loaders)):
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_for_ul.state_dict())
        train_loader = ul_client_loaders[c_id]
        
        local_state = local_train(local_model, train_loader, local_epochs, device)
            
        client_state_dicts.append(local_state)
        if c_id != 5:
            local_models.append(copy.deepcopy(local_model))
            
            
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")

    global_model_for_ul = fed_avg(global_model_for_ul, client_state_dicts, client_data_counts)
    old_GMs.append(copy.deepcopy(global_model_for_ul))
    old_CMs.append(copy.deepcopy(local_models))
    
    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_for_ul, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")
    
def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    old_param_update = dict()
    new_param_update = dict()
    
    new_global_model_state = global_model_after_forget.state_dict()
    old_global_model_state = global_model_before_forget.state_dict()
    
    return_model_state = dict()
    
    assert len(old_client_models) == len(new_client_models)
    
    for layer in old_global_model_state.keys():
        old_param_update[layer] = torch.zeros_like(old_global_model_state[layer], dtype=torch.float32)
        new_param_update[layer] = torch.zeros_like(old_global_model_state[layer], dtype=torch.float32)
        
        return_model_state[layer] = torch.zeros_like(old_global_model_state[layer], dtype=torch.float32)
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer].float()
            new_param_update[layer] += new_client_models[ii].state_dict()[layer].float()

        old_param_update[layer] = old_param_update[layer] / float(ii+1)
        new_param_update[layer] = new_param_update[layer] / float(ii+1)

        old_param_update[layer] = old_param_update[layer] - old_global_model_state[layer].float()
        new_param_update[layer] = new_param_update[layer] - new_global_model_state[layer].float()

        step_length = torch.norm(old_param_update[layer])
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])

        return_model_state[layer] = new_global_model_state[layer].float() + step_length*step_direction

        if new_global_model_state[layer].dtype == torch.long:
            return_model_state[layer] = torch.round(return_model_state[layer]).to(torch.long)
        else:
            return_model_state[layer] = return_model_state[layer].to(new_global_model_state[layer].dtype)
    
    return_global_model = copy.deepcopy(global_model_after_forget)
    return_global_model.load_state_dict(return_model_state)
    
    return return_global_model

client_data_counts = [len(loader.dataset) for loader in ul_client_loaders]
print("clients' data count:", client_data_counts)

global_model_ul = old_GMs[0]

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    if r % 2 == 1:
        continue
    
    client_state_dicts = []
    local_models = []
    for c_id in range(len(bd_client_loaders)):
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_ul.state_dict())
        train_loader = bd_client_loaders[c_id]
        local_state = local_train(local_model, train_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(local_model)
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")
    
    global_model_ul = unlearning_step_once(old_CMs[r], local_models, old_GMs[r+1], global_model_ul)
    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_ul, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")
    
client_data_counts = [len(loader.dataset) for loader in cv_client_loaders]
print("clients' data count:", client_data_counts)

global_model_ul_b = old_GMs_b[0]

for r in range(num_rounds):
    print(f"==== {r+1} communication round ====")
    if r % 2 == 1:
        continue
    
    client_state_dicts = []
    local_models = []
    for c_id in range(len(cv_client_loaders)):
        if c_id == 1:
            continue
        local_model = ResNet18ForCIFAR10client().to(device)
        local_model.load_state_dict(global_model_ul_b.state_dict())
        train_loader = cv_client_loaders[c_id]
        local_state = local_train(local_model, train_loader, local_epochs, device)
        client_state_dicts.append(local_state)
        local_models.append(local_model)
        local_clean_acc, local_class_acc, local_asr = evaluate_model(
            local_model, device, test_loader, bd_test_loader, backdoor_target=0)
        print(f"Client {c_id} local model test accuracy: {local_clean_acc:.2f}%")
        print(f"Backdoor Attack Success Rate (ASR): {local_asr:.2f}%")
    
    CMs = old_CMs_b[r].copy()
    CMs.pop(1)
    global_model_ul_b = unlearning_step_once(CMs, local_models, old_GMs_b[r+1], global_model_ul_b)
    global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
        global_model_ul_b, device, test_loader, bd_test_loader, backdoor_target=0)
    print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
    print("\n")
    
    
global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_clean, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")

global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_bd, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")

global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_cv, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")

global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_for_ul, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")

global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_ul, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")

global_clean_acc, global_clean_class_acc, global_clean_asr = evaluate_model(
    global_model_ul_b, device, test_loader, bd_test_loader, backdoor_target=0)
print(f"Global model after round {r+1} test accuracy: {global_clean_acc:.2f}%")
print(f"Backdoor Attack Success Rate (ASR): {global_clean_asr:.2f}%")
print("\n")