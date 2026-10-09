import torch
from config import Config
from .FedEraser.Fed_Unlearn_base import unlearning
from .utils import Arguments
from utils.comm_utils import make_client_split, make_client_loaders

def fedEraser(config:Config):
    # 주의: 이 래퍼는 보정(calibration) 대신 FedEraser/utils_federaser.global_train_once (공격자 참여 재학습)를 부른다.
    # 원 논문식 보정은 unlearn_method/federaser_faithful.py 를 쓴다.
    # 분할은 학습 때와 같게 (원본은 IID random_split(522) 을 다시 했다).
    trainloaders = make_client_loaders(make_client_split(config.trainset, config), config)

    testloader = torch.utils.data.DataLoader(config.testset, batch_size=32,
                                            shuffle=True, num_workers=0)
    Old_GMS = []
    Old_CMS = []
    for i in range(len(config.global_nets)):
        Old_GMS.append(config.global_nets[i])
    for i in range(len(config.local_nets)):
        Old_CMS.append([])
        for j in range(len(config.local_nets[i])):
            Old_CMS[-1].append(config.local_nets[i][j])
    return unlearning(Old_GMS,Old_CMS,trainloaders,testloader,Arguments(config),config)
