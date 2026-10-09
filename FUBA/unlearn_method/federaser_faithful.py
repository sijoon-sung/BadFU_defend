# -*- coding: utf-8 -*-
"""원 논문(Liu et al., FedEraser, IWQoS 2021)대로의 보정(calibration) 언러닝.

왜 따로 두나
  - 이 저장소의 unlearn_method/federaser.py 는 보정 대신 공격자가 그대로 참여하는 재학습 루프
    (FedEraser/utils_federaser.global_train_once) 를 부른다. 즉 "FedEraser" 이름이지만 재학습+재주입이다.
  - FedEraser/Fed_Unlearn_base.unlearning_step_once 는 step_length 를 새 보정 노름으로 잡아
    사실상 newGM + (newCM - newGM) = 보정 모델 평균이 된다(원식은 옛 노름).
  - 언러닝 전후 비교 실험에서는 "저장된 궤적만으로 요청자 기여를 빼는" 순수한 FU 가 필요하다.

원식 (층별)
    newGM_{t+1} = newGM_t + ||oldCM_t - oldGM_t|| * (newCM_t - newGM_t) / ||newCM_t - newGM_t||
  oldGM_t : 라운드 t 의 시작 모델 = 저장된 global_{t-1}
  oldCM_t : 저장된 라운드 t 로컬(요청자 제외)의 평균
  newCM_t : newGM_t 에서 남은 클라이언트가 자기 데이터로 짧게(ratio) 정직하게 학습한 모델의 평균
남은 클라이언트는 보정 중 공격하지 않는다. 보정은 언러닝 절차의 일부이지 새 학습이 아니기 때문이다.
"""
import math

import torch
import torch.nn as nn
import torch.optim as optim


def _avg(dicts):
    out = {}
    for k in dicts[0].keys():
        out[k] = torch.stack([d[k].float() for d in dicts], 0).mean(0)
    return out


def _calibrate(global_dict, loaders, keep, config, epochs, lr, max_batches):
    new_cms = []
    for k in keep:
        net = config.creat_cls_net().to(config.device)
        net.load_state_dict(global_dict)
        net.train()
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for i, (x, y) in enumerate(loaders[k]):
                x, y = x.to(config.device), y.to(config.device)
                opt.zero_grad()
                crit(net(x), y).backward()
                opt.step()
                if i + 1 >= max_batches:
                    break
        new_cms.append({kk: v.detach().cpu() for kk, v in net.state_dict().items()})
    return _avg(new_cms)


def fedEraser_faithful(global_dicts, local_dicts, loaders, target, config,
                       ratio=0.5, lr=0.005, max_batches=None):
    """global_dicts[r-1] = 라운드 r 끝의 글로벌(r=1..R), local_dicts[r-1] = 라운드 r 로컬 리스트(클라 인덱스 순).
    loaders[k] = 학습 때와 같은 분할의 DataLoader. 반환: 언러닝된 state_dict(CPU)."""
    R = len(local_dicts)
    K = len(local_dicts[0])
    keep = [k for k in range(K) if k != target and local_dicts[0][k] is not None]
    # FUBA 의 정상 로컬 학습은 1 에폭을 train_epoch 배치에서 끊는다. ratio 는 그 배치 수에 건다.
    mb = max_batches if max_batches is not None else max(1, math.ceil(ratio * (getattr(config, "train_epoch", 100) + 1)))

    # 1라운드: 초기 모델은 요청자 정보가 없으므로 요청자를 뺀 저장 로컬의 평균으로 시작한다.
    new_gm = _avg([local_dicts[0][k] for k in keep])
    for r in range(2, R + 1):
        old_gm = {k: v.float() for k, v in global_dicts[r - 2].items()}
        old_cm = _avg([local_dicts[r - 1][k] for k in keep])
        new_cm = _calibrate(new_gm, loaders, keep, config, 1, lr, mb)
        out = {}
        for layer in new_gm.keys():
            if not torch.is_floating_point(new_gm[layer]):
                out[layer] = new_cm[layer]
                continue
            old_up = old_cm[layer] - old_gm[layer]
            new_up = new_cm[layer] - new_gm[layer]
            n_new = torch.norm(new_up)
            if n_new == 0:
                out[layer] = new_gm[layer] + new_up
            else:
                out[layer] = new_gm[layer] + torch.norm(old_up) * new_up / n_new
        new_gm = out
    return new_gm
