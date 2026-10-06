# BadFU / UBA-Inf: Federated Unlearning Backdoor Attack

## Introduction
This repository contains the implementation of study on **federated unlearning-enabled backdoor attacks**.
It demonstrates how an attacker can implant a dormant backdoor and then use **unlearning requests** to activate it, revealing critical vulnerabilities in federated learning.
We provide examples runs on cifar10 with ResNet18.

## How to Run
1. Prepare data  
This script will download and preprocess CIFAR-10; prepare both backdoor and camouflage samples:
```
bash prepare_data.sh
```
2. Run experiment
```
python badfu.py
```

## Environment Requirements

This project was tested under the following environment:

- Python 3.8+
- PyTorch 2.4.1 + CUDA 11.8
- torchvision 0.19.1
- numpy ≥ 1.22
- Pillow ≥ 8.2
- tqdm ≥ 4.64

## BadFU

A preprint version is available on arXiv:

https://arxiv.org/abs/2508.15541

## Acknowledgment


This project is partially based on the UBA-Inf implementation, which provided the initial data preprocessing：
Original source: https://github.com/Huangzirui1206/UBA-Inf
```
@inproceedings{huang2024ubainf,
  author    = {Zirui Huang and Yunlong Mao and Sheng Zhong},
  title     = {{UBA-Inf}: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage},
  booktitle = {Proceedings of the 33rd USENIX Security Symposium (USENIX Security '24)},
  year      = {2024},
  address   = {Philadelphia, PA},
  pages     = {4211--4228},
  url       = {https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui},
  publisher = {USENIX Association},
  month     = aug
}
```


