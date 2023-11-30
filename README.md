# computervision

# CNN 모델 설계 튜토리얼

## 데이터 다운로드

### fashionMNIST
```python
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

# FashionMNIST 데이터 다운로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
fashion_trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_testset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

### QMNIST
```python
import torchvision.transforms as transforms
from torchvision.datasets import QMNIST

# QMNIST 데이터 다운로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
qmnist_trainset = QMNIST(root='./data', what='train', download=True, transform=transform)
qmnist_testset = QMNIST(root='./data', what='test', download=True, transform=transform)
```

## 데이터 로드

### FashionMNIST
```python
fashionMNIST_trainset = datasets.fashionMNIST(root='./data', split='train', download=True, transform=transform)
fashionMNIST_testset = datasets.fashionMNIST(root='./data', split='test', download=True, transform=transform)
```
### QMNIST
```python
QMNIST_trainset = datasets.QMNIST(root='./data', split='train', download=True, transform=transform)
QMNIST_testset = datasets.QMNIST(root='./data', split='test', download=True, transform=transform)
```
## FashionMNIST
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

# FashionMNIST 데이터 로드
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
fashion_trainset = FashionMNIST(root=음. 미리 학습된 모델 (pre-trained model)을 사용하기
- 하이퍼파라미터 튜닝: 학습률, 배치 크기, 레이어 수 등의 하이퍼파라미터를 조절하여 모델의 성능을 높이기
- 더 깊은 학습: 더 많은 에폭 동안 모델을 학습하거나, 더 많은 데이터를 사용하여 더 많은 패턴을 학습하게 하기
```
