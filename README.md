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
fashion_trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_testset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 생성
fashion_trainloader = DataLoader(fashion_trainset, batch_size=64, shuffle=True)
fashion_testloader = DataLoader(fashion_testset, batch_size=64, shuffle=False)

# 모델 설계
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화
model = SimpleCNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 5
for epoch in range(epochs):
    model.train()
    for data in fashion_trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 테스트 함수 정의
def test_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')

# FashionMNIST 테스트
test_model(model, fashion_testloader, criterion)

# 모델 저장
torch.save(model.state_dict(), 'simple_cnn_model_fashionmnist.pth')

# 저장된 모델 불러와서 테스트
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('simple_cnn_model_fashionmnist.pth'))
loaded_model.eval()

# FashionMNIST 테스트
test_model(loaded_model, fashion_testloader, criterion)
```

## QMNIST
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import QMNIST

# QMNIST 데이터 로드
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
qmnist_trainset = QMNIST(root='./data', what='train', download=True, transform=transform)
qmnist_testset = QMNIST(root='./data', what='test', download=True, transform=transform)

# DataLoader 생성
qmnist_trainloader = DataLoader(qmnist_trainset, batch_size=64, shuffle=True)
qmnist_testloader = DataLoader(qmnist_testset, batch_size=64, shuffle=False)

# 모델 설계
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화
model = SimpleCNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 5
for epoch in range(epochs):
    model.train()
    for data in qmnist_trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 테스트 함수 정의
def test_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')

# QMNIST 테스트
test_model(model, qmnist_testloader, criterion)

# 모델 저장
torch.save(model.state_dict(), 'simple_cnn_model_qmnist.pth')

# 저장된 모델 불러와서 테스트
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('simple_cnn_model_qmnist.pth'))
loaded_model.eval()

# QMNIST 테스트
test_model(loaded_model, qmnist_testloader, criterion)
```
