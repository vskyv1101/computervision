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
### FashionMNIST 개선방안
```
- 다양한 모델 시도 : 다양한 신경망 아키텍처를 시도하기. 간단한 모델부터 복잡한 모델까지 다양한 구조를 실험하기
- 모델 복잡도 증가 : 모델의 용량을 늘려 성능을 향상시킬 수 있음. 레이어를 추가하거나 각 레이어의 뉴런 수를 조절하여 모델의 표현력을 높이기
- 정규화 기법 사용 : Batch Normalization이나 Dropout과 같은 정규화 기법을 도입하여 모델의 안정성을 향상시키기
- 데이터 증강 : 이미지 데이터에 대해 다양한 변형을 적용하여 데이터를 증강하기
- 전이 학습 : 다른 이미지 분류 작업에서 미리 학습된 모델을 가져와 Fine-tuning하면 초기 학습 속도가 향상되고 성능이 향상될 수 있음. torchvision.models에서 제공하는 사전 훈련된 모델들을 사용하기
- 하이퍼파라미터 튜닝 : 학습률, 배치 크기, epoch 수 등의 하이퍼파라미터를 조절하여 모델의 성능을 향상시키기
- 다양한 최적화 알고리즘 사용: SGD 이외의 다양한 최적화 알고리즘을 시도하여 어떤 알고리즘이 가장 효과적인지 확인하기 ex) ADAM, Rmsprop
```

### QMNIST 개선방안
```
- 모델 복잡도 증가 : 모델의 복잡도를 높여 성능을 향상시킬 수 있음. 레이어를 추가하거나 각 레이어의 뉴런 수를 조절하기
- 학습률 조정 : 적절한 학습률을 선택하는 것이 중요함. 학습률이 너무 작으면 학습이 느려질 수 있고, 학습률이 너무 크면 발산할 수 있음.
- 데이터 증강 : 학습 데이터를 늘리기 위해 데이터 증강을 사용할 수 있음. 이미지 회전, 이동, 크기 조정 등을 통해 다양한 시점에서의 데이터를 학습하여 일반화 성능을 향상시키기
- 하이퍼파라미터 튜닝 : 학습률, 배치 크기, 레이어 수 등의 하이퍼파라미터를 조절하여 모델의 성능을 높이기
- 더 많은 epoch으로 훈련 : 모델이 더 많은 데이터를 보고 더 많은 학습을 할 수 있도록 epoch 수를 늘리기. 단, 너무 많은 epoch를 사용하면 과적합(overfitting)이 발생할 수 있음
```

### QMNIST, FashionMNIST 의 평가방법
```
- 모델 테스트 데이터에 대한 예측 수행 : 모델이 훈련되고 저장된 후, 테스트 데이터에 대한 예측을 수행함. 모델이 각 이미지에 대해 예측한 클래스를 얻음.
- 예측과 실제 레이블 비교 : 모델의 예측 결과와 실제 레이블을 비교하여 정확도 및 다른 평가 지표를 계산함.
- 평가 지표 계산 : 주로 사용되는 평가 지표 중 하나는 accuracy임. 정확도는 전체 예측 중 올바르게 분류된 예측의 비율로 계산됨.
```
