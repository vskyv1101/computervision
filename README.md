# computervision

# CNN 모델 설계 튜토리얼

## 데이터 로드
### SVHN
```python
import torchvision.datasets as datasets

svhn_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=...)
svhn_testset = datasets.SVHN(root='./data', split='test', download=True, transform=...)
