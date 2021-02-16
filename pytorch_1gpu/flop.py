import torch

from torchvision.models import resnet50
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
from thop import profile
# model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))