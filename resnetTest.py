import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # Remove the fully connected layer and average pooling
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        
    def forward(self, x):
        return self.resnet50(x)

# Initialize the ResNet-50 backbone
resnet50_backbone = ResNet50Backbone(pretrained=True)

# Test the backbone with a random input tensor
input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
output_features = resnet50_backbone(input_tensor)
print("Output features shape:", output_features)
