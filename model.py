import torch.nn as nn
import timm

# Siamese Network with pretrained encoder
class SiameseNetwork(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # Remove the final classifier layer
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.encoder(x)
        return self.fc(output)

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)
