import torch.nn as nn
import torchvision.models as models

def build_model(num_classes, backbone='resnet50', pretrained=True, freeze_backbone=False):
    backbone = backbone.lower()
    if backbone.startswith('resnet'):
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError('Unsupported backbone: {}'.format(backbone))

    if freeze_backbone:
        for name, param in model.named_parameters():
            # only train the classifier layer
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

    return model
