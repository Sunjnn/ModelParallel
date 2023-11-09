from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def resnetToSequential(model):
    """make the model to sequential"""

    return nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                          *model.layer1,
                          *model.layer2,
                          *model.layer3,
                          *model.layer4,
                          model.avgpool, nn.Flatten(1), model.fc)


def getResnet18():
    return resnetToSequential(resnet18(weights=None))

def getResnet34():
    return resnetToSequential(resnet34(weights=None))

def getResnet50():
    return resnetToSequential(resnet50(weights=None))

def getResnet101():
    return resnetToSequential(resnet101(weights=None))

def getResnet152():
    return resnetToSequential(resnet152(weights=None))


if __name__ == "__main__":
    print(getResnet152())
