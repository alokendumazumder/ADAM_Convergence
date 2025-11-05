import torch.nn as nn
import torchvision.models as models
import torch
from vit.model import vit_b_16

class Model(nn.Module):
    def __init__(self, input_dim, num_layers, nodes, init_weights=None, add_bn=True, output_nodes=10, d=None):
        super().__init__()
        self.init_weights = init_weights
        if add_bn:
            model = [
                nn.Linear(input_dim, nodes),
                nn.BatchNorm1d(nodes, track_running_stats=False),
                nn.ReLU(True)
            ]

            for i in range(num_layers-1):
                model += [
                    nn.Linear(nodes, nodes),
                    nn.BatchNorm1d(nodes, track_running_stats=False),
                    nn.ReLU(True)
                ]
            model += [
                nn.Linear(nodes, output_nodes),
                nn.BatchNorm1d(output_nodes, track_running_stats=False),
                nn.ReLU(True)
            ]

        else:
            model = [
                nn.Linear(input_dim, nodes, bias=False),
                #nn.BatchNorm1d(nodes, track_running_stats=False),
                nn.ReLU(True)
            ]

            for i in range(num_layers - 1):
                model += [
                    nn.Linear(nodes, nodes, bias=False),
                    #nn.BatchNorm1d(nodes, track_running_stats=False),
                    nn.ReLU(True)
                ]
            model += [
                nn.Linear(nodes, output_nodes, bias=False),
                #nn.BatchNorm1d(output_nodes, track_running_stats=False),
                nn.ReLU(True)
            ]

        self.model = nn.Sequential(*model)
        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        return self.model(x)

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.04)
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif self.init_weights == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

class AE(nn.Module):
    def __init__(self, input_dim, init_weights=None):
        super().__init__()
        self.init_weights = init_weights
        model = [
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.Linear(256, 32),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.ReLU(True)
        ]

        model += [
            nn.Linear(32, 256),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.BatchNorm1d(input_dim, track_running_stats=False),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        return self.model(x)

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.2)
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif self.init_weights == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)


class Vgg9(nn.Module):
    def __init__(self, out_dim, init_weights=None, d=None):
        super().__init__()
        self.init_weights = init_weights
        self.num_layers = 9
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 'same', bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d(output_size=(4, 4)),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(4096, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, out_dim, bias=True)
        )

        if self.init_weights is not None:
            self.init_weights_()
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.linear(out)
        # print(out.shape)
        return out

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.045)  #0.045
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif self.init_weights == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_weights == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)
    

class LeNet(nn.Module):

    def __init__(self, out_dim, init_weights=None, d=None):
        super(LeNet, self).__init__()
        self.init_weights = init_weights
        self.num_layers = 5
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400, 120),  # in_features = 16 x5x5
            nn.BatchNorm1d(120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(True),
            nn.Linear(84, out_dim),
        )

        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        a1 = self.feature_extractor(x)
        a1 = torch.flatten(a1, 1)
        a2 = self.classifier(a1)
        return a2

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.1)
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                elif self.init_weights == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_weights == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)


class Resnet18(nn.Module):
    def __init__(self, out_dim, init_weights=None, d=None):
        super().__init__()
        self.init_weights = init_weights
        self.num_layers = 18
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, out_dim)
        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        x = self.resnet18(x)
        return x
    
    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.025) #0.045 for cifar10
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                elif self.init_weights == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_weights == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)


class Mobilenet(nn.Module):
    def __init__(self, out_dim, init_weights=None, d=None):
        super().__init__()
        self.init_weights = init_weights
        #self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.num_layers = 54
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, out_dim)
        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        return self.model(x)

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.1)  #0.1
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                elif self.init_weights == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_weights == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)


class EfficientnetB0(nn.Module):
    def __init__(self, out_dim, init_weights=None):
        super().__init__()
        self.init_weights = init_weights
        #self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, out_dim)
        if self.init_weights is not None:
            self.init_weights_()

    def forward(self, x):
        return self.model(x)

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if self.init_weights == 'normal':
                    nn.init.normal_(m.weight, 0, 0.1)
                elif self.init_weights == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                elif self.init_weights == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_weights == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_weights == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise Exception

                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)


def get_model(model_num, size, num_layers, nodes, init_weights="normal", device="cuda"):
    output_dim = 1000
    model = None
    if model_num == 0:
        model = Model(size, num_layers, nodes, "normal", output_nodes=output_dim).to(device)
    if model_num == 1:
        model = Vgg9(output_dim, init_weights).to(device)
    if model_num == 2:
        model = LeNet(output_dim, init_weights).to(device)
    if model_num == 3:
        model = Resnet18(output_dim, init_weights).to(device)
    if model_num == 4:
        model = Mobilenet(output_dim, init_weights).to(device)
    if model_num == 5:
        model = EfficientnetB0(output_dim, init_weights).to(device)
    if model_num == 6:
        model = vit_b_16().to(device)
    return model

def get_model_name(model_num):
    if model_num == 0:
        model = "linear"
    elif model_num == 1:
        model = "vgg"
    elif model_num == 2:
        model = "lenet"
    elif model_num == 3:
        model = "resnet"
    elif model_num == 4:
        model = "mobilenet"
    elif model_num == 5:
        model = "efficientnet"
    elif model_num == 6:
        model = "vit"
    else:
        raise ValueError
    return model
