import torch.nn.functional as F
import torch.nn as nn


input_size = 224

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, use_1x1_convo=False):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=strides, padding=1,
        bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1_convo:
            self.c11 = nn.Sequential(nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=strides,
            bias=False),
                nn.BatchNorm2d(out_channels))

        else:
            self.c11 = None

    def forward(self, X):
        Z = F.relu(self.bn1(self.c1(X)))
        Z = self.bn2(self.c2(Z))
        if self.c11:
            X = self.c11(X)
        Z += X
        return F.relu(Z)


def resnet_block(in_channels, out_channels, num_residual_blocks, first_block=False):
    # Create as many residual blocks as necessary
    blocks = []
    for i in range(num_residual_blocks):
        if i == 0 and not first_block:
            blocks.append(
                ResidualBlock(in_channels, out_channels, use_1x1_convo=True, strides=2)
            )
        else:
            blocks.append(ResidualBlock(out_channels, out_channels))

    return blocks


class ResnetNN(nn.Module):
    def __init__(self,dropout=0.5,in_channels=1):
        super(ResnetNN, self).__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Body
        self.layer1 = nn.Sequential(*resnet_block(64, 64, 2, True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, 2))
        self.layer3 = nn.Sequential(*resnet_block(128, 256, 2))
        self.layer4 = nn.Sequential(*resnet_block(256, 512, 2))

        # Head
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(512, 1)

        # We don't use a sigmoid layer as we use BCE withlogit loss

    def forward(self, X):
        Z = self.stem(X)

        Z = self.layer1(Z)
        Z = self.layer2(Z)
        Z = self.layer3(Z)
        Z = self.layer4(Z)
        Z = self.avg(Z)
        # Remove last two dimensions (go from batch_size,512,1,1 => batch_size,512)
        Z = Z.squeeze(-1).squeeze(-1)
        Z = self.dropout(Z)
        Z = self.head(Z)

        return Z

    def freeze_backbone(self,verbose=False):
        '''
        Prevents the backbone from being modified by training, allows to focus training on the fully connected classification layer
        :param verbose: allows to check in console whether parameters were correctly frozen
        :return:
        '''
        for name,parameter in self.named_parameters():
            if parameter.requires_grad and "head" not in name:
                parameter.requires_grad = False
                if verbose:
                    print(f"Set require grad to {parameter.requires_grad} for: {name}")

    def predict_probas(self,img):
        pred = self(img)
        return F.sigmoid(pred).cpu()



