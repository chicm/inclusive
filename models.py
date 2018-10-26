import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import net.resnet
from net.senet import se_resnext50_32x4d, se_resnet50
import settings


def create_backbone_model(pretrained, num_classes=7272):
    if pretrained:
        basenet = se_resnext50_32x4d()
        basenet.last_linear = nn.Linear(2048, num_classes)
    else:
        basenet = se_resnext50_32x4d(num_classes=num_classes, pretrained=None)

    basenet.num_ftrs = 2048
    basenet.name = 'se_resnext50_32x4d'
    return basenet

def create_model(backbone_name, pretrained, num_classes, load_backbone_weights=False):
    if backbone_name == 'se_resnext50_32x4d':
        backbone = create_backbone_model(pretrained)
        fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(backbone.num_ftrs, num_classes))
        backbone.last_linear = fc
    elif backbone_name == 'resnet34':
        backbone, _ = create_pretrained_resnet(34, 7272)
    else:
        raise ValueError('unsupported backbone name {}'.format(backbone_name))
    backbone.name = backbone_name

    if pretrained:
        model_file = os.path.join(settings.MODEL_DIR, 'backbone', backbone.name, 'pretrained', 'best.pth')
    else:
        model_file = os.path.join(settings.MODEL_DIR, 'backbone', backbone.name, 'scratch', 'best.pth')
    
    if load_backbone_weights:
        print('loading {}...'.format(model_file))
        backbone.load_state_dict(torch.load(model_file))

    return backbone
    '''

    if backbone == 'se_resnext50_32x4d'

    if model_name == 'resnet' and pretrained:
        model, _ = create_pretrained_resnet(layers, num_classes)
        model.name = 'resnet_{}_{}'.format(layers, num_classes)
    elif model_name == 'resnet' and not pretrained:
        model = create_resnet_model(layers, num_classes)
        model.name = 'resnet_scratch_{}_{}'.format(layers, num_classes)

    return model
    '''

def create_resnet_model(layers, num_classes):
    if layers not in [18, 32, 34, 50, 101, 152]:
        raise ValueError('Wrong resnet layers')

    return eval('net.resnet.resnet'+str(layers))(pretrained=False, num_classes=num_classes)
    
def create_pretrained_resnet(layers, num_classes):
    print('create_pretrained_resnet', layers)
    if layers == 34:
        model, bottom_channels = resnet34(pretrained=True), 512
    elif layers == 18:
        model, bottom_channels = resnet18(pretrained=True), 512
    elif layers == 50:
        model, bottom_channels = resnet50(pretrained=True), 2048
    elif layers == 101:
        model, bottom_channels = resnet101(pretrained=True), 2048
    elif layers == 152:
        model, bottom_channels = resnet152(pretrained=True), 2048
    else:
        raise NotImplementedError('only 34, 50, 101, 152 version of Resnet are implemented')

    model.num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.num_ftrs, num_classes)) 

    return model, bottom_channels

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.size())
        return x

class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2

class AttentionResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes=100, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(AttentionResNet, self).__init__()
        self.name = 'AttentionResNet_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_pretrained_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout2d(p=self.dropout_2d),
            nn.Linear(bottom_channel_nr, 100)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        x = self.encoder2(x) #; print('e2:', e2.size())
        x = self.encoder3(x) #; print('e3:', e3.size())
        x = self.encoder4(x) #; print('e4:', e4.size())
        x = self.encoder5(x) #; print('e5:', x.size())
        x = F.dropout2d(x, p=self.dropout_2d)
        x = self.avgpool(x) #; print('out:', x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    model = AttentionResNet(50).cuda()
    model.freeze_bn()
    inputs = torch.randn(2,3,128,128).cuda()
    out, _ = model(inputs)
    #print(model)
    print(out.size()) #, cls_taret.size())
    #print(out)

def test2():
    model = create_model('se_resnext50_32x4d', pretrained=False, num_classes=100, load_backbone_weights=False).cuda()
    x = torch.randn(2,3,128,128).cuda()
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    #test()
    test2()