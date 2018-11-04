import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from net.senet import se_resnext50_32x4d, se_resnet50, senet154, se_resnet152
from net.densenet import densenet121, densenet161, densenet169, densenet201
import settings


class InclusiveNet(nn.Module):
    def __init__(self, backbone_name, num_classes=7172, pretrained=True):
        super(InclusiveNet, self).__init__()
        print('num_classes:', num_classes)
        if backbone_name in ['se_resnext50_32x4d', 'se_resnet50', 'senet154', 'se_resnet152']:
            self.backbone = eval(backbone_name)()
        elif backbone_name in ['resnet34', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']:
            self.backbone = eval(backbone_name)(pretrained=pretrained)
        else:
            raise ValueError('unsupported backbone name {}'.format(backbone_name))
        #self.backbone.last_linear = nn.Linear(2048, 7272) # for model convert

        if backbone_name == 'resnet34':
            ftr_num = 512
        elif backbone_name == 'densenet161':
            ftr_num = 2208
        elif backbone_name == 'densenet121':
            ftr_num = 1024
        elif backbone_name == 'densenet169':
            ftr_num = 1664
        elif backbone_name == 'densenet201':
            ftr_num = 1920
        else:
            ftr_num = 2048

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(ftr_num, num_classes)
        self.logit_num = nn.Linear(ftr_num, 1)
        self.name = 'InclusiveNet_' + backbone_name
    
    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, p=0.4)
        x = x.view(x.size(0), -1)
        return self.logit(x), self.logit_num(x)
    
    def forward(self, x):
        x = self.backbone.features(x)
        return self.logits(x)

    def get_logit_params(self, lr):
        group1 = [self.logit, self.logit_num]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': lr}

        return [param_group1]


def create_model(args):
    num_classes = args.end_index - args.start_index

    if args.backbone == 'resnet34':
        ftr_num = 512
    elif args.backbone == 'densenet161':
        ftr_num = 2208
    elif args.backbone == 'densenet121':
        ftr_num = 1024
    elif args.backbone == 'densenet169':
        ftr_num = 1664
    elif args.backbone == 'densenet201':
        ftr_num = 1920
    else:
        ftr_num = 2048

    if args.init_ckp is not None:
        model = InclusiveNet(backbone_name=args.backbone, num_classes=args.init_num_classes)
        model.load_state_dict(torch.load(args.init_ckp))
        if args.init_num_classes != num_classes:
            model.logit = nn.Linear(ftr_num, num_classes)
            model.logit_num = nn.Linear(ftr_num, 1)
    else:
        model = InclusiveNet(backbone_name=args.backbone, num_classes=num_classes)

    sub_dir = '{}_{}_{}'.format(args.cls_type, args.start_index, args.end_index)

    model_file = os.path.join(settings.MODEL_DIR, model.name, sub_dir, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    
    return model, model_file


def test():
    pass

if __name__ == '__main__':
    test()
    #convert_model4()
