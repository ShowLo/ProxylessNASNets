# -*- coding: UTF-8 -*-

'''
ProxylessNASNets From <ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware>, arXiv:1812.00332.
Ref: https://github.com/mit-han-lab/proxylessnas
     
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num

class Bottleneck(nn.Module):
    '''
    The basic unit of ProxylessNASNets, Inverted Residuals and Linear Bottlenecks proposed in MobileNetV2
    '''
    def __init__(self, in_channels_num, exp_ratio, out_channels_num, kernel_size, stride, use_residual, BN_momentum, BN_eps):
        '''
        exp_ratio: exp_size=in_channels_num * exp_ratio, the number of channels in the middle stage of the block
        use_residual: True or False -- use residual link or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        
        self.use_residual = use_residual
        exp_size = round(in_channels_num * exp_ratio)

        if exp_size == in_channels_num:
            # Without expansion, the first pointwise convolution is omitted
            self.conv = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=kernel_size, stride=stride, 
                          padding=(kernel_size-1)//2, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps),
                nn.ReLU6(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum, eps=BN_eps))]))
                            if use_residual else nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum, eps=BN_eps)
            )
        else:
            # With expansion
            self.conv = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps),
                nn.ReLU6(inplace=True),
                # Depthwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride, 
                          padding=(kernel_size-1)//2, groups=exp_size, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps),
                nn.ReLU6(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum, eps=BN_eps))]))
                            if use_residual else nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum, eps=BN_eps)
            )

    def forward(self, x):
        if self.use_residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


class ProxylessNASNets(nn.Module):
    '''
    
    '''
    def __init__(self, mode='gpu', num_classes=1000, input_size=224, width_multiplier=1.0, BN_momentum=0.1, BN_eps=1e-3, zero_gamma=False):
        '''
        configs: setting of the model
        mode: gpu, cpu, mobile or mobile_14
        '''
        super(ProxylessNASNets, self).__init__()

        assert mode in ['gpu', 'cpu', 'mobile']
        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100, Tiny-ImageNet or Downsampled ImageNet
            s = 1

        # setting of the model
        if mode == 'gpu':
            # Configuration of a ProxylessNASNet-GPU Model
            configs = [
                #kernel_size, exp_ratio, out_channels_num, use_residual, stride
                [3, 1, 24, False, 1],
                [5, 3, 32, False, s],
                [7, 3, 56, False, 2],
                [3, 3, 56, True, 1],
                [7, 6, 112, False, 2],
                [5, 3, 112, True, 1],
                [5, 6, 128, False, 1],
                [3, 3, 128, True, 1],
                [5, 3, 128, True, 1],
                [7, 6, 256, False, 2],
                [7, 6, 256, True, 1],
                [7, 6, 256, True, 1],
                [5, 6, 256, True, 1],
                [7, 6, 432, False, 1]
            ]
            first_channels_num = 40
            last_channels_num = 1728
        elif mode == 'cpu':
            # Configuration of a ProxylessNASNet-CPU Model
            configs = [
                #kernel_size, exp_ratio, out_channels_num, use_residual, stride
                [3, 1, 24, False, 1],
                [3, 6, 32, False, s],
                [3, 3, 32, True, 1],
                [3, 3, 32, True, 1],
                [3, 3, 32, True, 1],
                [3, 6, 48, False, 2],
                [3, 3, 48, True, 1],
                [3, 3, 48, True, 1],
                [5, 3, 48, True, 1],
                [3, 6, 88, False, 2],
                [3, 3, 88, True, 1],
                [5, 6, 104, False, 1],
                [3, 3, 104, True, 1],
                [3, 3, 104, True, 1],
                [3, 3, 104, True, 1],
                [5, 6, 216, False, 2],
                [5, 3, 216, True, 1],
                [5, 3, 216, True, 1],
                [3, 3, 216, True, 1],
                [5, 6, 360, False, 1]
            ]
            first_channels_num = 40
            last_channels_num = 1432
        elif mode == 'mobile':
            # Configuration of a ProxylessNASNet-Mobile Model
            configs = [
                #kernel_size, exp_ratio, out_channels_num, use_residual, stride
                [3, 1, 16, False, 1],
                [5, 3, 32, False, s],
                [3, 3, 32, True, 1],
                [7, 3, 40, False, 2],
                [3, 3, 40, True, 1],
                [5, 3, 40, True, 1],
                [5, 3, 40, True, 1],
                [7, 6, 80, False, 2],
                [5, 3, 80, True, 1],
                [5, 3, 80, True, 1],
                [5, 3, 80, True, 1],
                [5, 6, 96, False, 1],
                [5, 3, 96, True, 1],
                [5, 3, 96, True, 1],
                [5, 3, 96, True, 1],
                [7, 6, 192, False, 2],
                [7, 6, 192, True, 1],
                [7, 3, 192, True, 1],
                [7, 3, 192, True, 1],
                [7, 6, 320, False, 1]
            ]
            first_channels_num = 32
            last_channels_num = 1280

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum, eps=BN_eps),
            nn.ReLU6(inplace=True)
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_ratio, out_channels_num, use_residual, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(in_channels_num=input_channels_num, exp_ratio=exp_ratio, out_channels_num=output_channels_num, kernel_size=kernel_size, stride=stride, use_residual=use_residual, BN_momentum=BN_momentum, BN_eps=BN_eps))
            input_channels_num = output_channels_num
        
        # the last stage
        feature_mix_layer = nn.Sequential(
                nn.Conv2d(in_channels=input_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_channels_num, momentum=BN_momentum, eps=BN_eps),
                nn.ReLU6()
            )
        feature_extraction_layers.append(feature_mix_layer)
        feature_extraction_layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*feature_extraction_layers)

        ########################################################################################################################
        # Classification part
        self.classifier = nn.Sequential(
            nn.Linear(last_channels_num, num_classes)
        )

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
	            if hasattr(m, 'lastBN'):
	                nn.init.constant_(m.lastBN.weight, 0.0)

if __name__ == "__main__":
    import argparse
    from torchsummaryX import summary
    parser = argparse.ArgumentParser(description='width multiplier')
    parser.add_argument('--mode', type=str, default='gpu')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--wm', type=float, default=1.0)
    args = parser.parse_args()

    model = ProxylessNASNets(mode=args.mode, num_classes=args.num_classes, input_size=args.input_size, width_multiplier=args.wm)
    model.eval()
    summary(model, torch.zeros((1, 3, args.input_size, args.input_size)))
    print('ProxylessNASNet-%s-%.2f with input size %d and output %d classes' % (args.mode, args.wm, args.input_size, args.num_classes))
