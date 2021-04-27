import math
from collections import OrderedDict
import torch
from torch import nn


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class CORblock_Z(nn.Module):

    """
    CORblock_Z is a "computational region" of CORnet-Z
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        return x


class CORblock_S(nn.Module):

    """
    CORblock_S is a "computational region" of CORnet-S
    """

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)

        return x


class CORnet(nn.Module):

    """
    CORnet is a computational model of the visual cortex comprising multiple CORblock_Z/CORblock_S modules
    """

    def __init__(self, pretrained=False, architecture='CORnet-Z', feedback_connections='all', n_classes=10):
        
        """
        if pretrained=True, parameters from ImageNet pretraining will be loaded for all layers except the decoder
        architecture can be 'CORnet-Z' or 'CORnet-S'
        feedback connections can be {}, 'all', or a custom dictionary
        n_classes can be any integer corresponding to the number of output classes
        """
        
        super().__init__()

        self.architecture = architecture

        # convolutional architecture
        if self.architecture == 'CORnet-Z':
            self.regions = nn.ModuleDict({
                'V1': CORblock_Z(3, 64, kernel_size=7, stride=2),
                'V2': CORblock_Z(64, 128),
                'V4': CORblock_Z(128, 256),
                'IT': CORblock_Z(256, 512),
            })
            self.weights_file = 'cornet_z-5c427c9c.pth' 
        elif self.architecture == 'CORnet-S':
            self.regions = nn.ModuleDict({
                'V1': nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('norm1', nn.BatchNorm2d(64)),
                    ('nonlin1', nn.ReLU(inplace=True)),
                    ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                    ('norm2', nn.BatchNorm2d(64)),
                    ('nonlin2', nn.ReLU(inplace=True)),
                ])),
                'V2': CORblock_S(64, 128, times=2),
                'V4': CORblock_S(128, 256, times=4),
                'IT': CORblock_S(256, 512, times=2),
            })
            self.weights_file = 'cornet_s-XXXX.pth'  # TODO update filename
        else:
            raise ValueError('only supports \'CORnet-Z\' and \'CORnet-S\'')

        # decoder head
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, n_classes)),
        ]))
        
        # sizes of input and output for each region
        self.input_size = (1, 3, 224, 224)
        self.sizes = self.get_sizes()

        # feedback
        if feedback_connections == 'all':  # all possible feedback connections
            self.feedback_connections = {  # the output of the `key` region is combined with the outputs of the `value` regions
                'input': ['V1', 'V2', 'V4', 'IT'],
                'V1': ['V2', 'V4', 'IT'],
                'V2': ['V4', 'IT'],
                'V4': ['IT'],
            }
        else:  # custom feedback connections
            self.feedback_connections = feedback_connections
        self.feedback = self.create_feedback_layers()

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:  # parameters from ImageNet training
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = torch.device(device)
            weights = torch.load(self.weights_file, map_location=device)
            
            if self.architecture == 'CORnet-Z':
                for region_name in self.regions.keys():  # weights and biases only loaded for V1, V2, V4, IT
                    self.regions[region_name].conv.weight = nn.Parameter(weights['state_dict']['module.' + region_name + '.conv.weight'])
                    self.regions[region_name].conv.bias = nn.Parameter(weights['state_dict']['module.' + region_name + '.conv.bias'])
            elif self.architecture == 'CORnet-S':
                self.weights_file = ''
                # TODO add support for loading pretrained weights for CORblock-S

                print('to do')
            else:
                raise ValueError('only supports \'CORnet-Z\' and \'CORnet-S\'')
            
    def create_feedback_layers(self):
        feedback = {}
        for earlier_region_name, later_region_names in self.feedback_connections.items():
            input_region_names = later_region_names + [earlier_region_name]
            in_channels = sum([self.sizes[region_name]['output'][1] for region_name in input_region_names])
            out_channels = self.sizes[earlier_region_name]['output'][1]
            feedback[earlier_region_name] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        feedback = nn.ModuleDict(feedback)
        return feedback

    def forward(self, image):
        if self.feedback_connections is None:
            output = self.feedforward(image, return_intermediate_output=False)
        else:
            all_region_names = list(self.regions.keys())
            order = ['input', *self.regions.keys()]

            output = self.feedforward(image, return_intermediate_output=True)

            for earlier_region_name, later_region_names in self.feedback_connections.items():
                input_region_names = later_region_names + [earlier_region_name]
                target_region_name = all_region_names[order.index(earlier_region_name)]

                out_size = self.sizes[earlier_region_name]['input'][-2:]
                feedback = torch.cat([nn.Upsample(size=out_size)(output[region_name]) for region_name in input_region_names], 1)
                
                output[target_region_name] = self.regions[target_region_name](self.feedback[earlier_region_name](feedback))

            output = output[list(self.regions.keys())[-1]]

        output = self.decoder(output)
        return output

    def feedforward(self, image, return_intermediate_output=False):
        if not return_intermediate_output:
            output = image
            for region in self.regions.values():
                output = region(output)
        else:
            output = {'input': image}
            for i_region, (region_name, region) in enumerate(self.regions.items()):
                if i_region == 0:
                    output[region_name] = region(image)
                else:
                    output[region_name] = region(output[previous_region_name])
                previous_region_name = region_name
        return output

    def get_sizes(self):
        input_size = self.input_size
        outputs = self.feedforward(torch.rand(input_size), return_intermediate_output=True)
        sizes = {}
        for region_name in list(outputs.keys()):
            sizes[region_name] = {
                'input': input_size,
                'output': outputs[region_name].size(),
            }
            input_size = sizes[region_name]['output']
        return sizes
