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
    CORblock_Z is a "computational region" analagous to a region of the visual cortex and performs some canonical computations: convolution, nonlinearity, and pooling
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


class CORnet_Z(nn.Module):

    """
    CORnet_Z is a computational model of the visual cortex comprising multiple CORblock_Z modules
    """

    def __init__(self, pretrained=False, feedback_connections='all', n_classes=10):
        super().__init__()
        self.regions = nn.ModuleDict({
            'V1': CORblock_Z(3, 64, kernel_size=7, stride=2),
            'V2': CORblock_Z(64, 128),
            'V4': CORblock_Z(128, 256),
            'IT': CORblock_Z(256, 512),
        })
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, n_classes)),
        ]))
        
        self.input_size = (1, 3, 224, 224)
        self.sizes = self.get_sizes()  # get sizes of input and output for each region

        if feedback_connections is None:  # vanilla CORnet-Z
            self.feedback_connections = {}
        elif feedback_connections == 'all':  # all possible feedback connections
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
        if pretrained:
            weights = torch.load('cornet_z-5c427c9c.pth')  # CORnet-Z (no feedback) trained on ImageNet
            for region_name in self.regions.keys():  # weights and biases only loaded for V1, V2, V4, IT
                self.regions[region_name].conv.weight = weights['state_dict']['module.' + region_name + '.conv.weight']
                self.regions[region_name].conv.bias = weights['state_dict']['module.' + region_name + '.conv.bias']

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


# class CORblock_S(nn.Module):

#     scale = 4  # scale of the bottleneck convolution channels

#     def __init__(self, in_channels, out_channels, times=1):
#         super().__init__()

#         self.times = times

#         self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
#         self.norm_skip = nn.BatchNorm2d(out_channels)

#         self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
#         self.nonlin1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size=3, stride=2, padding=1, bias=False)
#         self.nonlin2 = nn.ReLU(inplace=True)

#         self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
#         self.nonlin3 = nn.ReLU(inplace=True)

#         self.output = Identity()  # for an easy access to this block's output

#         # need BatchNorm for each time step for training to work well
#         for t in range(self.times):
#             setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
#             setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
#             setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

#     def forward(self, inp):
#         x = self.conv_input(inp)

#         for t in range(self.times):
#             if t == 0:
#                 skip = self.norm_skip(self.skip(x))
#                 self.conv2.stride = (2, 2)
#             else:
#                 skip = x
#                 self.conv2.stride = (1, 1)

#             x = self.conv1(x)
#             x = getattr(self, f'norm1_{t}')(x)
#             x = self.nonlin1(x)

#             x = self.conv2(x)
#             x = getattr(self, f'norm2_{t}')(x)
#             x = self.nonlin2(x)

#             x = self.conv3(x)
#             x = getattr(self, f'norm3_{t}')(x)

#             x += skip
#             x = self.nonlin3(x)
#             output = self.output(x)

#         return output


# def CORnet_S():
#     model = nn.Sequential(OrderedDict([
#         ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
#             ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
#             ('norm1', nn.BatchNorm2d(64)),
#             ('nonlin1', nn.ReLU(inplace=True)),
#             ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
#             ('norm2', nn.BatchNorm2d(64)),
#             ('nonlin2', nn.ReLU(inplace=True)),
#             ('output', Identity())
#         ]))),
#         ('V2', CORblock_S(64, 128, times=2)),
#         ('V4', CORblock_S(128, 256, times=4)),
#         ('IT', CORblock_S(256, 512, times=2)),
#         ('decoder', nn.Sequential(OrderedDict([
#             ('avgpool', nn.AdaptiveAvgPool2d(1)),
#             ('flatten', Flatten()),
#             ('linear', nn.Linear(512, 1000)),
#             ('output', Identity())
#         ])))
#     ]))

#     # weight initialization
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         # nn.Linear is missing here because I originally forgot 
#         # to add it during the training of this network
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()

#     return model
