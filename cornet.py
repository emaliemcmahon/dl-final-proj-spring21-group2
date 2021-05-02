from collections import OrderedDict
import torch
from torch import nn


def invert_dictionary(x):
    y = {}
    if x:
        for key, values in x.items():
            for value in values:
                if value not in y.keys():
                    y[value] = []
                y[value].append(key)
    return y


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class CORblock_Z(nn.Module):
    """
    CORblock_Z is a computational area of CORnet-Z
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
    CORblock_S is a computational area of CORnet-S
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
    CORnet is a computational model of the visual cortex. Here, we enhance CORnet by implementing feedback connections.
    """
    def __init__(self, architecture='CORnet-Z', n_classes=10, pretrained=True, feedback_connections={}, n_passes=1):
        """
        architecture: 'CORnet-Z' or 'CORnet-S'
        n_classes: number of output classes
        pretrained: parameters from ImageNet pretraining will be loaded for all areas
        n_passes: number of recurrent passes through the network
        """
        super().__init__()
        self.state_dict_files = {
            'CORnet-Z': 'cornet_z-5c427c9c.pth',
            'CORnet-S': 'cornet_s-1d3f7974.pth',
        }
        self.input_size = (1, 3, 224, 224)

        self.architecture = architecture
        self.n_classes = n_classes
        self.pretrained = pretrained

        if feedback_connections == 'all':
            feedback_connections = {
                'V1': ('V1',),
                'V2': ('V1', 'V2'),
                'V4': ('V1', 'V2', 'V4'),
                'IT': ('V1', 'V2', 'V4', 'IT'),
            }
        self.feedback_connections = feedback_connections
        self.inverted_feedback_connections = invert_dictionary(feedback_connections)
        self.n_passes = n_passes

        self.areas = self.create_areas(self.architecture)
        self.decoder = self.create_decoder(self.n_classes)
        self.sizes = self.compute_sizes(self.input_size)
        self.feedback = self.create_feedback(self.sizes, self.feedback_connections, self.inverted_feedback_connections)

        self.initialize_weights(self.pretrained, self.architecture, self.state_dict_files)

        self.sequence = ['input'] + list(self.areas.keys())

    @staticmethod
    def create_areas(architecture):
        """
        creates the convolutional architecture of the CORnet, exactly as in CORnet-Z and CORnet-S
        """
        if architecture == 'CORnet-Z':
            areas = nn.ModuleDict({
                'V1': CORblock_Z(3, 64, kernel_size=7, stride=2),
                'V2': CORblock_Z(64, 128),
                'V4': CORblock_Z(128, 256),
                'IT': CORblock_Z(256, 512),
            })
        elif architecture == 'CORnet-S':
            areas = nn.ModuleDict({
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
        else:
            raise ValueError('allowed architectures are \'CORnet-Z\' and \'CORnet-S\'')
        return areas

    @staticmethod
    def create_decoder(n_classes):
        """
        creates a decoder that is used to predict raw class scores for classification
        """
        decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, n_classes)),
        ]))
        return decoder

    @staticmethod
    def create_feedback(sizes, feedback_connections, inverted_feedback_connections):
        """
        creates feedback layers (depthwise convolutions) that mix feedback information with the feedforward input
        """
        feedback = {area_name: None for area_name in feedback_connections.keys()}
        for receiver_name, giver_names in inverted_feedback_connections.items():
            out_channels = sizes[receiver_name]['input'][1]
            in_channels = sum([sizes[area_name]['output'][1] for area_name in giver_names]) + out_channels
            feedback[receiver_name] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        feedback = nn.ModuleDict(feedback)
        return feedback

    def initialize_weights(self, pretrained, architecture, state_dict_files):
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
            params = torch.load(state_dict_files[architecture], map_location=device)['state_dict']

            new_params = {}
            for key, value in params.items():
                new_key = key.split('.')
                new_key[0] = 'areas'
                new_key = '.'.join(new_key)
                new_params[new_key] = value
            self.load_state_dict(new_params, strict=False)

    def feedforward(self, x, activations_to_return=()):
        """
        computes a single feedforward pass through the convolutional part of the CORnet, returning any specified activation maps
        """
        activation = {area_name: None for area_name in activations_to_return}
        if 'input' in activations_to_return:
            activation['input'] = x
        for area_name, area in self.areas.items():
            x = area(x)
            if area_name in activations_to_return:
                activation[area_name] = x
        if activations_to_return:
            return activation
        else:
            return x

    def forward(self, image):
        """
        full forward pass through the CORnet
        """
        if self.feedback_connections:
            activations_to_return = ['input'] + list(self.feedback_connections.keys())
            activation = self.feedforward(image, activations_to_return=activations_to_return)
            for _ in range(self.n_passes):
                for receiver_name, giver_names in self.inverted_feedback_connections.items():
                    input_names = giver_names + [self.sequence[self.sequence.index(receiver_name) - 1]]
                    out_dimensions = self.sizes[receiver_name]['input'][-2:]
                    feedback = torch.cat([nn.Upsample(size=out_dimensions)(activation[area_name]) for area_name in input_names], 1)
                    activation[receiver_name] = self.areas[receiver_name](self.feedback[receiver_name](feedback))
            activation = activation[self.sequence[-1]]
        else:
            activation = self.feedforward(image, activations_to_return=())
        activation = self.decoder(activation)
        return activation

    def compute_sizes(self, input_size):
        random_input = torch.rand(input_size)
        input_size = random_input.size()
        activations = self.feedforward(random_input, activations_to_return=self.areas.keys())
        sizes = {}
        for area_name in activations.keys():
            sizes[area_name] = {
                'input': input_size,
                'output': activations[area_name].size(),
            }
            input_size = sizes[area_name]['output']
        return sizes


if __name__ == '__main__':

    feedback_connections = {
        'V1': ('V1',),
        'V2': ('V1', 'V2'),
        'V4': ('V1', 'V2', 'V4'),
        'IT': ('V1', 'V2', 'V4', 'IT'),
    }

    torch.manual_seed(0)

    with torch.no_grad():
        model = CORnet(architecture='CORnet-S', n_classes=10, feedback_connections=feedback_connections, pretrained=True, n_passes=1)
        try:
            print(model(torch.rand(1, 3, 224, 224)))
            print('model passes check')
        except:
            print('model fails check')
