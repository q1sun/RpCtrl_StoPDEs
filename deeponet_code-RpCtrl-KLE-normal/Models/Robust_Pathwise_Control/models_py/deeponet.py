"""
DeepONet implementations for robust-pathwise control of elliptic PDEs with KLE coefficients.
"""
import torch
import torch.nn as nn

from .module import StructureNN
from .fnn import FNN


class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=256,
                 activation='relu', initializer='Glorot normal'):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        self.initializer = initializer

        self.modus = self.__init_modules()
        self.params = self.__init_params()
        self.__initialize()

    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., self.branch_dim:]
        x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.trunk_depth + 1):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']

    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
                                self.activation, self.initializer)
        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth + 1):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params

    def __initialize(self):
        for i in range(1, self.trunk_depth + 1):
            self.weight_init_(self.modus['TrLinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['TrLinM{}'.format(i)].bias, 0)


class EfficientStandardDeepONet(nn.Module):
    '''DeepONet wrapper with a fixed coordinate grid cached as a buffer.'''
    def __init__(self, branch_dim, trunk_dim, img_size, branch_depth=2, trunk_depth=3, width=256,
                 activation='relu', initializer='Glorot normal'):
        super().__init__()
        self.img_size = img_size

        self.deeponet = DeepONet(
            branch_dim=branch_dim,
            trunk_dim=trunk_dim,
            branch_depth=branch_depth,
            trunk_depth=trunk_depth,
            width=width,
            activation=activation,
            initializer=initializer
        )
        self.register_buffer('grid', self._create_grid(img_size))

    def _create_grid(self, size):
        H, W = size
        x = torch.linspace(0, 1, H)
        y = torch.linspace(0, 1, W)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [H*W, 2]

    def forward(self, branch_input):
        batch_size = branch_input.size(0)
        H, W = self.img_size

        branch_flat = branch_input.view(batch_size, -1)
        branch_out = self.deeponet.modus['Branch'](branch_flat)

        trunk_out = self.deeponet.modus['TrLinM1'](self.grid)
        trunk_out = self.deeponet.modus['TrActM1'](trunk_out)
        for i in range(2, self.deeponet.trunk_depth + 1):
            trunk_out = self.deeponet.modus[f'TrLinM{i}'](trunk_out)
            trunk_out = self.deeponet.modus[f'TrActM{i}'](trunk_out)

        output = torch.matmul(branch_out, trunk_out.T)
        return output.view(batch_size, 1, H, W) + self.deeponet.params['bias']