import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    # generalized MLP class for both
    # generator and discriminator nets

    def __init__(self, ip_size, units, activations=[], bn=True):
        # constructor
        super().__init__()

        # track the parameters
        self.units = units
        self.bn = bn

        u = ip_size
        for idx, unit in enumerate(self.units):
            setattr(self, 'layer' + str(idx + 1), nn.Linear(u, unit))
            if idx != len(self.units) - 1 and self.bn:
                setattr(self, 'bn' + str(idx + 1), nn.BatchNorm1d(unit, momentum=0.8))
            u = unit

        if len(activations) == 0:
            self.activations = [ F.relu ] * len(self.units)
        else:
            self.activations = activations

    def forward(self, x):
        # forward pass
        
        for idx, _ in enumerate(self.units):
            x = self.activations[idx](getattr(self, 'layer' + str(idx + 1))(x))
            if idx != len(self.units) - 1 and self.bn:
                x = getattr(self, 'bn' + str(idx + 1))(x)

        return x
