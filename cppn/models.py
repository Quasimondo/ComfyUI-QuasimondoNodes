import torch.nn as nn
import torch
from functools import partial

def weights_init(m, mean=0.0, std=1.0, bias_mean=0.0, bias_std=1.0, zero_bias = True, generator=None):
    print("weights_init",mean, std, bias_mean, bias_std, zero_bias)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std, generator=generator)
        if m.bias is not None:
            if zero_bias:
                m.bias.data.fill_(0)
            else:
                m.bias.data.normal_(bias_mean, bias_std, generator=generator)

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class Modulo(nn.Module):
    def forward(self, input):
        return torch.fmod(input,1.0)


class CPPN(nn.Module):
    def __init__(self, dim_z, dim_c, ch, layer_count=3, first_activation="Tanh", middle_activations="Tanh", last_activation="Sigmoid",mean=0.0, std=1.0, bias_mean=0.0, bias_std=1.0, zero_bias=True, generator = None):
        super(CPPN, self).__init__()
        
        self.generator = generator
        
        self.l_z = nn.Linear(dim_z, ch)
        self.l_x = nn.Linear(1, ch, bias=False)
        self.l_y = nn.Linear(1, ch, bias=False)
        self.l_r = nn.Linear(1, ch, bias=False)

        activations = {"Tanh":nn.Tanh,"Sigmoid":nn.Sigmoid, "Sin":Sin,"Softplus":nn.Softplus, "ReLU":nn.ReLU, "ELU": nn.ELU, "Modulo": Modulo }


        layers = [activations[first_activation]()]
        
        for i in range(layer_count):
            layers.append(nn.Linear(ch, ch))
            layers.append(activations[middle_activations]())
                  
        layers.append(nn.Linear(ch, dim_c))
        layers.append(activations[last_activation]())
        

        self.ln_seq = nn.Sequential(*layers)

        self._initialize(mean, std, bias_mean, bias_std, zero_bias)

    def _initialize(self, mean=0.0, std=1.0, bias_mean=0.0, bias_std=1.0, zero_bias=True):
        init_fn = partial(weights_init,mean=mean, std=std, bias_mean=bias_mean, bias_std=bias_std, zero_bias=zero_bias, generator=self.generator)
        self.apply(init_fn)
        
    def forward(self, z, x, y, r):
        u = self.l_z(z) + self.l_x(x) + self.l_y(y) + self.l_r(r)
        out = self.ln_seq(u)
        return out
