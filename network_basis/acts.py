import torch
import torch.nn as nn
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.w = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        w_out = self.w(x)
        v_out = self.v(x)

        gate = v_out * torch.sigmoid(v_out)
        
        return w_out * gate
    
if __name__ == '__main__':
    x = torch.randn(8,512,20)
    swiglu = SwiGLU(x.shape[2],x.shape[2])
    out = swiglu(x)
    print(out.shape)
    