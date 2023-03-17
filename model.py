# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence = nn.Sequential(nn.Flatten(),
                                      nn.Linear(6720, 512),
                                      nn.SiLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(512, 1))

    def forward(self, x):
        return self.sequence(x)

