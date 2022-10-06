#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 08:40:20 2022

@author: alitaghibakhshi
"""

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
a = torch.tensor([1., 2., 3.]).float().to(device)
b = torch.tensor([12., 22., 32.]).float().to(device)

out = a*b
print('device = ', device, '; out = ', out)

with open('test.txt', 'w') as f:
    f.write('device = ' + str(device) + '; out = ' + str(out))