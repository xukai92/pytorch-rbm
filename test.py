"""
Test script for RBM in PyTorch.
"""


from __future__ import print_function
from rbm import *

import torch


print("[rbm.test] testing starts!")

print("[rbm.test] testing RBMBase...")

rbm_1 = RBMBase(10, 10)

print("[rbm.test] done.")

print("[rbm.test] testing RBMBer...")

rbm_2 = RBMBer(10, 10)
rbm_2.cd(torch.rand(2, 10), 1, 1e-3, 1e-2, 5e-1)

print("[rbm.test] done.")

print("[rbm.test] all tests pass.")
