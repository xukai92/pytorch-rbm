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

rbm_2 = RBMBer(25, 5)

v_data = torch.randn(50, 25)

for _ in range(100):

    error = rbm_2.cd(v_data, 1, 5e-2, alpha=5e-1, lam=1e-4)

    print("Reconstruction error: %.3f" % (error.data[0]))

print("[rbm.test] done.")

print("[rbm.test] all tests pass.")
