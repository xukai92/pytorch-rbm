# PyTorch implementation of all kinds of RBMs.

## Usage

```python
from rbm import *

rbm_1 = RBMBer(10, 10)
rbm_2 = RBMGaussHid(10, 10)

data = ...

rbm_1.cd(data)
rbm_2.cd(data)
```

## Plan

- [x] RBM with Bernoulli units
- [x] RBM with Gaussian hidden units
- [ ] RBM with Gaussian visible and hidden units
- [ ] Allow customized mu and sigma
- [ ] Provide customized initialization for bias of visible units
  - Ref: Section 24.8 of Hinton12

## Reference

Hinton, Geoffrey E. "A practical guide to training restricted boltzmann machines." Neural networks: Tricks of the trade. Springer Berlin Heidelberg, 2012. 599-619.

van der Maaten, Laurens. "Learning a parametric embedding by preserving local structure." RBM 500.500 (2009): 26.