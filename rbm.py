"""
PyTorch implementation of all (Bernoulli, Gaussian hid, Gaussian vis+hid) kindS of RBMs.

Reference

    [1] Hinton, Geoffrey E. "A practical guide to training restricted boltzmann machines." Neural networks: Tricks of the trade. Springer Berlin Heidelberg, 2012. 599-619.
"""


import torch
import torch.nn.functional as F


class RBMBase:

    def __init__(self, vis_num, hid_num):
        """
        Initialization for base RBM.
        """
        
        self.vis_num = vis_num
        self.hid_num = hid_num

        # Dictionary for storing parameters
        self.params = dict()

        self.w = torch.randn(vis_num, hid_num) * 0.01   # weight matrix
        # TODO: provide an init of a based on data; 24.8 of [1]
        self.a = torch.ones(vis_num) / vis_num          # bias for visiable units
        self.b = torch.zeros(hid_num)                   # bias for hidden units

        # Corresponding momentums; _v means velocity of
        self.w_v = torch.randn(vis_num, hid_num)
        self.a_v = torch.ones(vis_num)
        self.b_v = torch.ones(hid_num)

        if torch.cuda.is_available():

            gpu_id = torch.cuda.current_device()

            self.w = self.w.cuda(gpu_id)
            self.a = self.a.cuda(gpu_id)
            self.b = self.b.cuda(gpu_id)
            self.w_v = self.w_v.cuda(gpu_id)
            self.a_v = self.a_v.cuda(gpu_id)
            self.b_v = self.b_v.cuda(gpu_id)

            self.has_gpu = True
            self.gpu_id = gpu_id

        else:

            self.has_gpu = False
            
    def p_h_given_v(self, v):
        """
        Compute probability of hidden units given visible units.
        """

        raise NotImplementedError()

    def sample_h_given_v(self, v):
        """
        Sample hidden units given visible units.
        """

        raise NotImplementedError()

    def p_v_given_h(self, h):
        """
        Compute probability of visible units given hidden units.
        """

        raise NotImplementedError()

    def cd(self, v_data, k, eta, alpha=5e-1, lam=1e-4):
        """
        Perform contrastive divergence with k stpes, i.e. CD_k.

        @input

            v_data: visible data
                 k: MCMC step number
               eta: learning rate
             alpha: momentum coefficient
               lam: weight decay rate

        @return

            error: reconstruction error
        """
        if self.has_gpu:
            v_data = v_data.cuda(self.gpu_id)

        # Positive phase
        h_pos, h_prob_pos = self.sample_h_given_v(v_data)
        
        # Negative phase
        h_neg = h_pos.clone()

        for _ in range(k):

            v_prob_neg = self.p_v_given_h(h_neg)
            h_neg, h_prob_neg = self.sample_h_given_v(v_prob_neg)

        # Compute statistics
        stats_pos = torch.matmul(v_data.t(), h_prob_pos)
        stats_neg = torch.matmul(v_prob_neg.t(), h_prob_neg)

        # Compute gradients
        batch_size = v_data.size()[0]
        w_grad = (stats_pos - stats_neg) / batch_size
        a_grad = torch.sum(v_data - v_prob_neg, 0) / batch_size
        b_grad = torch.sum(h_prob_pos - h_prob_neg, 0) / batch_size

        # Update momentums
        self.w_v = alpha * self.w_v + eta * (w_grad - lam * self.w)
        self.a_v = alpha * self.a_v + eta * a_grad
        self.b_v = alpha * self.b_v + eta * b_grad

        # Update parameters
        self.w = self.w + self.w_v
        self.a = self.a + self.a_v
        self.b = self.b + self.b_v

        # Compute reconstruction error
        error = F.mse_loss(v_data, v_prob_neg)

        return error


class RBMBer(RBMBase):

    def __init__(self, vis_num, hid_num):

        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):

        return torch.sigmoid(torch.matmul(v, self.w) + self.b)

    def sample_h_given_v(self, v):

        r = torch.rand(self.hid_num)
        if self.has_gpu:
            r = r.cuda(self.gpu_id)

        h_prob = self.p_h_given_v(v)

        return (h_prob > r).float(), h_prob

    def p_v_given_h(self, h):

        return torch.sigmoid(torch.matmul(h, self.w.t()) + self.a)
