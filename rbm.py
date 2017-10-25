"""
PyTorch implementation of all (Bernoulli, Gaussian hid, Gaussian vis+hid) kindS of RBMs.

Reference

    Hinton, Geoffrey E. "A practical guide to training restricted boltzmann machines." Neural networks: Tricks of the trade. Springer Berlin Heidelberg, 2012. 599-619.
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

        self.w = torch.randn(vis_num, hid_num)  # weight matrix
        self.a = torch.ones(vis_num)            # bias for visiable units
        self.b = torch.ones(hid_num)            # bias for hidden units

        # Corresponding momentums
        self.w_m = torch.randn(vis_num, hid_num)
        self.a_m = torch.ones(vis_num)
        self.b_m = torch.ones(hid_num)

        if torch.cuda.is_available():

            gpu_id = torch.cuda.current_device()

            params = (self.w, self.a, self.b, self.w_m, self.a_m, self.b_m)
            
            for p in params:

                p = p.cuda(gpu_id)
            
    def p_h_given_v(self, v):
        """
        Compute probability of hidden units given visible units.
        """

        raise NotImplementedError()

    def p_v_given_h(self, h):
        """
        Compute probability of visible units given hidden units.
        """

        raise NotImplementedError()

    def cd(self, v_data, k, lr, beta, lam):
        """
        Perform contrastive divergence with k stpes, i.e. CD_k.

        @input

            v_data: visible data
                 k: MCMC step number
                lr: learning rate
              beta: momentum coefficient
               lam: weight decay rate

        @return

            error: reconstruction error
        """

        v_num = v_data.size()[0]

        # Positive phase
        h_prob_pos = self.p_h_given_v(v_data)
        
        stats_pos = torch.matmul(v_data.t(), h_prob_pos)

        # Negative phase
        h_prob_neg = h_prob_pos.clone()

        for _ in range(k):

            v_prob_neg = self.p_v_given_h(h_prob_neg)
            h_prob_neg = self.p_h_given_v(v_prob_neg)

        stats_neg = torch.matmul(v_prob_neg.t(), h_prob_neg)

        # Update parameters
        self.w_m = beta * self.w_m + stats_pos - stats_neg
        self.a_m = beta * self.a_m + torch.sum(v_data - v_prob_neg, 0)
        self.b_m = beta * self.b_m + torch.sum(h_prob_pos - h_prob_neg, 0)

        self.w = self.w_m * lr / v_num
        self.a = self.a_m * lr / v_num
        self.b = self.b_m * lr / v_num

        self.w = (1 - lam) * self.w     # weight decay

        # Compute reconstruction error
        error = F.mse_loss(v_data, v_prob_neg)

        return error


class RBMBer(RBMBase):

    def __init__(self, vis_num, hid_num):

        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):

        return torch.sigmoid(torch.matmul(v, self.w) + self.b)

    def p_v_given_h(self, h):

        return torch.sigmoid(torch.matmul(h, self.w.t()) + self.a)
