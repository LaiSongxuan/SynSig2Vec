#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class _bilinearHistogram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim, n_bins, span=1.):
        # compute the step size in the histogram
        step = span / n_bins 
        index = sim / step
        lower = index.floor()
        upper = index.ceil()
        #If index == lower == upper, then this point does not contribute to the histogram loss. 
        delta_u = index - lower 
        delta_l = upper - index
        lower = lower.long()
        upper = upper.long()
        hist = torch.bincount(upper, delta_u, n_bins + 1) + torch.bincount(lower, delta_l, n_bins + 1)
        ctx.save_for_backward(upper, lower)
        return hist

    @staticmethod
    def backward(ctx, grad_hist):
        upper, lower = ctx.saved_tensors
        grad_sim = grad_hist[upper] - grad_hist[lower]
        return grad_sim, None, None

class differientialHistogramAUC(nn.Module): 
    """Learning Deep Embeddings with Histogram Loss
    """
    def __init__(self, n_bins=10):
        super(differientialHistogramAUC, self).__init__()
        self.n_bins = n_bins

    def forward(self, sim_pos, sim_neg):  
        sim_pos = sim_pos.flatten()
        sim_neg = sim_neg.flatten()

        ''' Is it right to do normalization?'''
        max_pos, min_pos = torch.max(sim_pos.data), torch.min(sim_pos.data)
        max_neg, min_neg = torch.max(sim_neg.data), torch.min(sim_neg.data)
        max_ = max_pos if max_pos >= max_neg else max_neg
        min_ = min_pos if min_pos <= min_neg else min_neg
        sim_pos = (sim_pos - min_ + 0.05) / (max_ - min_ + 0.1)
        sim_neg = (sim_neg - min_ + 0.05) / (max_ - min_ + 0.1)

        pdf_pos = _bilinearHistogram.apply(sim_pos, self.n_bins) / sim_pos.shape[0]
        pdf_neg = _bilinearHistogram.apply(sim_neg, self.n_bins) / sim_neg.shape[0]

        '''The two formulation are equivalent and both optimizing the AUC of the ROC curve.'''
        ### p(neg)^i * ∑_{j=0}^{i} p(pos)^j
        cdf_pos = torch.cumsum(pdf_pos, dim=0) 
        loss = (cdf_pos * pdf_neg).sum()
        ### p(pos)^i * ∑_{j=i}^{1} p(neg)^j, 
        # cdf_neg = torch.cumsum(pdf_neg, dim=0)
        # cdf_neg_reverse = cdf_neg[-1] - cdf_neg + pdf_neg
        # loss = (pdf_pos * cdf_neg_reverse).sum()

        return loss, loss

class _fastAP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_pos, h_neg):
        H_pos = torch.cumsum(h_pos, dim=0)
        H = torch.cumsum(h_pos + h_neg, dim=0) + 1e-3
        AP = (H_pos * h_pos / H).sum() / H_pos[-1] #H_pos[-1]: number of positive samples
        ctx.save_for_backward(H, H_pos, h_pos, h_neg)
        return AP

    @staticmethod
    def backward(ctx, grad):
        H, H_pos, h_pos, h_neg = ctx.saved_tensors
        H_neg = H - H_pos
        H2 = torch.pow(H, 2)
        L = h_pos.size(0)
        LTM = torch.tril(torch.ones(L, L), 0).cuda()  # lower traingular matrix

        # 1. d(FastAP)/d(h+)
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0

        d_AP_h_pos = H_pos / H 
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1.unsqueeze(0), LTM).squeeze()
        d_AP_h_pos = d_AP_h_pos / H_pos[-1]
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0

        # 2. d(FastAP)/d(h-)
        tmp2 = - h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0

        d_AP_h_neg = torch.mm(tmp2.unsqueeze(0), LTM).squeeze()
        d_AP_h_neg = d_AP_h_neg / H_pos[-1]
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0

        return grad * d_AP_h_pos, grad * d_AP_h_neg

class differientialHistogramAP(nn.Module):
    """Ref: Deep Metric Learning to Rank 
    """
    def __init__(self, n_bins=10):
        super(differientialHistogramAP, self).__init__()
        self.n_bins = n_bins

    def forward(self, sim_pos, sim_neg):
        # Should be distance rather than similarity. Assume the similarity is between -1 and 1.
        sim_pos = 1. - sim_pos.flatten() 
        sim_neg = 1. - sim_neg.flatten()

        ''' Is it right to do normalization?'''
        max_pos, min_pos = torch.max(sim_pos.data), torch.min(sim_pos.data)
        max_neg, min_neg = torch.max(sim_neg.data), torch.min(sim_neg.data)
        max_ = max_pos if max_pos >= max_neg else max_neg
        min_ = min_pos if min_pos <= min_neg else min_neg
        sim_pos = (sim_pos - min_ + 0.05) / (max_ - min_ + 0.1)
        sim_neg = (sim_neg - min_ + 0.05) / (max_ - min_ + 0.1)

        hist_pos = _bilinearHistogram.apply(sim_pos, self.n_bins, 1.)
        hist_neg = _bilinearHistogram.apply(sim_neg, self.n_bins, 1.)

        # Does autograd work well for this case?
        cumsum_hist_pos = torch.cumsum(hist_pos, dim=0)
        cumsum_hist = torch.cumsum(hist_pos + hist_neg, dim=0) + 1e-3
        AP = (cumsum_hist_pos * hist_pos / cumsum_hist).sum() / sim_pos.shape[0]
        #### AP = _fastAP.apply(hist_pos, hist_neg) #Replace the above three lines
        
        loss = 1. - AP

        return loss, AP

def cummax(x):
    N = x.shape[0] - 1
    x = x.view(1, 1, x.shape[0])
    for i in range(N):
        x = F.max_pool1d(x, kernel_size=2, stride=1, padding=1)[:,:,1:] #Drop the first element
    return x[0, 0]

def piecewiseStepFunc(x, invDelta):
    return (x * invDelta + 0.5).clamp(min=0, max=1)

class perceptronAP(torch.autograd.Function):
    """Ref: Towards Accurate One-Stage Object Detection with AP-Loss
    """
    @staticmethod
    def forward(ctx, sim_pos, sim_neg, invDelta): #invDelta = 0.5 / delta
        H_p = piecewiseStepFunc(sim_pos.unsqueeze(0) - sim_pos.unsqueeze(1), invDelta)
        H_count_p = torch.sum(H_p, dim=1) - 0.5 
        H = piecewiseStepFunc(sim_neg.unsqueeze(0) - sim_pos.unsqueeze(1), invDelta)
        H_count = torch.sum(H, dim=1)
        L = H / (1 + H_count_p + H_count).unsqueeze(1) 

        prec = 1. - torch.sum(L, dim=1) 
        ### Interpolated AP. sim_pos should be sorted in descending order.
        max_prec = cummax(prec) # Accumulative maximum
        L = L * ((1 - max_prec) / (1 - prec + 1e-4)).unsqueeze(1)
        ctx.save_for_backward(L)        

        loss = torch.sum(L)
        return loss, torch.mean(prec)

    @staticmethod
    def backward(ctx, grad_L, temp):
        grad_x = ctx.saved_tensors[0]
        grad_pos = - torch.sum(grad_x, dim=1) / grad_x.shape[0] 
        grad_neg = torch.sum(grad_x, dim=0) / grad_x.shape[0]
        
        return grad_pos, grad_neg, None

class LossAugmentedInferenceAP(object):
  """
  Loss augmented inference algorithm of Song et al.
  for the task loss of Average Precision (AP).

  """

  def __init__(self, num_pos, num_neg, epsilon, positive_update=True):
    """
    :param phi_pos: cosine similarities between the query and each positive point
    :param phi_neg: cosine similarities between the query and each negative point
    :param epsilon: float used by DLM (see the paper for details)
    :param positive_update: whether or not to perform positive update of DLM
    """

    self.num_pos = num_pos
    self.num_neg = num_neg
    self.num_pairs = float(self.num_pos * self.num_neg)
    self.num_acc = torch.arange(1, self.num_pos + 1, dtype=torch.float32)
    
    if positive_update:
      self.negative_update = -1 #decrease the AP (APloss = 1 - AP)
    else:
      self.negative_update = 1 #increase the AP
    
    self.epsilon = epsilon

  def __call__(self, phi_pos, phi_neg):
    B, G = self.compute_B_and_G(phi_pos, phi_neg)
    H, d = self.compute_H_and_d(B, G)
    return d

  def AP(self, d):
    ranking = self.recover_ranking(d)
    return torch.mean(self.num_acc / (ranking[:self.num_pos] + 1))

  def compute_B_and_G(self, phi_pos, phi_neg):
    B = torch.zeros(self.num_pos + 1, self.num_neg + 1)
    G = torch.zeros(self.num_pos + 1, self.num_neg + 1)
    phi_diff = (phi_pos.unsqueeze(1) - phi_neg.unsqueeze(0)) / self.num_pairs
    B[1:, 1:] = - torch.cumsum(phi_diff, dim=1)
    G[1:, 1:] = torch.cumsum(phi_diff, dim=0)

    # for i in range(1, self.num_pos + 1):
    #   for j in range(1, self.num_neg + 1):
    #     B[i, j] = B[i, j - 1] - (phi_pos[i - 1] - phi_neg[j - 1]) / float(
    #         self.num_pos * self.num_neg)
    #     G[i, j] = G[i - 1, j] + (phi_pos[i - 1] - phi_neg[j - 1]) / float(
    #         self.num_pos * self.num_neg)

    return B, G

  def compute_H_and_d(self, B, G):
    H = torch.zeros(self.num_pos + 1, self.num_neg + 1)
    direction = torch.zeros(self.num_pos + 1, self.num_neg + 1)
    for i in range(self.num_pos + 1):
      for j in range(self.num_neg + 1):
        if i == 0 and j == 0:
          H[i, j] = 0
          direction[i, j] = 0
          continue
        if i == 1 and j == 0:
          H[i, j] = self.epsilon * self.negative_update / float(self.num_pos)
          direction[i, j] = 1 # positive 
          continue
        if i == 0 and j == 1:
          H[i, j] = 0
          direction[i, j] = -1 # negative
          continue
        if i == 0:  # but j > 1
          H[i, j] = H[i, j - 1] + G[i, j]
          direction[i, j] = -1
          continue
        ### Optimizing AUC of TPF-FPR curve
        # _add_pos = self.epsilon * 1.0 / self.num_pos / self.num_neg * float(
        #     self.num_neg - j) * self.negative_update + B[i, j]
        ### Optimizing average precision (or, AUC of Precision-Recall curve)
        _add_pos = self.epsilon * 1.0 / self.num_pos * i / float(
            i + j) * self.negative_update + B[i, j]
        if j == 0:
          H[i, j] = H[i - 1, j] + _add_pos
          direction[i, j] = 1
          continue
        if (H[i, j - 1] + G[i, j]) > (H[i - 1, j] + _add_pos):
          H[i, j] = H[i, j - 1] + G[i, j]
          direction[i, j] = -1
        else:
          H[i, j] = H[i - 1, j] + _add_pos
          direction[i, j] = 1
    
    return H, direction

  def recover_ranking(self, d):
    ranking = torch.zeros(self.num_pos + self.num_neg)
    i = self.num_pos
    j = self.num_neg
    while (i >= 0 and j >= 0 and not (i == 0 and j == 0)):
      if d[i, j] == 1:
        ranking[i - 1] = i + j - 1
        i -= 1
      else:
        ranking[j + self.num_pos - 1] = i + j - 1
        j -= 1
    return ranking

