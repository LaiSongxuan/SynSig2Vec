#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy
from optimAP import LossAugmentedInferenceAP
from optimAP import perceptronAP, differientialHistogramAP

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, vocab_size):
        assert 0.0 < label_smoothing <= 0.5
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (vocab_size - 1)
        one_hot = torch.full((vocab_size,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, logit, target):
        """
        logit (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        target_prob = self.one_hot.repeat(target.size(0), 1)
        target_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        # Note: y*(logy-x)
        return F.kl_div(F.log_softmax(logit, dim=1), target_prob, reduction='none').sum(1).mean(0)

class selectivePooling(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, tau=1.0):
        super(selectivePooling, self).__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads, head_dim), requires_grad=True)
        self.w_q = nn.Conv1d(in_dim, head_dim * num_heads, kernel_size=1)
        self.norm = 1 / head_dim**0.5
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.count = 0
        nn.init.orthogonal_(self.keys, gain=1)
        # nn.init.kaiming_normal_(self.keys, a=1)
        nn.init.kaiming_normal_(self.w_q.weight, a=1)
        nn.init.zeros_(self.w_q.bias)

    def forward(self, x, mask, save=False):
        N = x.shape[0]; T = x.shape[2]
        queries = values = self.w_q(x).transpose(1, 2).view(N, T, self.num_heads, self.head_dim) #(N, T, num_heads, head_dim)
        atten = F.softmax(torch.sum(queries * self.keys, dim=-1) * self.norm - (1.-mask).unsqueeze(2)*1000, dim=1) #(N, T, num_heads)  
        head = torch.sum(values * atten.unsqueeze(3), dim=1).view(N, -1) #(N, num_heads * head_dim)
        # if save: numpy.save("./expScripts/attenWeight_bsid/atten%d.npy"%self.count, atten.detach().cpu().numpy()); self.count += 1
        return head

    def orthoNorm(self):
        keys = F.normalize(self.keys, dim=1)
        corr = torch.mm(keys, keys.transpose(0, 1))
        return torch.sum(torch.triu(corr, 1).abs_())

class Sig2Vec(nn.Module):
    def __init__(self, n_in,
                n_classes, 
                n_task = 1,
                n_shot_g = 5, 
                n_shot_f = 5,
                APAlpha = 5.):
        super(Sig2Vec, self).__init__() 
        ''' Define the network and the training loss. '''
        # mAP direct loss minimization parameters
        self.n_classes = n_classes
        self.n_task = n_task 
        self.n_shot_g = n_shot_g 
        self.n_shot_f = n_shot_f
        self.positive_update = True
        self.epsilon = 1 
        self.APAlpha = APAlpha if self.positive_update else 1.0
        self.LAI = LossAugmentedInferenceAP(n_shot_g, n_shot_f, self.epsilon, self.positive_update)
        self.smoothCElossMask = torch.zeros(n_task * (1 + n_shot_g + n_shot_f)).cuda()
        # for i in range(n_task):
        #     self.smoothCElossMask[i*(1+n_shot_g+n_shot_f):i*(1+n_shot_g+n_shot_f)+1+n_shot_g]=(1.0+n_shot_g+n_shot_f)/(1.0+n_shot_g)
        # 1-D ConvNet
        self.convNet = nn.Sequential(
                        nn.Conv1d(n_in, 64, kernel_size=7, padding=3, stride=1, dilation=1),
                        nn.MaxPool1d(2, 2, ceil_mode=True),
                        nn.SELU(inplace=True),
                        nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1),
                        nn.SELU(inplace=True),
                    )
        self.convNet2 = nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, dilation=1),
                        nn.MaxPool1d(2, 2, ceil_mode=True),
                        nn.SELU(inplace=True),
                        nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=1, dilation=1),
                        nn.SELU(inplace=True),
                    )
        self.convNet3 = nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, dilation=1),
                        nn.MaxPool1d(2, 2, ceil_mode=True),
                        nn.SELU(inplace=True),
                        nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=1, dilation=1),
                        nn.SELU(inplace=True),
                    )
        self.Upsample = nn.Upsample(scale_factor=2)
        self.convNet2_1x1 = nn.Conv1d(128, 256, kernel_size=1, padding=0, stride=1, dilation=1)
        self.cls = nn.Linear(512+512, n_classes, bias=False)
        self.bn = nn.BatchNorm1d(512+512, affine=False, momentum=0.001)
        self.sp2 = selectivePooling(256, head_dim=32, num_heads=16)
        self.sp3 = selectivePooling(256, head_dim=32, num_heads=16)
        # Normal initialization with zero mean and sqrt(1/n_in) variance for SELU
        nn.init.kaiming_normal_(self.convNet[0].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet[3].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet2[0].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet2[3].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet3[0].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet3[3].weight, a=1) 
        nn.init.kaiming_normal_(self.convNet2_1x1.weight, a=0) 
        nn.init.kaiming_normal_(self.cls.weight, a=1)
        nn.init.zeros_(self.convNet[0].bias) 
        nn.init.zeros_(self.convNet[3].bias) 
        nn.init.zeros_(self.convNet2[0].bias) 
        nn.init.zeros_(self.convNet2[3].bias) 
        nn.init.zeros_(self.convNet3[0].bias) 
        nn.init.zeros_(self.convNet3[3].bias) 
        nn.init.zeros_(self.convNet2_1x1.bias) 
        ### For Triplet loss
        # self.triplet_dense = nn.Linear(1024, 1024, bias=True)
        ### For BCE
        # self.PDSN_diff = nn.Linear(1024, 512, bias=True)
        # self.PDSN_pos = nn.Linear(1024, 512, bias=True)
        # self.PDSN = nn.Sequential(
        #     nn.Linear(1024, 1024, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1, bias=True),
        #     nn.Sigmoid()
        # )
        # nn.init.kaiming_normal_(self.PDSN_diff.weight, a=0)
        # nn.init.kaiming_normal_(self.PDSN_pos.weight, a=0)
        # nn.init.kaiming_normal_(self.PDSN[0].weight, a=0)
        # nn.init.kaiming_normal_(self.PDSN[2].weight, a=0)
        # nn.init.zeros_(self.PDSN_diff.bias) 
        # nn.init.zeros_(self.PDSN_pos.bias) 
        # nn.init.zeros_(self.PDSN[0].bias) 
        # nn.init.zeros_(self.PDSN[2].bias)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def getOutputMask(self, lens):    
        lens = numpy.array(lens, dtype=numpy.int32)
        for i in range(2):
            lens = (lens + 1) // 2
        N = len(lens); D = numpy.max(lens)
        mask = numpy.zeros((N, D), dtype=numpy.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask
    
    def l2(self, x):
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        return x

    # def APLoss_DH(self, x):
    #     step = 1 + self.n_shot_g + self.n_shot_f
    #     loss = 0
    #     mAP = 0
    #     for i in range(self.n_task):
    #         anchor = x[i*step]
    #         pos = x[i*step+1:i*step+1+self.n_shot_g]
    #         neg = x[i*step+1+self.n_shot_g:(i+1)*step]
    #         la = torch.norm(anchor) + 1e-8
    #         lp = torch.norm(pos, dim=1) + 1e-8
    #         ln = torch.norm(neg, dim=1) + 1e-8
    #         scoreAP = torch.sum(anchor.unsqueeze(0) * pos, dim=1) / (la * lp) #Cosine similarity as score
    #         scoreAN = torch.sum(anchor.unsqueeze(0) * neg, dim=1) / (la * ln)
    #         scoreAP, _ = torch.sort(scoreAP, descending=True)
    #         scoreAN, _ = torch.sort(scoreAN, descending=True)
    #         l, AP = self.dhAP(scoreAP, scoreAN) 
    #         loss += l; mAP += AP
    #     loss = loss / self.n_task
    #     mAP = mAP / self.n_task
    #     return loss, mAP

    # def APLoss_perceptron(self, x):
    #     step = 1 + self.n_shot_g + self.n_shot_f
    #     loss = 0
    #     mAP = 0
    #     for i in range(self.n_task):
    #         anchor = x[i*step]
    #         pos = x[i*step+1:i*step+1+self.n_shot_g]
    #         neg = x[i*step+1+self.n_shot_g:(i+1)*step]
    #         la = torch.norm(anchor) + 1e-8
    #         lp = torch.norm(pos, dim=1) + 1e-8
    #         ln = torch.norm(neg, dim=1) + 1e-8
    #         scoreAP = torch.sum(anchor.unsqueeze(0) * pos, dim=1) / (la * lp) #Cosine similarity as score
    #         scoreAN = torch.sum(anchor.unsqueeze(0) * neg, dim=1) / (la * ln)
    #         scoreAP, _ = torch.sort(scoreAP, descending=True)
    #         scoreAN, _ = torch.sort(scoreAN, descending=True)
    #         _loss, AP = perceptronAP.apply(scoreAP, scoreAN, 0.5) #invDelta = 0.5 / delta
    #         loss += _loss; mAP += AP
    #     loss = loss / self.n_task
    #     mAP = mAP / self.n_task
    #     return loss, mAP

    def APLoss_DLM(self, x):
        step = 1 + self.n_shot_g + self.n_shot_f
        score_std = score_aug = score_GT = 0
        mAP = 0
        for i in range(self.n_task):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+1+self.n_shot_g]
            neg = x[i*step+1+self.n_shot_g:(i+1)*step]
            la = torch.norm(anchor) + 1e-8
            lp = torch.norm(pos, dim=1) + 1e-8
            ln = torch.norm(neg, dim=1) + 1e-8
            scoreAP = torch.sum(anchor.unsqueeze(0) * pos, dim=1) / (la * lp) #Cosine similarity as score
            scoreAN = torch.sum(anchor.unsqueeze(0) * neg, dim=1) / (la * ln)
            scoreAP, _ = torch.sort(scoreAP, descending=True)
            scoreAN, _ = torch.sort(scoreAN, descending=True)
            simDiff = scoreAP.unsqueeze(1) - scoreAN.unsqueeze(0) #(n_shot_g, n_shot_f)
            Y_std = torch.sign(simDiff)
            score_std += torch.mean(Y_std * simDiff)
            # score_GT += torch.mean(simDiff)
            direction = self.LAI(scoreAP.cpu(), scoreAN.cpu())
            mAP += self.LAI.AP(direction)
            Y_aug = -1 * direction[1:, 1:]
            score_aug += torch.mean(Y_aug.cuda() * simDiff)
        score_std = score_std / self.n_task
        score_aug = score_aug / self.n_task
        # score_GT = score_GT / self.n_task
        mAP = mAP / self.n_task
        loss = 1 / self.epsilon * (self.APAlpha * score_aug - score_std) #DLM
        # loss = (1 / self.epsilon) * (self.APAlpha * score_aug - score_GT) #SSVM
        if not self.positive_update:
            loss = -loss
        return loss, mAP

    def tripletLoss(self, x, margin=0.25):
        step = 1 + self.n_shot_g + self.n_shot_f
        var = triLoss_std = triLoss_hard = 0
        # x = F.relu(self.triplet_dense(x), inplace=True)
        for i in range(self.n_task):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+1+self.n_shot_g]
            neg = x[i*step+1+self.n_shot_g:(i+1)*step]
            anchor = anchor / torch.norm(anchor)
            pos = pos / torch.norm(pos, dim=1, keepdim=True)
            neg = neg / torch.norm(neg, dim=1, keepdim=True)
            dist_g = torch.sum((anchor.unsqueeze(0) - pos)**2, dim=1)
            dist_f = torch.sum((anchor.unsqueeze(0) - neg)**2, dim=1)
            ### Inner class variation
            var += torch.sum(dist_g) / self.n_shot_g
            ### Triplet loss, self.n_shot_g * self.n_shot_f triplets in total
            triLoss = F.relu(dist_g.unsqueeze(1) - dist_f.unsqueeze(0) + margin) #(self.n_shot_g, self.n_shot_f)
            # triLoss_std += torch.mean(triLoss) 
            triLoss_hard += torch.sum(triLoss) / (triLoss.data.nonzero().size(0) + 1) # batch hard sample mining
            # dist_g, _ = torch.sort(dist_g, descending=True)
            # dist_f, _ = torch.sort(dist_f, descending=True)
            # triLoss_hard += torch.mean(F.relu(dist_g[:2].unsqueeze(1) - dist_f[-4:].unsqueeze(0) + margin))
        var = var / self.n_task
        triLoss_std = triLoss_std / self.n_task
        triLoss_hard = triLoss_hard / self.n_task
        return triLoss_hard, triLoss_std, var

    def siameseLoss(self, x):
        step = 1 + self.n_shot_g + self.n_shot_f
        loss = 0
        for i in range(self.n_task):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+1+self.n_shot_g]
            neg = x[i*step+1+self.n_shot_g:(i+1)*step]
            diff_ap = torch.abs(anchor - pos)
            pos_ap = (anchor + pos) / 2
            diff_an = torch.abs(anchor - neg)
            pos_an = (anchor + neg) / 2
            pos = torch.cat([self.l2(F.relu(self.PDSN_diff(diff_ap), inplace=True)), self.l2(F.relu(self.PDSN_pos(pos_ap), inplace=True))], dim=1)
            neg = torch.cat([self.l2(F.relu(self.PDSN_diff(diff_an), inplace=True)), self.l2(F.relu(self.PDSN_pos(pos_an), inplace=True))], dim=1)
            scoreAP = self.PDSN(pos)
            scoreAN = self.PDSN(neg)
            loss += F.binary_cross_entropy(scoreAP, torch.ones([self.n_shot_g, 1],dtype=torch.float32).cuda()-0.1)
            loss += F.binary_cross_entropy(scoreAN, torch.ones([self.n_shot_f, 1],dtype=torch.float32).cuda()-0.9)
        loss = loss / self.n_task
        return loss

    def smoothCEloss(self, logit, target, eps=0.1):
        n_class = logit.size(1)
        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean(dim=0)
        # return -((one_hot * log_prb).sum(dim=1) * self.smoothCElossMask).mean(dim=0)

    def forward(self, x, mask, length):
        mask = mask.unsqueeze(1)
        x = x.transpose(1,2) #(N,D,T)
        output = self.convNet(x)
        length = (length+1)//2

        output2 = self.convNet2(output)
        length = (length+1)//2  

        output3 = self.convNet3(output2)
        length = (length+1)//2

        output2 = F.selu(self.Upsample(output3)[:,:,:output2.shape[2]] + self.convNet2_1x1(output2), inplace=True)
        feat2 = self.sp2(output2, torch.squeeze(mask))
        # feat2 = torch.sum(output2 * mask, dim=2) / (length.unsqueeze(1)*2-1)
        mask = F.max_pool1d(mask, 2, ceil_mode=True)
        feat3 = self.sp3(output3, torch.squeeze(mask))
        # feat3 = torch.sum(output3 * mask, dim=2) / length.unsqueeze(1)

        output = torch.cat([feat2, feat3], dim=1)
        # Real handwritten signatures: task 1 or 4 & no BN; Synthetic signatures: task 4 & with BN
        output = self.bn(output) 

        return output, self.cls(output)
