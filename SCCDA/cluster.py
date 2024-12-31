import time
import torch
import os
import math

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from model.build_net import Generator, Classifier

# def train_compute_class_mean(train_loader_source, train_loader_source_batch, train_loader_target,
#                              train_loader_target_batch, model, criterion, criterion_afem, optimizer, itern,
#                              current_epoch, args):
def train_compute_class_mean(img_s, label_s , feat1_s, feat2_s, pred_t, feat1_t, feat2_t,
                             criterion, criterion_afem,cs_1,ct_1,cs_2,ct_2,predicter):

    feat1_s = feat1_s.view(feat1_s.size(0), -1)
    feat2_s = feat2_s.view(feat2_s.size(0), -1)
    feat2_t = feat2_t.view(feat2_t.size(0), -1)
    feat1_t = feat1_t.view(feat1_t.size(0), -1)
    target_source_var = label_s
    cs_target_var = Variable(torch.arange(0, 2).cuda(non_blocking=True))
    ct_target_var = Variable(torch.arange(0, 2).cuda(non_blocking=True))

    # model.train()  # turn to training mode


    target_source = label_s.cuda(non_blocking=True)
    input_source_var =  img_s #Variable(input_source)   #图


    # compute source and target centroids on respective batches at the current iteration
    prob_t = F.softmax(pred_t - pred_t.max(1, True)[0], dim=1)  #预测软标签
    idx_max_prob = prob_t.topk(1, 1, True, True)[-1]           #预测硬标签

    cs_1_temp = Variable(torch.cuda.FloatTensor(cs_1.size()).fill_(0))    #临时储存
    cs_count = torch.cuda.FloatTensor(2, 1).fill_(0)
    ct_1_temp = Variable(torch.cuda.FloatTensor(ct_1.size()).fill_(0))
    ct_count = torch.cuda.FloatTensor(2, 1).fill_(0)
    cs_2_temp = Variable(torch.cuda.FloatTensor(cs_2.size()).fill_(0))
    ct_2_temp = Variable(torch.cuda.FloatTensor(ct_2.size()).fill_(0))

    for i in range(input_source_var.size(0)):    

        cs_1_temp[target_source[i]] += feat1_s[i]
        cs_count[target_source[i]] += 1
        cs_2_temp[target_source[i]] += feat2_s[i]

        ct_1_temp[idx_max_prob[i]] += feat1_t[i]
        ct_count[idx_max_prob[i]] += 1
        ct_2_temp[idx_max_prob[i]] += feat2_t[i]

    # exponential moving average centroids
    cs_1 = Variable(cs_1.data.clone())  #深拷贝 我拷我自己 ？
    ct_1 = Variable(ct_1.data.clone())
    cs_2 = Variable(cs_2.data.clone())
    ct_2 = Variable(ct_2.data.clone())

    mask_s = ((cs_1.data != 0).sum(1, keepdim=True) != 0).float() * 0.7  #args.remain
    mask_t = ((ct_1.data != 0).sum(1, keepdim=True) != 0).float() * 0.7  #args.remain
    mask_s[cs_count == 0] = 1.0
    mask_t[ct_count == 0] = 1.0
    cs_count[cs_count == 0] = 1e-6  #'a small value to prevent underflow'
    ct_count[ct_count == 0] = 1e-6


    cs_1 = mask_s * cs_1 + (1 - mask_s) * (cs_1_temp / cs_count)
    ct_1 = mask_t * ct_1 + (1 - mask_t) * (ct_1_temp / ct_count)
    cs_2 = mask_s * cs_2 + (1 - mask_s) * (cs_2_temp / cs_count)
    ct_2 = mask_t * ct_2 + (1 - mask_t) * (ct_2_temp / ct_count)

    #centroid forward
    ###fusion
    # pred_s_1 = predicter.fc2(F.relu(predicter.fc1(F.relu(cs_1))))
    # pred_t_1 = predicter.fc2(F.relu(predicter.fc1(F.relu(ct_1))))
    # pred_s_2 = predicter.fc2(F.relu(cs_2))
    # pred_t_2 = predicter.fc2(F.relu(ct_2))
    ##no_fusion
    pred_s_1 = predicter.fc2(F.relu(cs_1))
    pred_t_1 = predicter.fc2(F.relu(cs_1))
    pred_s_2 = predicter.fc2(F.relu(cs_2))
    pred_t_2 = predicter.fc2(F.relu(cs_2))
    #
    # compute instance-to-centroid distances
    dist_fs_cs_1 = (feat1_s.unsqueeze(1) - cs_1.unsqueeze(0)).pow(2).sum(2)
    sim_fs_cs_1 = F.softmax(-1 * dist_fs_cs_1, dim=1)#source_cen1 feature and cen dis
    dist_fs_cs_2 = (feat2_s.unsqueeze(1) - cs_2.unsqueeze(0)).pow(2).sum(2)
    sim_fs_cs_2 = F.softmax(-1 * dist_fs_cs_2, dim=1)#source_cen2 feature and cen dis

    dist_ft_ct_1 = (feat1_t.unsqueeze(1) - ct_1.unsqueeze(0)).pow(2).sum(2)
    sim_ft_ct_1 = F.softmax(-1 * dist_ft_ct_1, dim=1)#target_cen1 feature and cen dis
    dist_ft_ct_2 = (feat2_t.unsqueeze(1) - ct_2.unsqueeze(0)).pow(2).sum(2)
    sim_ft_ct_2 = F.softmax(-1 * dist_ft_ct_2, dim=1)#target_cen2 feature and cen dis

    # compute centroid-to-centroid distances
    dist_cs_cs_1 = (cs_1.unsqueeze(1) - cs_1.unsqueeze(0)).pow(2).sum(2)#source_cen1 cen and cen dis
    dist_cs_cs_2 = (cs_2.unsqueeze(1) - cs_2.unsqueeze(0)).pow(2).sum(2)#source_cen2 cen and cen dis
    dist_ct_ct_1 = (ct_1.unsqueeze(1) - ct_1.unsqueeze(0)).pow(2).sum(2)#target_cen1 cen and cen dis
    dist_ct_ct_2 = (ct_2.unsqueeze(1) - ct_2.unsqueeze(0)).pow(2).sum(2)#target_cen2 cen and cen dis

    #compute instance-to-instance distances
    dist_fs_ft_1 = (feat1_s.unsqueeze(1) - feat1_t.unsqueeze(0)).pow(2).sum(2)
    sim_fs_ft_1 = F.softmax(-1 * dist_fs_ft_1, dim=1)

    dist_fs_ft_2 = (feat2_s.unsqueeze(1) - feat2_t.unsqueeze(0)).pow(2).sum(2)
    sim_fs_ft_2 = F.softmax(-1 * dist_fs_ft_2, dim=1)

    ##compute s_centroid-to-t_centroid distances
    dist_cs_ct_1 = (cs_1.unsqueeze(1) - ct_1.unsqueeze(0)).pow(2).sum(2)
    dist_cs_ct_2 = (cs_2.unsqueeze(1) - ct_2.unsqueeze(0)).pow(2).sum(2)


    loss_p1 = (criterion(sim_fs_cs_1, target_source_var) +
               criterion(-1 * dist_cs_cs_1, cs_target_var) +  # -1 * dist_fs_cs_1   ???
               criterion_afem(sim_ft_ct_1) +
               criterion(-1 * dist_ct_ct_1 , ct_target_var) +
               criterion(pred_s_1,cs_target_var) +criterion(pred_t_1,ct_target_var)
               )


    loss_p2 = (criterion(sim_fs_cs_2, target_source_var) +
               criterion(-1 * dist_cs_cs_2, cs_target_var) +  # -1 * dist_fs_cs_2
               criterion_afem(sim_ft_ct_2) +
               criterion(-1 * dist_ct_ct_2, ct_target_var)+
               criterion(dist_cs_ct_2,cs_target_var)+
               criterion(pred_s_2,cs_target_var) +criterion(pred_t_2,ct_target_var)
               )
    loss_prob_t = criterion_afem(prob_t)



    return cs_1,ct_1,cs_2,ct_2,loss_p1,loss_p2,loss_prob_t



