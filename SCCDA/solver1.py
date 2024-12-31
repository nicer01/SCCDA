from __future__ import print_function
import sys, os
import numpy as np
import math
import matplotlib.pyplot as plt
sys.path.append('./datasets')
from load_data import UnalignedDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from model.build_net import Generator, Classifier
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from loss_sct import simpleshot, get_prototype_label_source_2, cal_mean_loss, cal_angle_loss, cal_git_loss, \
    classification_loss_function, compute_cluster_center, cal_cos_loss, make_dirs
from  loss_mmd import mmd_rbf
from sklearn.metrics import auc,roc_curve
from sklearn import preprocessing
from sklearn.manifold import TSNE


class AdaptiveFilteringEMLossForTarget(nn.Module):

    def __init__(self, eps):
        super(AdaptiveFilteringEMLossForTarget, self).__init__()
        self.eps = eps

    def forward(self, prob):
        temp = torch.zeros(prob.size()).cuda(prob.device)
        temp[prob.data == 0] = self.eps
        temp = Variable(temp)

        neg_ent = ((prob * ((prob + temp).log())).sum(1)).exp()

        loss = - (((prob * ((prob + temp).log())).sum(1)) * neg_ent).mean()

        return loss

class Solver(object):
    def __init__(self, args, model_name='none', batch_size_tra=32, batch_size_tes=1, source='svhn', target='mnist',
                 lpp_dim=512,
                 num_workers=4, pin_memory=True, learning_rate=0.0001,
                 interval=100, num_k=4,
                 all_use=False, checkpoint_dir=None, save_epoch=5,
                 record_train='train', record_test='test', tlabel=100,
                 test_shuffle=True, seed=1, rate=0, ts_rate=0, lpp_pt=0, mmd_pt=0,condition_pt=0 ,gama=0.9, expl=None):
        self.batch_size = batch_size_tra
        self.batch_size_tes = batch_size_tes
        self.source = source
        self.target = target
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_k = num_k  # 生成器更新次数
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.record_train = record_train
        self.record_test = record_test
        self.lpp_dim = lpp_dim
        self.tlabel = tlabel
        self.test_shuffle = test_shuffle
        self.seed = seed
        self.rate = rate
        self.ts_rate = ts_rate
        self.model = model_name
        self.mmd_pt = mmd_pt
        self.lpp_pt = lpp_pt
        self.condition_pt = condition_pt
        self.gama = gama
        self.model_name = model_name
        self.expl = expl
        self.maxacc = 0
        self.maxf1 = 0
        self.maxsen = 0
        self.maxspe = 0


        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading...')

        S_train = '/media/li/A21E09551E09243F1/YGX/my_sct/dataset/'+self.source+'/train/train'  # 源域（有标签）

        train_loader = UnalignedDataLoader()
        self.dataset_S_train, self.S_train_num,_ = train_loader.initialize(S_train, self.batch_size, True,
                                                                         self.num_workers,
                                                                         self.pin_memory)

        T_train = '/media/li/A21E09551E09243F1/YGX/my_sct/dataset/'+self.target+'/data16/data'  # 目标域（无标签）
        train_loader = UnalignedDataLoader()
        self.dataset_T_train, self.T_train_num,_ = train_loader.initialize(T_train, self.batch_size, True,
                                                                         self.num_workers,
                                                                         self.pin_memory)


        print('load finished!\n')

        self.G = Generator(model=self.model)

        print("self.G: ", self.G)
        with open(record_train, 'a') as record:
            record.write('self.G: %s\n' % (self.G))

        self.C = Classifier(model=self.model)
        print("self.C: ", self.C)
        with open(record_train, 'a') as record:
            record.write('self.C: %s\n' % (self.C))




        with open(record_train, 'a') as record:
            record.write('--source_train_number: %s\n--target_train_number: %s\n' % (
                self.S_train_num, self.T_train_num))

        print("self.model: ", self.model)
        with open(record_train, 'a') as record:
            record.write('self.model: %s\n' % (self.model))

        self.G.cuda()
        self.C.cuda()


        self.interval = interval
        self.lr = learning_rate

    def train(self, epoch,  record_file=None):

        label_s_t = []
        s_t_f = []

        s_cluster_center = Variable(torch.cuda.FloatTensor(2, 1152).fill_(0))
        t_cluster_center = Variable(torch.cuda.FloatTensor(2, 1152).fill_(0))

        s_center = Variable(torch.cuda.FloatTensor(1, 1152).fill_(0))
        t_center = Variable(torch.cuda.FloatTensor(1, 1152).fill_(0))

        criterion = nn.CrossEntropyLoss().cuda()

        self.G.train()
        self.C.train()

        correct1 = 0
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        t_num = 0

        if epoch == 0:
            self.lr_last = self.lr
        opt_G = optim.Adam(self.G.parameters(), lr=self.lr_last)
        opt_C = optim.Adam(self.C.parameters(), lr=self.lr_last)

        for batch_idx, data in enumerate(self.dataset_S_train):
            if (batch_idx + 1) * self.batch_size > self.S_train_num * self.ts_rate:
                break

            for batch_idx_T, data_T in enumerate(self.dataset_T_train, t_num):
                #print(t_num)

                t_num += 1
                img_t = data_T['data']  # torch.Size([64, 3, 224, 224])
                img_s = data['data']
                label_s = data['label']
                #if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                if img_t.size()[0] < self.batch_size:
                    break
                img_s = Variable(img_s.cuda())
                img_t = Variable(img_t.cuda())
                label_s = Variable(label_s.long().cuda())
                opt_G.zero_grad()
                opt_C.zero_grad()
                feat_s,_,_ = self.G(img_s)
                feat_t,_,_ = self.G(img_t)
                output_s1, s_1152, s_512, s_1152bn = self.C(feat_s)
                output_t1, t_1152, t_512, t_1152bn = self.C(feat_t)

                pred1 = output_s1.data.max(1)[1]
                correct1 += pred1.eq(label_s.data).cpu().sum()
                loss_s = criterion(output_s1, label_s)

                s_center_temp = torch.mean(s_1152, dim=0)
                s_center_temp.reshape(1,1152)

                s_center = cen_updte(t_num, s_center, s_center_temp)


                s_1152 = simpleshot(s_1152, s_center)
                t_1152 = simpleshot(t_1152, s_center)
                #MMD_loss
                mmd_loss = mmd_rbf(s_1152, t_1152)

                s_center_temp = torch.mean(s_1152, dim=0).reshape(1,1152)
                t_center_temp = torch.mean(t_1152, dim=0).reshape(1,1152)

                s_center = cen_updte(t_num, s_center, s_center_temp)
                t_center = cen_updte(t_num, t_center, t_center_temp)
                #DPA Loss
                dpa_loss = F.pairwise_distance(s_center, t_center)


                #target_pseudo_label
                prob_t = F.softmax(output_t1 - output_t1.max(1, True)[0], dim=1)  # 预测软标签
                _, target_pseudo_label = torch.max(prob_t.data, 1)

                #cluster_center
                s_cluster_center_temp = compute_cluster_center(features=s_1152, labels=label_s, class_num=2)
                t_cluster_center_temp = compute_cluster_center(features=t_1152, labels=target_pseudo_label, class_num=2)

                s_cluster_center = cen_updte(t_num, s_cluster_center, s_cluster_center_temp)
                t_cluster_center = cen_updte(t_num, t_cluster_center, t_cluster_center_temp)

                #CPA_loss and SCT_loss
                cpa_loss, sct_loss = cal_cos_loss(s_center, t_center, s_cluster_center, t_cluster_center)

                loss = loss_s + 1*mmd_loss + 1 *dpa_loss + 1*cpa_loss  - 1*sct_loss
                # loss = loss_s + 1 * mmd_loss + 1 * dpa_loss  - 1 * sct_loss
                #loss = loss_s + mmd_loss  + cpa_loss + sct_loss

                batch_cnt = batch_idx + 1
                img_select = self.batch_size * batch_cnt

                loss.backward()
                opt_G.step()
                opt_C.step()

                label_t = data_T['label']
                s_f = s_1152
                t_f = t_1152


                s_t_f.append(s_f.cpu().detach().numpy().reshape(21, 1152))
                s_t_f.append(t_f.cpu().detach().numpy().reshape(21, 1152))


                for i in range(21):
                    if label_s[i] == 0:
                        label_s_t.append(0)
                    if label_s[i] == 1:
                        label_s_t.append(1)
                for i in range(21):
                    if label_t[i] == 0:
                        label_s_t.append(2)
                    if label_t[i] == 1:
                        label_s_t.append(3)

                if batch_cnt % self.interval == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.2f}%)]\t Accuracy: {}/{} ({:.6f}%)\t '.format(
                            epoch, img_select, self.S_train_num * self.ts_rate, img_select / (self.S_train_num * self.ts_rate) * 100,
                            correct1, img_select, float(correct1) / img_select * 100,
                            ))
                    print(
                          'source_cross_Loss:{}\t  DPA_Loss:{}\t  CPA_Loss:{}\nSCT_Loss:{}\t MMD_Loss:{}\t Loss_all:{}\n'
                          .format(loss_s, dpa_loss, cpa_loss, sct_loss, mmd_loss, loss))
                    # print(
                    #     'source_cross_Loss:{}\t  DPA_Loss:{}\t  CPA_Loss:{}\n\t MMD_Loss:{}\t Loss_all:{}\n'
                    #     .format(loss_s, dpa_loss, cpa_loss, mmd_loss, loss))
                    # print(
                    #     'source_cross_Loss:{}\t  CPA_Loss:{}\nSCT_Loss:{}\t MMD_Loss:{}\t Loss_all:{}\n'
                    #     .format(loss_s,cpa_loss, sct_loss, mmd_loss, loss))

                    if record_file:
                        record = open(record_file, 'a')
                        record.write('Train Epoch: %s\t Accuracy: %.6f\t \n' % (epoch, float(correct1) / img_select))
                        record.write(
                                    'source_cross_Loss:%.6f\t  DPA_Loss:%.6f\t  CPA_Loss:%.6f\nSCT_Loss:%.6f\t MMD_Loss:%.6f\t Loss_all:%.6f\n'
                                     %(loss_s, dpa_loss, cpa_loss, sct_loss, mmd_loss, loss))
                        # record.write(
                        #     'source_cross_Loss:%.6f\t  DPA_Loss:%.6f\t  CPA_Loss:%.6f\n\t MMD_Loss:%.6f\t Loss_all:%.6f\n'
                        #     % (loss_s, dpa_loss, cpa_loss, mmd_loss, loss))
                        # record.write(
                        #     'source_cross_Loss:%.6f\t  CPA_Loss:%.6f\nSCT_Loss:%.6f\t MMD_Loss:%.6f\t Loss_all:%.6f\n'
                        #     % (loss_s, cpa_loss, sct_loss, mmd_loss, loss))
                        record.close()
                break

        print(epoch, 'lr={:.10f}'.format(self.lr_last))
        #scheduler.step()
        #self.lr_last = scheduler.get_lr()[0]
        self.lr_last = self.lr_last * 0.9
        print(self.lr_last)
        print('train_batch', batch_idx + 1)

        PKL_DIR = self.source + '_' + self.target + '_' + self.model_name + '_' + self.expl  # 创建文件夹保存模型pkl文件
        PKL_DIR = 'PKL/' + PKL_DIR
        if not os.path.exists(PKL_DIR):
            os.mkdir(PKL_DIR)
        if epoch == int(self.save_epoch):
            torch.save(self.G.state_dict(), PKL_DIR + '/'+self.source+'-'+self.target+'---G_m.pkl')
            torch.save(self.C.state_dict(), PKL_DIR + '/'+self.source+'-'+self.target+'---C_m.pkl')

        s_t_f = np.array(s_t_f).reshape(-1, 1152)
        s_t_f.tolist()
        tsne = TSNE(n_components=2, init='pca', random_state=10)
        s_t_f = tsne.fit_transform(s_t_f)
        scaler = preprocessing.MinMaxScaler()
        s_t_f = scaler.fit_transform(s_t_f)
        # s_t_lable = label_s_t.index(1) * 21
        plt.switch_backend('agg')
        ST_DIR = self.source + '_' + self.target + '_' + self.model_name + '_' + self.expl

        fig1 = plt.figure(1)
        for i in range(len(label_s_t)):
            if label_s_t[i] == 0:
                plt.scatter(s_t_f[i, 0], s_t_f[i, 1], s=10, c='r', marker='.')#s0
            if label_s_t[i] == 1:
                plt.scatter(s_t_f[i, 0], s_t_f[i, 1], s=10, c='g', marker='.')#s1
            if label_s_t[i] == 2:
                plt.scatter(s_t_f[i, 0], s_t_f[i, 1], s=10, c='b', marker='.')#t0
            if label_s_t[i] == 3:
                plt.scatter(s_t_f[i, 0], s_t_f[i, 1], s=10, c='k', marker='.')#t1
        plt.legend(loc=1)
        plt.draw()
        TSNE_LPPst_DIR = 'record/s_t_f/' + ST_DIR
        if not os.path.exists(TSNE_LPPst_DIR):
            os.mkdir(TSNE_LPPst_DIR)
        plt.savefig(TSNE_LPPst_DIR + '/feature_%s.jpg' % (epoch))
        plt.pause(1)  # 间隔的秒数： 4s
        plt.close(fig1)

        return loss



    def test(self, epoch, record_file=None):

        self.G.eval()
        self.C.eval()

        patch_TP = 0
        patch_FP = 0
        patch_TN = 0
        patch_FN = 0

        label_dict = []

        probs = []

        patch_total = 0
        patch_correct = 0
        T_val = '/media/li/A21E09551E09243F1/YGX/my_sct/dataset/'+self.target+'/data10'  # 目标域

        for file_test in os.listdir(T_val):

            T_test = T_val + '/' + file_test
            test_loader = UnalignedDataLoader()
            self.dataset_V_train, self.V_train_num,_ = test_loader.initialize(T_test, self.batch_size_tes, True,
                                                                            self.num_workers,
                                                                            self.pin_memory)

            for batch_idx_V, data_V in enumerate(self.dataset_V_train):
                img = data_V['data']
                label = data_V['label']


                img = img.cuda()

                feat_tt, tt1, tt3 = self.G(img)
                output1, t_1152, t_512, t_1152bn = self.C(feat_tt)



                pred1 = output1.data  # .max(1)[1].cpu()
                prob = F.softmax(pred1,dim=1)
                prob = prob.cpu().numpy()
                pred2 = pred1.max(1)[1]
                pred2 = pred2.cpu().numpy()

                patch_total += self.batch_size_tes
                for i in range(self.batch_size_tes):
                    probs.append(prob[i][1])
                    label_dict.append(int(label[i]))
                    if pred2[i] == int(label[i]):
                        patch_correct += 1
                    if pred2[i] == int(label[i]) == 1:
                        patch_TP += 1
                    if pred2[i] == 1 and pred2[i] != int(label[i]):
                        patch_FP += 1
                    if pred2[i] == int(label[i]) == 0:
                        patch_TN += 1
                    if pred2[i] == 0 and pred2[i] != int(label[i]):
                        patch_FN += 1

        #         img_file_label = img_file_label.cpu().numpy()
        #         pre_num = 0
        #         for key_name in img_file_label:
        #             if key_name not in label_dict.keys():
        #                 label_dict[key_name] = pred1[pre_num]
        #             else:
        #                 label_dict[key_name] += pred1[pre_num]
        #             pre_num += 1
        #     for key in label_dict.keys():
        #         bb = label_dict[key]
        #         list1.append(bb.tolist())
        #         y_predict = np.argmax(bb)
        #         y_actual = int(float(label))
        #         aa = 1 if y_predict == y_actual else 0
        #         list_acc.append(aa)
        #
        #         if y_actual == y_predict == 1:
        #             TP += 1
        #         if y_predict == 1 and y_actual != y_predict:
        #             FP += 1
        #         if y_actual == y_predict == 0:
        #             TN += 1
        #         if y_predict == 0 and y_actual != y_predict:
        #             FN += 1
        #
        # print('TP', TP, 'FP', FP, 'TN', TN, 'FN', FN)
        #
        # sensitivity = TP / (TP + FN+0.000000000001)
        # specificity = TN / (FP + TN+0.000000000001)
        # precision = TP / (TP + FP+0.00000001)
        # recall = sensitivity
        # F1_Score = 2 * (precision * recall) / ((precision + recall)+0.0000001)
        # acc = float(TN + TP) / (TN + TP + FN + FP)
        fpr , tpr, thre = roc_curve(y_true=label_dict,y_score=probs,pos_label=1)
        roc_auc =auc(fpr,tpr)
        patch_acc = patch_correct/patch_total
        patch_sensitivity = patch_TP / (patch_TP + patch_FN)
        patch_specificity = patch_TN / (patch_FP + patch_TN)
        patch_precision = patch_TP / (patch_TP + patch_FP +0.0000001)
        patch_recall = patch_sensitivity
        patch_F1 = 2 * (patch_precision * patch_recall)/(patch_precision + patch_recall+0.0000001)

        # if acc > self.maxacc:
        #     self.maxacc = acc
        #     self.maxf1 = F1_Score
        #     self.maxsen = sensitivity
        #     self.maxspe = specificity
        #     print('\nTest Epoch: {}\t F1_Score: {}\t Accuracy: {:.6f}%\t sensitivity: {}\t specificity: {}'.format(
        #         epoch, F1_Score, acc,
        #         sensitivity,
        #         specificity))
        # else:
        #     print(
        #         '\nTest Epoch: {}\t F1_Score: {}\t Accuracy: {:.6f}% \t sensitivity: {}\t specificity: {}'.format(
        #             epoch, self.maxf1, self.maxacc,
        #             self.maxsen,
        #             self.maxspe))
        print('\nTest Epoch: {}\t F1_Score: {}\t Accuracy: {:.6f}%\t Auc:{:.6f}%\t sensitivity: {}\t specificity: {}'.format(
            epoch, patch_F1, patch_acc,roc_auc,
            patch_sensitivity,
            patch_specificity))
        print('\t\t\t\t\tpatch_TP:{}\t patch_TN:{}\tpatch_FP:{}\tpatch_FN:{}'.format(
            patch_TP, patch_TN, patch_FP, patch_FN
        ))
        if record_file:
            record = open(record_file, 'a')
            print('recording: ', record_file, '\n')
            record.write('Test Epoch: %s\t\n' % (epoch))
            record.write('patch_F1:%.6f\t patch_acc:%.6f\t patch_auc:%.6f\t patch_sen:%.6f\t patch_spe:%.6f\t\n'%(
                patch_F1,patch_acc,roc_auc,patch_sensitivity,patch_specificity))
            record.write('patch_TP: %s\t patch_TN: %s\t patch_FP: %s\t patch_FN: %s\t ' % (patch_TP, patch_TN, patch_FP,patch_FN))
            record.close()
        return patch_acc, patch_F1, patch_acc



    def test_wsi(self, epoch, record_file=None):

        self.G.load_state_dict(torch.load('PKL/mice_new_PAIP_test_mlm_lout_flaten_param/mice_new-PAIP_test---G_m.pkl'))
        self.C.load_state_dict(torch.load('PKL/mice_new_PAIP_test_mlm_lout_flaten_param/mice_new-PAIP_test---C_m.pkl'))

        self.G.eval()
        self.C.eval()

        patch_TP = 0
        patch_FP = 0
        patch_TN = 0
        patch_FN = 0
        patch_total = 0
        patch_correct = 0
        WSI_name = 'training_data_29_0.png'
        WSI_path = '/media/li/A21E09551E09243F1/YGX/data_pre/png/level0/patch_final/'+ WSI_name
        np_cam1 = np.zeros((1000,1000))
        label_dict = []
        probs = []

        for file_test in os.listdir(WSI_path):

            T_test = WSI_path + '/' + file_test
            test_loader = UnalignedDataLoader()
            self.dataset_V_train, self.V_train_num,self.wsi_dataset = test_loader.initialize(T_test, self.batch_size_tes, False,
                                                                            self.num_workers,
                                                                            self.pin_memory,drop_last=False)


            for batch_idx_V, data_V in enumerate(self.dataset_V_train):

                img = data_V['data']
                label = data_V['label']

                img = img.cuda()

                feat_tt, tt1, tt3 = self.G(img)
                output1, t_1152, t_512, t_1152bn = self.C(feat_tt)

                pred1 = output1.data  # .max(1)[1].cpu()
                prob = F.softmax(pred1,dim=1)
                probb = prob.cpu().numpy()
                pred2 = pred1.max(1)[1]
                pred2 = pred2.cpu().numpy()

                for i in range(len(label)):
                    probs.append(probb[i][1])
                    label_dict.append(int(label[i]))
                    if pred2[i] == int(label[i]):
                        patch_correct += 1

                    if pred2[i] == int(label[i]) == 1:
                        patch_TP += 1

                        img_name = self.wsi_dataset.imgs[patch_total + i][0].split('/')[-1]
                        w_id = int(img_name.split('.')[0].split('_')[-2])
                        h_id = int(img_name.split('.')[0].split('_')[-1])
                        np_cam1[w_id,h_id] = prob[i][1]


                    if pred2[i] == 1 and pred2[i] != int(label[i]):
                        patch_FP += 1
                        img_name = self.wsi_dataset.imgs[patch_total + i][0].split('/')[-1]
                        w_id = int(img_name.split('.')[0].split('_')[-2])
                        h_id = int(img_name.split('.')[0].split('_')[-1])
                        np_cam1[w_id, h_id] = prob[i][1]


                    if pred2[i] == int(label[i]) == 0:
                        patch_TN += 1
                    if pred2[i] == 0 and pred2[i] != int(label[i]):
                        patch_FN += 1
                patch_total += len(label)
        fpr, tpr, thre = roc_curve(y_true=label_dict, y_score=probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        patch_acc = patch_correct/patch_total
        patch_sensitivity = patch_TP / (patch_TP + patch_FN)
        patch_specificity = patch_TN / (patch_FP + patch_TN)
        patch_precision = patch_TP / (patch_TP + patch_FP +0.0000001)
        patch_recall = patch_sensitivity
        patch_F1 = 2 * (patch_precision * patch_recall)/(patch_precision + patch_recall+0.0000001)

        np.savetxt( '/media/li/A21E09551E09243F1/YGX/data_pre/png/level0/patch_final/np_txt/' + WSI_name + '_np_cam1.txt',np_cam1)


        print('\nTest Epoch: {}\t F1_Score: {}\t Accuracy: {:.6f}%\t Auc: {:.6f}%\t sensitivity: {}\t specificity: {}'.format(
            epoch, patch_F1, patch_acc,roc_auc,
            patch_sensitivity,
            patch_specificity))
        print('\t\t\t\t\tpatch_TP:{}\t patch_TN:{}\tpatch_FP:{}\tpatch_FN:{}'.format(
            patch_TP, patch_TN, patch_FP, patch_FN
        ))
        if record_file:
            record = open(record_file, 'a')
            print('recording: ', record_file, '\n')
            record.write('Test Epoch: %s\t\n' % (epoch))
            record.write('patch_F1:%.6f\t patch_acc:%.6f\t patch_auc:%.6f\t patch_sen:%.6f\t patch_spe:%.6f\t\n'%(
                patch_F1,patch_acc,roc_auc,patch_sensitivity,patch_specificity))
            record.write('patch_TP: %s\t patch_TN: %s\t patch_FP: %s\t patch_FN: %s\t ' % (patch_TP, patch_TN, patch_FP,patch_FN))
            record.close()
        return patch_acc, patch_F1, patch_acc


def cen_dis(cs, ct):
    cs = Variable(cs.data.clone())
    ct = Variable(ct.data.clone())
    dis_0 = 0
    dis_1 = 1
    for i in range(2):
        for t in range(int(cs[0].size(0))):
            if i == 0:
                d_temp = cs[i][t] - ct[i][t]
                dis_0 = dis_0 + d_temp**2
            else:
                d_temp = cs[i][t] - ct[i][t]
                dis_1 = dis_1 + d_temp**2
    dis_0 = dis_0**0.5
    dis_1 = dis_1**0.5
    dis_c = dis_0 + dis_1

    return dis_c

def cen_updte(t,cen,centemp):
    if t == 0:
        cen_new = centemp
    else:
        cen = Variable(cen.cuda())
        cen_new = centemp + cen * 0.2

    return cen_new



if __name__ == '__main__':
    code = 0
