from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from Models import MMSSL, Discriminator
from utility.batch_test import *
from utility.logging import Logger

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        # 记录日志信息
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr  # 0.00056
        self.emb_dim = args.embed_size  # 64
        self.batch_size = args.batch_size  # 1024
        self.weight_size = eval(args.weight_size)  # [64, 64]
        self.n_layers = len(self.weight_size)  # 2
        self.regs = eval(args.regs)  # [1e-5,1e-5,1e-2]
        self.decay = self.regs[0]  # 1e-5

        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))  # n_item*2048
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))  # n_item*20
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]
        self.ui_graph = self.ui_graph_raw = pickle.load(
            open(args.data_path + args.dataset + '/train_mat', 'rb'))  # csr稀疏交互矩阵  n_user*n_item
        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(self.ui_graph_raw.todense()).cuda()
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(self.ui_graph_raw.T.todense()).cuda()
        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph
        self.model = MMSSL(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.image_feats,
                           self.text_feats)
        self.model = self.model.cuda()
        self.D = Discriminator(self.n_items).cuda()  # 初始化判别器
        self.D.apply(self.weights_init)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.D_lr, betas=(0.5, 0.9))

        self.optimizer_D = optim.AdamW([{'params': self.model.parameters()}, ], lr=self.lr)
        self.scheduler_D = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)  # 匿名函数，学习率随迭代次数衰减
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))  # 按行 计算与某用户交互的item总和
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)  # 构造对角矩阵   公式2中归一化

        colsum = np.array(csr_mat.sum(0))  # 按列 计算与某项目交互的user总和
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)  # 构造对角矩阵   公式2中归一化

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat  # 在交互矩阵基础上乘  -0.5次方的度

    def matrix_to_tensor(self, cur_matrix):  # 稀疏矩阵转稀疏tensor
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def weights_init(self, m):
        if isinstance(m, nn.Linear):  # 判断这一层是否为线性层，如果为线性层则初始化权值
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    # self.D, 拟合交互, 多模态交互.detach()
    def gradient_penalty(self, D, xr, xf):

        LAMBDA = 0.3

        xf = xf.detach()  # 不进行梯度传播
        xr = xr.detach()

        alpha = torch.rand(args.batch_size * 2, 1).cuda()
        alpha = alpha.expand_as(xr)  # 将一个张量扩展为指定形状的张量

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()  # 计算梯度

        disc_interpolates = D(interpolates)
        # 计算某个向量损失函数对指定输入的梯度  输出为(batch_size * 2)*n_item
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),  # 全为1的向量  (batch_size * 2)*n_item
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        #  create_graph=True 计算高阶导数 only_inputs=True 返回指定的梯度
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())  # 执行矩阵乘法

    # G_image_user_id[users], G_user_emb[users]
    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / args.tau)  #

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))  # 视觉特征自身对比
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))  # 视觉特征和id对比

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)
            # 公式11
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                    refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                           i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    # G_image_item_embeds, G_text_item_embeds,    G_image_user_embeds, G_text_user_embeds)
    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]  # batch_size*64
        u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()  # batch_size*n_item

        num_batches = (self.n_items - 1) // args.batch_size + 1  # 减1是索引从0开始
        indices = torch.arange(0, self.n_items).cuda()  # item的索引
        u_sim_list = []
        # 逐块求相似度
        for i_b in range(num_batches):
            index = indices[i_b * args.batch_size:(i_b + 1) * args.batch_size]  # 0:1024    1024：2048  ……
            sim = torch.mm(topk_u, item_final[index].T)  # batch_size*batch_size
            sim_gt = torch.multiply(sim, (1 - u_ui[:, index]))  # 逐元素相乘，求没交互过的相似度
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)  # cat后为batch_size*n_item，按行标准化
        return u_sim

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                             self.image_iu_graph, self.text_ui_graph,
                                                             self.text_iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.
        # tb_writer = SummaryWriter(log_dir="/home/ww/Code/work5/MICRO2Ours/tensorboard/")
        # tensorboard_cnt = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            sample_time = 0.
            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model.train()  # 启用 BatchNormalization 和 Dropout
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                with torch.no_grad():  # 不计算梯度
                    ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds, _, _, _, _ = self.model(
                        self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                        self.text_ui_graph, self.text_iu_graph)
                ui_u_sim_detach = self.u_sim_calculation(users, ua_embeddings, ia_embeddings).detach()  # 计算最终表示未交互过得相似度
                image_u_sim_detach = self.u_sim_calculation(users, image_user_embeds,
                                                            image_item_embeds).detach()  # 计算视觉特征未交互过得相似度
                text_u_sim_detach = self.u_sim_calculation(users, text_user_embeds,
                                                           text_item_embeds).detach()  # 计算文本特征未交互过得相似度
                inputf = torch.cat((image_u_sim_detach, text_u_sim_detach), dim=0)  # (batch_size*2）*n_item
                predf = (self.D(inputf))  # 2048*1  (batch_size*2）*1
                lossf = (predf.mean())  # 对多模态特征识别的打分
                u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()  # batch_size*n_item
                u_ui = F.softmax(u_ui - args.log_log_scale * torch.log(-torch.log(
                    torch.empty((u_ui.shape[0], u_ui.shape[1]),
                                dtype=torch.float32).uniform_(0, 1).cuda() + 1e-8) + 1e-8) / args.real_data_tau,
                                 dim=1)  # 公式5前半部分
                u_ui += ui_u_sim_detach * args.ui_pre_scale  # 公式5后半部分
                u_ui = F.normalize(u_ui, dim=1)  # batch_size*n_item    按行标准化
                inputr = torch.cat((u_ui, u_ui), dim=0)  # (batch_size*2）*n_item
                predr = (self.D(inputr))
                lossr = - (predr.mean())  # 对拟合交互矩阵的打分
                gp = self.gradient_penalty(self.D, inputr, inputf.detach())
                loss_D = lossr + lossf + args.gp_rate * gp  # 公式7
                self.optim_D.zero_grad()
                loss_D.backward()
                self.optim_D.step()
                line_d_loss.append(loss_D.detach().data)
                # 未返回  image_item_id, text_item_id
                G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds \
                    , G_user_emb, _, G_image_user_id, G_text_user_id = self.model(self.ui_graph, self.iu_graph,
                                                                                  self.image_ui_graph,
                                                                                  self.image_iu_graph,
                                                                                  self.text_ui_graph,
                                                                                  self.text_iu_graph)

                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings,
                                                                                    G_pos_i_g_embeddings,
                                                                                    G_neg_i_g_embeddings)
                G_image_u_sim = self.u_sim_calculation(users, G_image_user_embeds,
                                                       G_image_item_embeds)  # 计算视觉特征未交互过得相似度
                G_text_u_sim = self.u_sim_calculation(users, G_text_user_embeds, G_text_item_embeds)  # 计算文本特征未交互过得相似度
                G_image_u_sim_detach = G_image_u_sim.detach()
                G_text_u_sim_detach = G_text_u_sim.detach()

                if idx % args.T == 0 and idx != 0:  # args.T为1  前面永远为True  第一次epoch不更新
                    self.image_ui_graph_tmp = csr_matrix((torch.ones(len(self.image_ui_index['x'])),
                                                          # 生成新的稀疏矩阵  交互对象为与user最相似的item
                                                          (self.image_ui_index['x'], self.image_ui_index['y'])),
                                                         shape=(self.n_users, self.n_items))
                    self.text_ui_graph_tmp = csr_matrix(
                        (torch.ones(len(self.text_ui_index['x'])), (self.text_ui_index['x'], self.text_ui_index['y'])),
                        shape=(self.n_users, self.n_items))
                    self.image_iu_graph_tmp = self.image_ui_graph_tmp.T
                    self.text_iu_graph_tmp = self.text_ui_graph_tmp.T
                    self.image_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.image_ui_graph_tmp, mean_flag=True)).cuda()  # 生成交互矩阵
                    self.text_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_ui_graph_tmp, mean_flag=True)).cuda()
                    self.image_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.image_iu_graph_tmp, mean_flag=True)).cuda()
                    self.text_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_iu_graph_tmp, mean_flag=True)).cuda()

                    self.image_ui_index = {'x': [], 'y': []}
                    self.text_ui_index = {'x': [], 'y': []}

                else:
                    _, image_ui_id = torch.topk(G_image_u_sim_detach, int(self.n_items * args.m_topk_rate),
                                                dim=-1)  # 取出batch_size中未交互的图像中每一个user最相似的一个item   #1024*1
                    self.image_ui_index['x'] += np.array(  # 取出user索引
                        torch.tensor(users).repeat(1, int(self.n_items * args.m_topk_rate)).view(-1)).tolist()
                    self.image_ui_index['y'] += np.array(image_ui_id.cpu().view(-1)).tolist()  # 取出item索引
                    _, text_ui_id = torch.topk(G_text_u_sim_detach, int(self.n_items * args.m_topk_rate), dim=-1)
                    self.text_ui_index['x'] += np.array(
                        torch.tensor(users).repeat(1, int(self.n_items * args.m_topk_rate)).view(-1)).tolist()
                    self.text_ui_index['y'] += np.array(text_ui_id.cpu().view(-1)).tolist()
                # 计算用户 项目 多模态正则化嵌入
                feat_emb_loss = self.feat_reg_loss_calculation(G_image_item_embeds, G_text_item_embeds,
                                                               G_image_user_embeds, G_text_user_embeds)
                # 只做了用户嵌入的对比
                batch_contrastive_loss1 = self.batched_contrastive_loss(G_image_user_id[users], G_user_emb[users])
                batch_contrastive_loss2 = self.batched_contrastive_loss(G_text_user_id[users], G_user_emb[users])

                batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

                G_inputf = torch.cat((G_image_u_sim, G_text_u_sim), dim=0)
                G_predf = (self.D(G_inputf))

                G_lossf = -(G_predf.mean())
                batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + \
                             args.cl_rate * batch_contrastive_loss + args.G_rate * G_lossf  # feat_emb_loss

                line_var_loss.append(batch_loss.detach().data)
                line_g_loss.append(G_lossf.detach().data)
                line_cl_loss.append(batch_contrastive_loss.detach().data)

                self.optimizer_D.zero_grad()  # 清空过往梯度
                batch_loss.backward()  # 反向传播，计算中间变量的梯度
                self.optimizer_D.step()  # 进行网络的参数更新

                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)

            del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings,
            del G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings
            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'].data)
            pre_loger.append(ret['precision'].data)
            ndcg_loger.append(ret['ndcg'].data)
            hit_loger.append(ret['hit_ratio'].data)

            line_var_recall.append(ret['recall'][1])
            line_var_precision.append(ret['precision'][1])
            line_var_ndcg.append(ret['ndcg'][1])

            tags = ["recall", "precision", "ndcg"]
            # tb_writer.add_scalar(tags[0], ret['recall'][1], epoch)
            # tb_writer.add_scalar(tags[1], ret['precision'][1], epoch)
            # tb_writer.add_scalar(tags[2], ret['ndcg'][1], epoch)

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train=[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][1], ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (
                    eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break
        self.logger.logging(str(test_ret))

        return best_recall, run_time

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (
                neg_items ** 2).sum()  # 1. / 2 = 0.5
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()
