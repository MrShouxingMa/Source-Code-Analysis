# coding: utf-8

import os
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender

class MMHCL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMHCL, self).__init__(config, dataset)

        self.embeddings_dim = self._as_scalar(config['embedding_size'], 64)
        self.n_ui_layers = int(self._as_scalar(config['n_ui_layers'], 2))
        self.n_user_layers = int(self._as_scalar(config['n_user_layers'], 3))
        self.n_item_layers = int(self._as_scalar(config['n_item_layers'], 2))
        self.user_loss_ratio = float(self._as_scalar(config['user_loss_ratio'], 0.03))
        self.item_loss_ratio = float(self._as_scalar(config['item_loss_ratio'], 0.07))
        self.regs = float(self._as_scalar(config['regs'], 1e-3))
        self.tau = float(self._as_scalar(config['temperature'], 0.6))
        self.topk = int(self._as_scalar(config['topk'], 5))
        self.cf_model = str(self._as_scalar(config['cf_model'], 'lightgcn')).lower()
        self.contrastive_batch_size = int(self._as_scalar(config['contrastive_batch_size'], 4096))
        self.ui_norm = self._as_scalar(config['ui_norm'], 'sym')
        self.u2u_norm = self._as_scalar(config['u2u_norm'], 'rw')
        self.i2i_norm = self._as_scalar(config['i2i_norm'], 'sym')
        self.ngcf_dropout = float(self._as_scalar(config['ngcf_dropout'], 0.1))

        self.user_ui_embedding = nn.Embedding(self.n_users, self.embeddings_dim)
        self.item_ui_embedding = nn.Embedding(self.n_items, self.embeddings_dim)

        self.uu_embedding = nn.Embedding(self.n_users, self.embeddings_dim)
        self.ii_embedding = nn.Embedding(self.n_items, self.embeddings_dim)

        if self.cf_model == 'ngcf':
            self.weight_size = self._parse_weight_size(config['weight_size'])
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(self.ngcf_dropout))

        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.cache_dir = self._build_cache_dir(config)
        self.UI_mat = self.get_UI_mat(self.ui_norm)
        self.User_mat = self.get_U2U_mat(self.u2u_norm)
        self.Item_mat = self.get_I2I_Hypergraph_mul_mat(self.i2i_norm)

    @staticmethod
    def _as_scalar(value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    def _parse_weight_size(self, value):
        value = self._as_scalar(value, None)
        if value is None:
            return [self.embeddings_dim] * (self.n_ui_layers + 1)
        if isinstance(value, str):
            value = eval(value)
        value = list(value)
        if len(value) < self.n_ui_layers + 1:
            value = value + [value[-1]] * (self.n_ui_layers + 1 - len(value))
        return value

    def _build_cache_dir(self, config):
        cache_root = config['graph_cache_path'] or './graph_cache/'
        cache_dir = os.path.abspath(os.path.join(cache_root, config['dataset']))
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _cache_path(self, name):
        return os.path.join(self.cache_dir, name)

    def _load_sparse_cache(self, name):
        cache_file = self._cache_path(name)
        if os.path.isfile(cache_file):
            # return torch.load(cache_file, map_location=self.device).to(self.device)
            return torch.load(cache_file).to(self.device)
        return None

    def _save_sparse_cache(self, tensor, name):
        torch.save(tensor.detach().cpu(), self._cache_path(name))

    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        matrix = matrix.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack((matrix.row, matrix.col)))
        values = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(self.device)

    # ---------------------------------------Own--------------------------------------------------
    def norm_sparse(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = np.array(adj.sum(1)).flatten()
            with np.errstate(divide='ignore'):
                d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        if normalization == 'rw':
            rowsum = np.array(adj.sum(1)).flatten()
            with np.errstate(divide='ignore'):
                d_inv = np.power(rowsum, -1.0)
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv.dot(adj).tocoo()
        if normalization == 'origin':
            return adj.tocoo()
        raise ValueError('Unsupported sparse normalization: {}'.format(normalization))

    def norm_dense(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == '2sym':
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.
            d_row_mat_inv_sqrt = torch.diagflat(d_row_inv_sqrt)

            colsum = torch.sum(adj, -2)
            d_col_inv_sqrt = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.
            d_col_mat_inv_sqrt = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt)
        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)
        elif normalization == 'origin':
            L_norm = adj
        else:
            raise ValueError('Unsupported dense normalization: {}'.format(normalization))
        return L_norm

    def get_UI_mat(self, norm_type='sym'):
        #   UI_mat default use sym normalization,and No-self-connection
        cache_name = 'UI_mat_{}_u{}_i{}.pth'.format(norm_type, self.n_users, self.n_items)
        UI_mat = self._load_sparse_cache(cache_name)
        if UI_mat is not None:
            return UI_mat

        t = time()
        inter_M = self.interaction_matrix.tocoo()
        inter_M_t = inter_M.transpose().tocoo()
        n_nodes = self.n_users + self.n_items
        rows = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        cols = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones_like(rows, dtype=np.float32)
        adj_mat = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        UI_mat = self.scipy_matrix_to_sparse_tensor(self.norm_sparse(adj_mat, norm_type), torch.Size((n_nodes, n_nodes)))
        self._save_sparse_cache(UI_mat, cache_name)
        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_U2U_mat(self, norm_type='rw'):
        # U2U_mat default use row normalization,and No-self-connection
        cache_name = 'User_mat_{}_u{}_i{}.pth'.format(norm_type, self.n_users, self.n_items)
        User_mat = self._load_sparse_cache(cache_name)
        if User_mat is not None:
            return User_mat

        t = time()
        R = self.interaction_matrix.tocsr()
        User_mat = R.dot(R.transpose()).tolil()  # 计算共现交互数量
        User_mat.setdiag(0)  # 抹去自连接
        User_mat = User_mat.tocoo()
        User_mat.eliminate_zeros()
        User_mat = self.scipy_matrix_to_sparse_tensor(
            self.norm_sparse(User_mat, norm_type),
            torch.Size((self.n_users, self.n_users))
        )
        self._save_sparse_cache(User_mat, cache_name)
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    def _modal_features(self):
        feats = []
        if self.v_feat is not None:
            feats.append(('image', self.v_feat[:self.n_items]))
        if self.t_feat is not None:
            feats.append(('text', self.t_feat[:self.n_items]))
        if self.a_feat is not None:
            feats.append(('audio', self.a_feat[:self.n_items]))
        if not feats:
            raise RuntimeError('MMHCL requires at least one modal feature matrix.')
        return feats

    # Order to speed up when Model forward this is be replaced
    def get_I2I_Hypergrah_mat(self, norm_type='origin'):
        # I2I_Hypergraph_mat use origin normalization
        modal_names = '-'.join(name for name, _ in self._modal_features())
        cache_name = 'hypergraph_mat_{}_topk_{}_{}.pth'.format(norm_type, self.topk, modal_names)
        Hypergraph = self._load_sparse_cache(cache_name)
        if Hypergraph is not None:
            return Hypergraph

        t = time()
        adj_list = []
        for _, feat in self._modal_features():
            modal_adj = self.build_sim(feat)  # 做L2归一化 然后计算相似度
            modal_adj = self.build_knn_normalized_graph(modal_adj, topk=self.topk)
            adj_list.append(modal_adj)

        Hypergraph = torch.cat(adj_list, dim=1)
        Hypergraph = self.norm_dense(Hypergraph, norm_type)  # norm_type为origin  没进行任何处理
        Hypergraph = Hypergraph.to_sparse().coalesce()
        self._save_sparse_cache(Hypergraph, cache_name)
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type='sym'):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        modal_names = '-'.join(name for name, _ in self._modal_features())
        cache_name = 'hypergraph_mat_mul_{}_topk_{}_{}.pth'.format(norm_type, self.topk, modal_names)
        Hypergraph_mul = self._load_sparse_cache(cache_name)
        if Hypergraph_mul is not None:
            return Hypergraph_mul

        t = time()
        Hypergraph = self.get_I2I_Hypergrah_mat('origin')
        Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)  # 高阶的共邻关系
        Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)  # 进行对称归一化
        Hypergraph_mul = Hypergraph_mul.to_sparse().coalesce()
        self._save_sparse_cache(Hypergraph_mul, cache_name)
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True).clamp_min(1e-12))
        context_norm[torch.isnan(context_norm)] = 0
        context_norm[torch.isinf(context_norm)] = 0
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_knn_normalized_graph(self, adj, topk):
        topk = min(topk, adj.shape[-1])
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.
        return adj

    def forward(self):
        ii_emb = self.ii_embedding.weight
        uu_emb = self.uu_embedding.weight

        if self.item_loss_ratio != 0:
            for _ in range(self.n_item_layers):
                ii_emb = torch.sparse.mm(self.Item_mat, ii_emb)  # [items, items]* [items, feat_dim]

        if self.user_loss_ratio != 0:
            for _ in range(self.n_user_layers):
                uu_emb = torch.sparse.mm(self.User_mat, uu_emb)

        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(self.UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(self.UI_mat, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        elif self.cf_model == 'mf':
            u_ui_emb, i_ui_emb = self.user_ui_embedding.weight, self.item_ui_embedding.weight
        else:
            raise ValueError('Unsupported cf_model: {}'.format(self.cf_model))

        if self.item_loss_ratio != 0:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

        if self.user_loss_ratio != 0:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return u_ui_emb, i_ui_emb, ii_emb, uu_emb

    # 单项，但是负样本包含除自身外所有样本，包括同特征但是不同样本的对比
    def batched_contrastive_loss(self, z1, z2, batch_size=None):
        if batch_size is None:
            batch_size = self.contrastive_batch_size
        device = z1.device   # 18357-64  [items, feat_dim]
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.regs * regularizer
        reg_loss = torch.zeros((), device=users.device)
        return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, ii, uu = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        batch_contrastive_loss1 = torch.zeros((), device=self.device)
        batch_contrastive_loss2 = torch.zeros((), device=self.device)
        if self.item_loss_ratio != 0:
            batch_contrastive_loss1 = self.batched_contrastive_loss(ia_embeddings, ii)
            batch_contrastive_loss1 *= self.item_loss_ratio
        if self.user_loss_ratio != 0:
            batch_contrastive_loss2 = self.batched_contrastive_loss(ua_embeddings, uu)
            batch_contrastive_loss2 *= self.user_loss_ratio

        batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

        return batch_mf_loss, batch_emb_loss + batch_reg_loss, batch_contrastive_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores
