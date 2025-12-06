import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args

args = parse_args()


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path  # + '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/train.json'  # + '/%d-core/train.json' % (args.core)
        val_file = path + '/val.json'  # + '/%d-core/val.json' % (args.core)
        test_file = path + '/test.json'  # + '/%d-core/test.json'  % (args.core)

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_val, self.n_test = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue
        # 找到数据集总最大item和user,并计数（均从0开始，所以+1）
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()
        # 创建并赋值稀疏矩阵
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for _, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items  # {0:[1,2,3],1:[4,5,6]}  把train的key从str变成了int了

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]  # 否则运行batch_size次，每次选一个

        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]  # 找到所有训练样本中与用户交互的item
            n_pos_items = len(pos_items)  # 计算数量
            pos_batch = []
            while True:
                if len(pos_batch) == num: break  # 采集到符合条件的终止
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]  # 选出索引
                pos_i_id = pos_items[pos_id]  # 取出正样本

                if pos_i_id not in pos_batch:  # 判定正样本 之前没有取出
                    pos_batch.append(pos_i_id)  # 存入
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:   # 判定负样本之前没有取出并且不在训练的正样本中
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_val=%d,n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_val, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
