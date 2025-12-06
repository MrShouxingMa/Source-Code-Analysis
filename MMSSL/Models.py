import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args

args = parse_args()


class MMSSL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, image_feats, text_feats):

        super().__init__()
        self.n_users = n_users  # 19805
        self.n_items = n_items  # 10068
        self.embedding_dim = embedding_dim  # 64
        self.weight_size = weight_size  # [64，64]
        self.n_ui_layers = len(self.weight_size)  # 2
        self.weight_size = [self.embedding_dim] + self.weight_size  # [64, 64, 64]

        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        self.encoder = nn.ModuleDict()
        self.encoder['image_encoder'] = self.image_trans
        self.encoder['text_encoder'] = self.text_trans

        self.common_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.common_trans.weight)
        self.align = nn.ModuleDict()
        self.align['common_trans'] = self.common_trans

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)  # freeze=F  训练过程更新
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)  # args.drop_rate=0.2
        self.batch_norm = nn.BatchNorm1d(args.embed_size)  # args.embed_size=64
        self.tau = 0.5

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_k': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_v': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_cat': nn.Parameter(
                initializer(torch.empty([args.head_num * args.embed_size, args.embed_size]))),
        })
        self.embedding_dict = {'user': {}, 'item': {}}

    def mm(self, x, y):  # 稀疏矩阵乘法
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):  # 归一化后做矩阵乘法
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def para_dict_to_tenser(self, para_dict):
        """
        :param para_dict: nn.ParameterDict()    {['image']:id , ['text'] :id}  N*64
        :return: tensor
        """
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)  # 'image'  ||text
        return tensors  # 2*N*64

    # self.weight_dict, self.embedding_dict['user'],   self.embedding_dict['user']
    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):

        q = self.para_dict_to_tenser(embedding_t)  # 2*N*64
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.embed_size / args.head_num  # beh=2   N=19805   d_h=64/4=16

        Q = torch.matmul(q, trans_w['w_q'])  # 矩阵向量相乘   2*N*64
        K = torch.matmul(k, trans_w['w_k'])  # 矩阵向量相乘   2*N*64
        V = v

        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  # 2*N*4*16  =>  4*2*N*16
        K = K.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  # 2*N*4*16  =>  4*2*N*16

        Q = torch.unsqueeze(Q, 2)  # 4*2*N*16   =>   4*2*1*N*16
        K = torch.unsqueeze(K, 1)  # 4*2*N*16   =>   4*1*2*N*16
        V = torch.unsqueeze(V, 1)  # 2*N*16   =>   2*1*N*64

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  # （4*2*1*N*16）（4*1*2*N*16）=4*2*2*N*16  做点击运算，包含广播机制
        att = torch.sum(att, dim=-1)  # 4*2*2*N
        att = torch.unsqueeze(att, dim=-1)  # 4*2*2*N*1
        att = F.softmax(att, dim=2)  # 按照第4*2*“2”*N*1 归一化  4*2*2*N*1

        Z = torch.mul(att, V)  # （4*2*2*N*1）(2*1*N*64）=4*2*2*N*64
        Z = torch.sum(Z, dim=2)  # 4*2*N*64

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])  # (2*n*256)*(256*64)=2*N*64

        args.model_cat_rate * F.normalize(Z, p=2, dim=2)
        return Z, att.detach()  # detach()不对该tensor进行梯度计算

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):

        image_feats = image_item_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = text_item_feats = self.dropout(self.text_trans(self.text_feats))

        for i in range(args.layers):  # 只有一层
            image_user_feats = self.mm(ui_graph, image_feats)  # 公式2
            image_item_feats = self.mm(iu_graph, image_user_feats)  # 公式2
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)  # 仅仅左乘归一
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id
        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'],
                                                   self.embedding_dict['user'])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])
        user_emb = user_z.mean(0)  # 2*N*64 => N*64
        item_emb = item_z.mean(0)
        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate * F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate * F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):#进行2层GNN
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = u_g_embeddings + \
                         args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1) + \
                         args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + \
                         args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1) + \
                         args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id
        # return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 4)),
            nn.LeakyReLU(True),  # 直接修改输入数据
            nn.BatchNorm1d(int(dim / 4)),  # 输入为特征数
            nn.Dropout(args.G_drop1),

            nn.Linear(int(dim / 4), int(dim / 8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(args.G_drop2),

            nn.Linear(int(dim / 8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = 100 * self.net(x.float())
        return output.view(-1)
