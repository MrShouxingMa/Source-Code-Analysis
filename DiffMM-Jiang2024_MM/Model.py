import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self, image_embedding, text_embedding, audio_embedding=None):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge(args.keepRate)

        if args.trans == 1:  # tiktok和sport是1，图像、文本、音频都用 训练一个线性投影
            self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
        elif args.trans == 0:  # baby是0， 不是 Linear，没有 bias，参数更少，更简单
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
        else:  # 文本用 Linear，其余用 Parameter（更灵活、文本更复杂）
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
        if audio_embedding != None:
            if args.trans == 1:  # 训练一个线性投影
                self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
            else:
                self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        if audio_embedding != None:
            self.audio_embedding = audio_embedding
        else:
            self.audio_embedding = None

        if audio_embedding != None:
            self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
        else:
            self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))  # 两个模态的权重均分
        self.softmax = nn.Softmax(dim=0)

        self.dropout = nn.Dropout(p=0.1)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def getItemEmbeds(self):
        return self.iEmbeds

    def getUserEmbeds(self):
        return self.uEmbeds

    def getImageFeats(self):
        if args.trans == 0 or args.trans == 2:  # 0和2都是直接使用参数拟合
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            return image_feats
        else:
            return self.image_trans(self.image_embedding)

    def getTextFeats(self):
        if args.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
            return text_feats
        else:
            return self.text_trans(self.text_embedding)

    def getAudioFeats(self):
        if self.audio_embedding == None:
            return None
        else:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)
        return audio_feats

    def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
        # 如果 args.trans == 0（不使用线性层进行转换），则使用 leakyrelu 激活函数对图像和文本特征进行变换
        if args.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        # 如果 args.trans == 1，使用线性层进行特征变换
        elif args.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        if audio_adj != None:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)

            # 通过 softmax 对模态权重进行归一化
        weight = self.softmax(self.modal_weight)

        # 视觉邻接矩阵处理：拼接用户和项目ID嵌入，先进行生成特定的去噪视觉多模态交互图卷积  equation20最后一项
        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        # 处理图像特征：拼接用户ID嵌入和标准化后的图像特征，在用户物品交互图上进行图卷积 equation20第一项
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)

        # 再次更新图像特征，使用从用户ID嵌入和项目嵌入的组合进行图卷积   equation20第二项
        embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_

        # 文本邻接矩阵处理：拼接用户和项目嵌入，进行图卷积
        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        # 处理文本特征：拼接用户嵌入和标准化后的文本特征，并进行图卷积
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)

        # 再次更新文本特征，使用从用户嵌入和项目嵌入的组合进行图卷积
        embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_

        if audio_adj != None:
            embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
            embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

            embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
            embedsAudio = torch.spmm(adj, embedsAudio)

            embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
            embedsAudio_ = torch.spmm(adj, embedsAudio_)
            embedsAudio += embedsAudio_

        # 加入 RIS（Residual Information Smoothing）正则化项，对图像、文本的特征进行额外的邻接矩阵处理  更像是在公式20中对最后项加了个系数
        embedsImage += args.ris_adj_lambda * embedsImageAdj
        embedsText += args.ris_adj_lambda * embedsTextAdj

        if audio_adj != None:
            embedsAudio += args.ris_adj_lambda * embedsAudioAdj

        # 加权多模态特征的融合  eq21
        if audio_adj == None:
            embedsModal = weight[0] * embedsImage + weight[1] * embedsText
        else:
            embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

        # 将多模态融合后的嵌入输入到 GCN 层中，进行多层图卷积  eq22
        embeds = embedsModal
        embedsLst = [embeds]  # 保存每一层的嵌入
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)  # 将每一层的嵌入结果相加

        # 加入 RIS 正则化项，对最终的嵌入结果进行归一化处理 eq23
        embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

        # 返回用户嵌入和项目嵌入
        return embeds[:args.user], embeds[args.user:]

    def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
        # 使用 leakyrelu 激活函数和矩阵乘法转换图像和文本特征
        if args.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        # 使用线性层转换图像和文本特征
        elif args.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        if audio_adj != None:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)

        # 将用户嵌入和标准化后的图像特征拼接，使用图像邻接矩阵进行图卷积
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)

        # 将用户嵌入和标准化后的文本特征拼接，使用文本邻接矩阵进行图卷积
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)

        if audio_adj != None:
            embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
            embedsAudio = torch.spmm(audio_adj, embedsAudio)

        # 对图像特征进行多层图卷积处理
        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:  # 遍历 GCN 层，进行图卷积
            embeds1 = gcn(adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)  # 将每一层的图卷积结果相加，形成最终的图像嵌入

        # 对文本特征进行多层图卷积处理
        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:  # 遍历 GCN 层，进行图卷积
            embeds2 = gcn(adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)  # 将每一层的图卷积结果相加，形成最终的文本嵌入

        if audio_adj != None:
            embeds3 = embedsAudio
            embedsLst3 = [embeds3]
            for gcn in self.gcnLayers:
                embeds3 = gcn(adj, embedsLst3[-1])
                embedsLst3.append(embeds3)
            embeds3 = sum(embedsLst3)

        if audio_adj == None:
            return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
        else:
            return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[
                                                                                                       :args.user], embeds3[
                                                                                                                    args.user:]

    def reg_loss(self):
        ret = 0
        ret += self.uEmbeds.norm(2).square()
        ret += self.iEmbeds.norm(2).square()
        return ret


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()  # 取得所有非零元素的值
        idxs = adj._indices()  # 返回稀疏矩阵的非零元素位置
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)  # 邻接矩阵随机 DropEdge 机制,keepRate为保留率

        newVals = vals[mask] / self.keepRate  # DropEdge 会丢掉一些边，但为了保持图卷积期望一致，对保留下来的边用 1/keepRate 做补偿，从而让整体图结构的统计性质不变
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        # 输入和输出维度的列表，以及时间嵌入维度
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        # 定义时间嵌入的线性层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 输入层的维度：将时间嵌入与原始输入数据的第一个维度（如特征维度）相加  [num_item + time_emb_dim, 1000]
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        # [1000, num_item] 输出层重新变回原来的维度
        out_dims_temp = self.out_dims

        # 定义输入层的多层线性变换（使用 ModuleList 保存多层的 nn.Linear）
        # num_item + time_emb_dim >> 1000
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 1000 >> num_item
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        # dropout=0.5
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))  # xavier_normal 初始化
            layer.weight.data.normal_(0.0, std)  # 使用正态分布初始化权重
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为一个较小的随机值

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化时间嵌入层的权重
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        # 计算时间嵌入，使用正弦和余弦位置编码
        # torch.arange 生成一个从 0 到 time_emb_dim // 2 的张量（指数衰减的频率）用于时间编码,[1, 10000−1/(d/2), 10000−2/(d/2), …],d为time_emb_dim
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                self.time_emb_dim // 2)).cuda()

        # 将 timesteps 扩展到相应的维度，并与 freqs 相乘以得到时间嵌入
        temp = timesteps[:, None].float() * freqs[None]  # [B, 1]*[1, d/2]=[B, d/2]

        # 使用 cos 和 sin 函数构造时间嵌入， [B, d/2]||[B, d/2]=[B, d]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

        # 如果时间嵌入维度是奇数，补齐为偶数维度，最终维度正好是 time_emb_dim，#补0是因为最简单、最安全的占位，不会引入额外信息（比随机值更稳定）
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        # 将时间嵌入通过线性层进行处理  [batchsize, d_emb_size(10)]
        emb = self.emb_layer(time_emb)
        #对生成噪声的 正则化和dropout
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)

        # 将输入 x 和时间嵌入 emb 进行拼接，作为输入层的输入  [batchsize,   item_num+ time_d_emb_size(10)]  [1024,7060]
        h = torch.cat([x, emb], dim=-1)

        # 依次通过每一层输入层的线性变换，并使用 tanh 激活函数
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        # 依次通过每一层输出层的线性变换，除了最后一层不使用激活函数
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1: #判断是不是最后一层 如果不是执行下面代码
                h = torch.tanh(h)

        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        # 扩散过程中的噪声相关参数
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps  # 扩散的步数

        # 控制扩散噪声强度的全局参数，如果噪声比例不为0，计算每一步的噪声系数 beta
        if noise_scale != 0:  # 不为零的意思就是使用扩散模型
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001  # 这里其实与DDPM有冲突，理论上第一次加噪要小，按代码计算出为1e-5（start值），后面一次变大，但这里写死1e4

            self.calculate_for_diffusion()

    # 计算扩散过程中的 beta 系数，用于在每一步添加噪声
    def get_betas(self):
        start = self.noise_scale * self.noise_min  # 噪声的起始值
        end = self.noise_scale * self.noise_max  # 噪声的结束值
        # 在扩散步数范围内线性插值，得到每一步的方差
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance  # 计算 alpha_bar，用于表示去噪过程中的保持率
        betas = []
        betas.append(1 - alpha_bar[0])  # 初始 beta 值

        # 逐步计算每一步的 beta 值
        for i in range(1, self.steps):
            betas.append(
                min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))  # 取min是DDPM要求beta取值在（0,1），大于1没法1-beta没法开方，方差不能为负

        # 返回 beta 的数组
        return np.array(betas)

    # 计算扩散和去噪过程中需要的参数
    def calculate_for_diffusion(self):
        # alpha 用于表示每一步中加噪后保留的数据比例
        alphas = 1.0 - self.betas
        # 计算 alpha 的累积乘积，即 alpha 的逐步积累过程[a1, a1*a2, a1*a2*a3]
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        # 前一步的 alpha 累积乘积，初始时假设为 1 (根据公式设定的)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        # 下一步的 alpha 累积乘积，最后一步假设为 0
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        # 计算 alpha 累积乘积的平方根，用于加噪过程中保留的比例
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # 计算 1 - alpha 累积乘积的平方根，用于加噪过程中噪声的比例
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # 计算 log(1 - alpha 累积乘积)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        # 计算 alpha 累积乘积的倒数平方根
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        # 计算 1/alpha 累积乘积 - 1 的平方根，用于后续采样的方差调整
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算后验分布的方差，公式来源于扩散模型中后验的推导：
        # betas * (1 - 前一步 alpha 累积乘积) / (1 - 当前步的 alpha 累积乘积)
        # eq8中的方差
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        # 计算后验方差的对数，并将第一个元素固定为后续计算方便。DDPM中设置为一个很小的值和1e-20比，本论文直接设置为第一个元素
        # 第一个 posterior variance 为 0，直接做 log(0) 就是 −∞，后续很容易在 NLL/VLB 计算里出现 inf/NaN
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))

        # 计算后验均值的两个系数，分别用于表示在去噪过程中均值的线性组合  xₜ → xₜ₋₁，_prev是有1的<==>上一时刻的扩散过程累计保留率，另一个是当前时刻的累计保留率
        # 系数 1：betas * sqrt(前一步 alpha 累积乘积) / (1 - 当前步 alpha 累积乘积)  eq8和eq10中后面一项的系数
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # 系数 2：(1 - 前一步 alpha 累积乘积) * sqrt(alpha) / (1 - 当前步 alpha 累积乘积)  eq8和eq10中前面一项的系数
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    # 从给定的初始状态 x_start 中逐步采样，恢复出原始数据
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # 如果步数是 0，直接使用初始用户-项目交互序列
        if steps == 0:
            x_t = x_start
        else:
            # 构造一个长度为 x_start 的 t 张量，值为 steps - 1，用于从扩散过程中提取样本
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            # 调用 q_sample 函数，生成带噪声的 x_t
            x_t = self.q_sample(x_start, t)

        # 创建一个索引列表，表示反向采样步骤的顺序，从 steps-1 到 0
        indices = list(range(self.steps))[::-1]

        # 逐步执行从 t = steps-1 到 t = 0 的采样过程
        for i in indices:
            # 为每一个步数创建一个 t 张量
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            # 通过模型计算后验均值和对数方差
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            # 如果开启了采样噪声，则加入噪声(想尽量还原一个确定的交互向量) True：有随机性、多样性 False：更平滑、更稳定、但少了随机性
            if sampling_noise:
                # 生成与 x_t 形状相同的标准正态噪声
                noise = torch.randn_like(x_t)
                # 确保在时间步t=0时不会加噪声
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                # 更新 x_t，基于模型的均值和噪声
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean  # 如果不加噪声，直接使用均值作为下一步的at-1
        return x_t

    # 执行扩散模型中的前向过程，它在每一步中向数据中加入噪声 eq2
    def q_sample(self, x_start, t, noise=None):
        # x_start代表论文中的a0，表示原始用户项目交互序列
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取 alpha 的平方根并对 x_start 加权,获得\sqrt{\bar{\alpha}_t}\,x_0 （DDPM 公式4 第1项）
        alpha_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start

        # 提取 (1 - alpha) 的平方根并对噪声加权,获得(1 − α ̄t)*噪音 （DDPM 公式4 第2项）
        one_minus_alpha = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return alpha_t + one_minus_alpha

    # 从给定的数组 arr (sqrt_alphas_cumprod) 中提取与时间步 t 对应的值，并扩展维度以适应 broadcast_shape
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        # 根据时间步 t 提取数组中对应的值，并将其转换为浮点数
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):  # 验证维度是都一致
            res = res[..., None]  # 增加维度以适合广播
        return res.expand(broadcast_shape)  # 不占内存的扩展维度

    # 根据模型输出和扩散过程中的时间步 t，计算模型的均值和方差  eq4的均值和方差，这里的输入x是x_start,即x0
    def p_mean_variance(self, model, x, t):
        # 使用模型输出，假设模型根据输入 x(at) 和时间步 t 返回结果
        model_output = model(x, t, False)  #时刻t下的噪声

        # 后验分布q(x_{t-1}|x_t,x0)的方差和对数方差，已经预先计算好
        # model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        # 根据时间步 t 从方差和对数方差中提取对应的值，并扩展到输入 x 的形状
        # model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        # 计算后验均值。通过 posterior_mean_coef1 和 posterior_mean_coef2 加权模型输出和输入 x
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        # 返回模型均值和对数方差
        return model_mean, model_log_variance

    # ELBO 损失  （self.denoise_model_image, batch_item, Embeds, batch_index, image_feats)
    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        batch_size = x_start.size(0)

        # 随机选择时间步 ts，范围为 0 到 self.steps
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()  # torch.Size([1024])
        # 生成与 x_start 形状相同的随机噪声(正态分布 N(0,1))
        noise = torch.randn_like(x_start)
        # 如果噪声比例不为 0，执行前向加噪扩散过程生成 x_t (随机t时刻下加噪后的x，这里指用户和项目交互概率的加噪)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts,
                                noise)  # 生成带噪声的用户项目交互序列  这里的x_start 是[1024, 7050(item num)]，是每个1024个user对所有item交互值
        else:
            x_t = x_start

        # 通过模型生成预测输出(去噪过程p)
        model_output = model(x_t, ts)

        # 计算均方误差 MSE，L0部分(eq12)  “去掉第一维”是想不在这里把 batch 合并掉，保留每个样本的 loss 以便逐样本加权
        mse = self.mean_flat((x_start - model_output) ** 2)

        # 计算 ts-1 和 ts 之间的 SNR 差异，用于权重调节
        # weight 计算了时间步 $t$ 上的 SNR 差异，这反映了不同时间步 KL 散度的加权
        weight = self.SNR(ts - 1) - self.SNR(ts)
        # 如果时间步 ts 为 0，则将权重设置为 1.0（即不衰减）
        weight = torch.where((ts == 0), 1.0, weight)

        # diff_loss 是加权后的 ELBO 损失
        diff_loss = weight * mse

        # ==============模态感知信号注入===================
        # 计算用户模型嵌入与模型特征之间的点积,加噪后输出的交互概率*item的多模态信息
        usr_model_embeds = torch.mm(model_output, model_feats)  #[1024, 7050]*[7050, 64]
        # 计算用户 ID 嵌入与物品嵌入ID信息之间的点积
        usr_id_embeds = torch.mm(x_start, itmEmbeds)

        # gc_loss，衡量用户模型嵌入和用户 ID 嵌入之间的差异(eq14中的msi损失)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss

    def mean_flat(self, tensor):
        # 计算张量 tensor 除了第一维度外的均值
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    # 计算扩散过程中的信噪比
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
