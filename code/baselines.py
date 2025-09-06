import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import functions as fn
import copy

# 设置设备（GPU或CPU）
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)  # 设置随机种子以确保结果可复现


class VAR(nn.Module):
    """向量自回归模型 (Vector AutoRegression)
    简单的线性基准模型，将时空序列展平后通过全连接层进行预测
    """

    def __init__(self, node=247, seq=12, feature=2):  # input_dim = seq_length
        super(VAR, self).__init__()
        self.linear = nn.Linear(node * seq * feature, node)  # 全连接层

    def forward(self, occ, prc):
        x = torch.cat((occ, prc), dim=2)  # 拼接占位和价格特征
        x = torch.flatten(x, 1, 2)  # 展平时空维度
        x = self.linear(x)  # 线性变换
        return x


class LSTM(nn.Module):
    """长短期记忆网络模型
    使用CNN编码特征后，通过LSTM捕捉时间依赖关系
    """

    def __init__(self, seq, n_fea, node=247):
        super(LSTM, self).__init__()
        self.nodes = node
        # 2D卷积编码器，处理时空特征
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))  # input.shape: [batch, channel, width, height]
        # LSTM层捕捉时间依赖
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        # 解码器，将序列映射为单步预测
        self.decoder = nn.Linear(seq - n_fea + 1, 1)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)  # 堆叠特征
        x = self.encoder(x)  # 卷积编码
        x = torch.transpose(x.squeeze(), 1, 2)  # shape [batch, seq-n_fea+1, node]
        x, _ = self.lstm(x)  # LSTM处理
        x = torch.transpose(x, 1, 2)  # shape [batch, node, seq-n_fea+1]
        x = self.decoder(x)  # 解码预测
        x = torch.squeeze(x)
        return x


class GCN(nn.Module):
    """图卷积网络模型
    使用图卷积捕捉空间依赖关系
    """

    def __init__(self, seq, n_fea, adj_dense):
        super(GCN, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        # 图卷积层
        self.gcn_l1 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1)
        self.gcn_l2 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1)
        self.A = adj_dense  # 邻接矩阵
        self.act = nn.ReLU()  # 激活函数
        self.decoder = nn.Linear(seq - n_fea + 1, 1)

        # 计算归一化的邻接矩阵 (A_hat = D^(-1/2) A D^(-1/2))
        deg = torch.sum(adj_dense, dim=0)  # 度矩阵
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))  # D^(-1/2)
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)  # 归一化邻接矩阵
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        # 第一层图卷积
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)  # 图传播
        x = self.act(x)
        # 第二层图卷积
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        x = self.decoder(x)
        return x


class LstmGcn(nn.Module):
    """LSTM与GCN的混合模型
    结合图卷积的空间建模能力和LSTM的时间建模能力
    """

    def __init__(self, seq, n_fea, adj_dense):
        super(LstmGcn, self).__init__()
        self.A = adj_dense
        self.nodes = adj_dense.shape[0]
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gcn_l1 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)
        self.gcn_l2 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # 计算归一化的邻接矩阵
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        # 第一层图卷积
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        # 第二层图卷积
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        # LSTM处理时间维度
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class LstmGat(nn.Module):
    """LSTM与图注意力网络的混合模型
    使用注意力机制动态学习节点间的重要性
    """

    def __init__(self, seq, n_fea, adj_dense, adj_sparse):
        super(LstmGat, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.gcn = nn.Linear(in_features=seq - n_fea + 1, out_features=seq - n_fea + 1, device=device)
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        # 多头图注意力层
        self.gat_l1 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.gat_l2 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # 激活函数和正则化
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        # 第一层图注意力
        atts_mat = self.gat_l1(x)  # 注意力矩阵, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, x)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # 第二层图注意力
        atts_mat2 = self.gat_l2(occ_conv1)  # 注意力矩阵, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        # LSTM处理时间维度
        x = occ_conv2.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # 修正：移除了多余的x参数

        # 解码
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class TPA(nn.Module):
    """时序模式注意力模型 (Temporal Pattern Attention)
    使用注意力机制捕捉时间模式
    """

    def __init__(self, seq, n_fea):
        super(TPA, self).__init__()
        # 注意：此实现不完整，缺少nodes和seq的定义
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        # TPA组件
        self.lstm = nn.LSTM(2, 2, num_layers=2, batch_first=True, device=device)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=2, device=device)
        self.fc2 = nn.Linear(in_features=2, out_features=2, device=device)
        self.fc3 = nn.Linear(in_features=2 + 2, out_features=1, device=device)
        self.decoder = nn.Linear(self.seq, 1, device=device)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        # TPA机制
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2
        ht = lstm_out[:, -1, :]  # 最后时间步的隐藏状态
        hw = lstm_out[:, :-1, :]  # 从h(t-1)到h1的所有隐藏状态
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.unsqueeze(ht, dim=2)
        a = torch.bmm(Hn, ht)  # 计算注意力权重
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)  # 加权和
        ht = torch.transpose(ht, 1, 2)
        hx = torch.cat((vt, ht), dim=2)  # 拼接
        y = self.fc3(hx)  # 最终预测
        print(y.shape)
        return y


# https://doi.org/10.1016/j.trc.2023.104205
class HSTGCN(nn.Module):
    """混合时空图卷积网络 (Hybrid Spatio-Temporal Graph Convolutional Network)
    同时考虑距离和需求两种图结构
    """

    def __init__(self, seq, n_fea, adj_distance, adj_demand, alpha=0.5):
        super(HSTGCN, self).__init__()
        # 超参数
        self.nodes = adj_distance.shape[0]
        self.alpha = alpha  # 距离图和需求图的权重
        hidden = seq - n_fea + 1

        # 网络组件
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        self.linear = nn.Linear(hidden, hidden)
        # 距离图分支
        self.distance_gcn_l1 = nn.Linear(hidden, hidden)
        self.distance_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru1 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        # 需求图分支
        self.demand_gcn_l1 = nn.Linear(hidden, hidden)
        self.demand_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru2 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # 计算距离图的归一化邻接矩阵
        deg = torch.sum(adj_distance, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_distance), deg_delta)
        self.A_dis = a_delta

        # 计算需求图的归一化邻接矩阵
        deg = torch.sum(adj_demand, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_demand), deg_delta)
        self.A_dem = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        x = self.act(self.linear(x))  # 初始特征变换

        # 距离图传播分支
        # 第一层
        x1 = self.distance_gcn_l1(x)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        # 第二层
        x1 = self.distance_gcn_l2(x1)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        # GRU处理时间维度
        x1 = x1.transpose(1, 2)
        x1, _ = self.gru1(x1)
        x1 = x1.transpose(1, 2)

        # 需求图传播分支
        # 第一层
        x2 = self.demand_gcn_l1(x)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        # 第二层
        x2 = self.demand_gcn_l2(x2)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        # GRU处理时间维度
        x2 = x2.transpose(1, 2)
        x2, _ = self.gru2(x2)
        x2 = x2.transpose(1, 2)

        # 融合两个分支的结果并解码
        output = self.alpha * x1 + (1 - self.alpha) * x2  # 加权融合
        output = self.decoder(output)
        output = torch.squeeze(output)
        return output


# https://arxiv.org/abs/2311.06190
class FGN(nn.Module):
    """傅里叶图网络 (Fourier Graph Network)
    在频域进行图卷积操作
    """

    def __init__(self, pre_length=1, embed_size=64,
                 feature_size=0, seq_length=12, hidden_size=32, hard_thresholding_fraction=1, hidden_size_factor=1,
                 sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # 可学习的嵌入向量

        self.encoder = nn.Linear(2, 1)
        # 傅里叶变换的参数
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        # 全连接解码器
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')  # 确保模型在GPU上

    def tokenEmb(self, x):
        """标记嵌入函数"""
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        """傅里叶图卷积"""
        o1_real = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        # 第一层傅里叶变换
        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 第一层
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)  # 软阈值化，用于去噪

        # 第二层傅里叶变换
        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 第二层
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y  # 残差连接

        # 第三层傅里叶变换
        o3_real = F.relu(
            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
            self.b3[1]
        )

        # 第三层
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x  # 残差连接
        z = torch.view_as_complex(z)  # 转换为复数形式
        return z

    def forward(self, occ, prc):
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # 嵌入 B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D (转换到频域)
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)

        bias = x  # 保留原始频域表示作为偏置

        # 傅里叶图卷积
        x = self.fourierGC(x, B, N, L)

        x = x + bias  # 残差连接

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)

        # 逆FFT (转换回时域)
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # 投影
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)  # 最终预测
        x = torch.squeeze(x)
        return x

# 其他基线模型参考其原始代码实现