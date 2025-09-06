import torch
import torch.nn as nn
import torch.nn.functional as F
import functions as fn
import copy

# 设置设备（GPU或CPU）
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)  # 设置随机种子以确保结果可复现


class MultiHeadsGATLayer(nn.Module):
    """多头图注意力网络层
    实现多头注意力机制，用于捕捉图中节点间复杂的关系
    """
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n  # 注意力头数量
        self.heads_dict = dict()  # 存储每个头的参数
        # 初始化每个注意力头的参数
        for n in range(head_n):
            # 线性变换权重矩阵
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=device))
            # 注意力计算权重向量
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=device))
            # Xavier初始化
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        # 多头注意力融合线性层
        self.linear = nn.Linear(head_n, 1, device=device)

        # 正则化组件
        self.leakyrelu = nn.LeakyReLU(alpha)  # LeakyReLU激活函数
        self.dropout = nn.Dropout(dropout)     # Dropout层
        self.softmax = nn.Softmax(dim=0)       # Softmax层

        # 稀疏矩阵处理
        self.a_sparse = a_sparse  # 稀疏邻接矩阵
        self.edges = a_sparse.indices()  # 边的索引
        self.values = a_sparse.values()  # 边的值
        self.N = a_sparse.shape[0]      # 节点数量
        a_dense = a_sparse.to_dense()   # 转换为密集矩阵
        # 创建掩码：将不存在的边设为极小值，存在的边设为0
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense  # 掩码矩阵

    def forward(self, x):
        """前向传播
        参数:
            x: 输入特征 [batch_size, num_nodes, seq_length]
        返回:
            atts_mat: 注意力权重矩阵 [num_nodes, num_nodes]
        """
        b, n, s = x.shape
        x = x.reshape(b*n, s)  # 重塑为 [batch_size * num_nodes, seq_length]

        atts_stack = []  # 存储每个头的注意力权重
        # 多头注意力计算
        for n in range(self.head_n):
            # 线性变换
            h = torch.matmul(x, self.heads_dict[n, 0])
            # 拼接相邻节点的特征
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            # 计算注意力分数
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)  # LeakyReLU激活
            atts_stack.append(atts)

        # 融合多头注意力
        mt_atts = torch.stack(atts_stack, dim=1)  # 堆叠所有头的注意力
        mt_atts = self.linear(mt_atts)  # 线性融合
        new_values = self.values * mt_atts.squeeze()  # 应用注意力权重到边值
        # 构建稀疏注意力矩阵
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask  # 转换为密集矩阵并添加掩码
        atts_mat = self.softmax(atts_mat)  # Softmax归一化
        return atts_mat


class MLP(nn.Module):
    """多层感知机
    简单的全连接神经网络
    """
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        # 三层全连接网络
        self.l1 = nn.Linear(in_features=in_channel, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=out_channel)
        # self.dropout = nn.Dropout(p=0.5)  # 可选的Dropout层
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        """前向传播"""
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class PAG(nn.Module):
    """物理感知图网络 (Physics-Aware Graph Network)
    结合图注意力网络(GAT)和时间模式注意力(TPA)的混合模型
    """
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAG, self).__init__()
        self.feature = seq  # 特征维度
        self.seq = seq - kcnn + 1  # 卷积后的序列长度
        self.alpha = 0.5  # 残差连接权重
        self.m = m  # LSTM隐藏状态维度
        self.a_sparse = a_sparse  # 稀疏邻接矩阵
        self.nodes = a_sparse.shape[0]  # 节点数量

        # GAT组件
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # 2D卷积，输入形状=[batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)  # 多头图注意力层
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)  # 图卷积线性层

        # TPA(时间模式注意力)组件
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)  # LSTM层
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)  # 全连接层1
        self.fc2 = nn.Linear(in_features=k, out_features=m)  # 全连接层2
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)  # 全连接层3
        self.decoder = nn.Linear(self.seq, 1)  # 解码器

        # 激活函数和正则化
        self.dropout = nn.Dropout(p=0.5)  # Dropout层
        self.LeakyReLU = nn.LeakyReLU()  # LeakyReLU激活函数

        # 预处理邻接矩阵
        adj1 = copy.deepcopy(self.a_sparse.to_dense())  # 转换为密集矩阵
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001  # 对角线设为极小值（避免除零）
            adj2[i, i] = 0  # 对角线设为0
        degree = 1.0 / (torch.sum(adj1, dim=0))  # 计算度矩阵的倒数
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)  # 创建度矩阵
        for i in range(12):
            degree_matrix[:, i] = degree  # 复制度值到所有特征维度
        self.degree_matrix = degree_matrix  # 度矩阵
        self.adj2 = adj2  # 处理后的邻接矩阵

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        """前向传播
        参数:
            occ: 占用率数据 [batch_size, num_nodes, seq_length]
            prc: 价格数据 [batch_size, num_nodes, seq_length]
        返回:
            y: 预测结果 [batch_size, num_nodes]
        """
        b, n, s = occ.shape
        # 拼接占用率和价格数据
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = self.conv2d(data)  # 2D卷积
        data = data.squeeze().reshape(b, n, -1)  # 重塑形状

        # 第一层图注意力
        atts_mat = self.gat_lyr(data)  # 计算注意力矩阵 [nodes, nodes]
        occ_conv1 = torch.matmul(atts_mat, data)  # 应用注意力 [batch, nodes, seq]
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))  # 线性变换+激活+Dropout

        # 第二层图注意力
        atts_mat2 = self.gat_lyr(occ_conv1)  # 计算注意力矩阵
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # 应用注意力
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))  # 线性变换+激活+Dropout

        # 残差连接
        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)  # 重塑形状
        occ_conv2 = occ_conv2.view(b * n, self.seq)  # 重塑形状

        # 堆叠两层输出
        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # 最佳组合方式
        # LSTM处理时间序列
        lstm_out, (_, _) = self.lstm(x)  # [batch*nodes, seq, 2]

        # TPA(时间模式注意力)机制
        ht = lstm_out[:, -1, :]  # 最后一个时间步的隐藏状态
        hw = lstm_out[:, :-1, :]  # 从h(t-1)到h1的所有隐藏状态
        hw = torch.transpose(hw, 1, 2)  # 转置
        Hc = self.fc1(hw)  # 全连接变换
        Hn = self.fc2(Hc)  # 全连接变换
        ht = torch.unsqueeze(ht, dim=2)  # 增加维度
        a = torch.bmm(Hn, ht)  # 计算注意力权重
        a = torch.sigmoid(a)  # Sigmoid激活
        a = torch.transpose(a, 1, 2)  # 转置
        vt = torch.matmul(a, Hc)  # 加权和
        ht = torch.transpose(ht, 1, 2)  # 转置
        hx = torch.cat((vt, ht), dim=2)  # 拼接
        y = self.fc3(hx)  # 最终预测
        y = y.view(b, n)  # 重塑为 [batch_size, num_nodes]
        return y