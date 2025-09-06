import pandas as pd
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def read_dataset():
    """
    读取数据集文件
    返回:
        occ: 充电桩占用率数据 (归一化后的)
        prc: 价格数据
        adj: 邻接矩阵
        col: 列名
        dis: 距离矩阵
        cap: 充电桩容量
        time: 时间信息
        inf: 基本信息
    """
    occ = pd.read_csv('occupancy.csv', index_col=0, header=0)  # 占用率数据
    inf = pd.read_csv('information.csv', index_col=None, header=0)  # 充电桩基本信息
    prc = pd.read_csv('price.csv', index_col=0, header=0)  # 价格数据
    adj = pd.read_csv('adj.csv', index_col=0, header=0)  # 邻接矩阵
    dis = pd.read_csv('distance.csv', index_col=0, header=0)  # 距离矩阵
    time = pd.read_csv('time.csv', index_col=None, header=0)  # 时间信息

    col = occ.columns  # 列名（充电桩ID）
    cap = np.array(inf['count'], dtype=float).reshape(1, -1)  # 充电桩容量
    occ = np.array(occ, dtype=float) / cap  # 归一化占用率（占用数/容量）
    prc = np.array(prc, dtype=float)  # 转换为numpy数组
    adj = np.array(adj, dtype=float)  # 转换为numpy数组
    dis = np.array(dis, dtype=float)  # 转换为numpy数组
    time = pd.to_datetime(time, dayfirst=True)  # 转换为日期时间格式
    return occ, prc, adj, col, dis, cap, time, inf


# ---------数据转换函数-----------
def create_rnn_data(dataset, lookback, predict_time):
    """
    创建RNN训练数据（滑动窗口）
    参数:
        dataset: 输入数据集
        lookback: 回看时间步长
        predict_time: 预测时间步长
    返回:
        x: 输入序列 [样本数, lookback, 特征数]
        y: 目标值 [样本数, 特征数]
    """
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])  # 输入序列
        y.append(dataset[i + lookback + predict_time - 1])  # 目标值
    return np.array(x), np.array(y)


def get_a_delta(adj):
    """
    计算归一化的邻接矩阵: D^(-1/2) * A * D^(-1/2)
    参数:
        adj: 原始邻接矩阵
    返回:
        a_delta: 归一化的邻接矩阵
    """
    # adj.shape = np.size(node, node)
    deg = np.sum(adj, axis=0)  # 计算度矩阵
    deg = np.diag(deg)  # 转换为对角矩阵
    deg_delta = np.linalg.inv(np.sqrt(deg))  # D^(-1/2)
    a_delta = np.matmul(np.matmul(deg_delta, adj), deg_delta)  # D^(-1/2) * A * D^(-1/2)
    return a_delta


def division(data, train_rate, valid_rate, test_rate):
    """
    划分数据集为训练集、验证集和测试集
    参数:
        data: 输入数据
        train_rate: 训练集比例
        valid_rate: 验证集比例
        test_rate: 测试集比例
    返回:
        train_data, valid_data, test_data: 划分后的数据集
    """
    data_length = len(data)
    train_division_index = int(data_length * train_rate)
    valid_division_index = int(data_length * (train_rate + valid_rate))
    test_division_index = int(data_length * (1 - test_rate))
    train_data = data[:train_division_index, :]
    valid_data = data[train_division_index:valid_division_index, :]
    test_data = data[test_division_index:, :]
    return train_data, valid_data, test_data


def set_seed(seed, flag):
    """
    设置随机种子以确保实验结果可复现
    参数:
        seed: 随机种子
        flag: 是否设置随机种子
    """
    if flag == True:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def metrics(test_pre, test_real):
    """
    计算多种评估指标
    参数:
        test_pre: 预测值
        test_real: 真实值
    返回:
        output_list: 包含各种评估指标的列表
    """
    eps = 0.01  # 小值，防止除以零
    MAPE_test_real = test_real.copy()
    MAPE_test_pre = test_pre.copy()
    # 处理零值，避免MAPE计算错误
    MAPE_test_real[np.where(MAPE_test_real == 0)] = MAPE_test_real[np.where(MAPE_test_real == 0)] + eps
    MAPE_test_pre[np.where(MAPE_test_real == 0)] = MAPE_test_pre[np.where(MAPE_test_real == 0)] + eps
    # 计算各种评估指标
    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))
    # 打印指标
    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))
    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list


class CreateDataset(Dataset):
    """
    创建标准数据集类，用于模型训练
    """
    def __init__(self, occ, prc, lb, pt, device, adj):
        """
        初始化数据集
        参数:
            occ: 占用率数据
            prc: 价格数据
            lb: 回看时间步长 (lookback)
            pt: 预测时间步长 (predict_time)
            device: 设备 (CPU/GPU)
            adj: 邻接矩阵
        """
        occ, label = create_rnn_data(occ, lb, pt)  # 创建占用率序列和标签
        prc, _ = create_rnn_data(prc, lb, pt)  # 创建价格序列
        self.occ = torch.Tensor(occ)  # 转换为张量
        self.prc = torch.Tensor(prc)  # 转换为张量
        self.label = torch.Tensor(label)  # 转换为张量
        self.device = device  # 设备

    def __len__(self):
        """返回数据集大小"""
        return len(self.occ)

    def __getitem__(self, idx):
        """
        获取单个样本
        参数:
            idx: 样本索引
        返回:
            output_occ: 占用率输入序列 [节点数, 序列长度]
            output_prc: 价格输入序列 [节点数, 序列长度]
            output_label: 标签 [节点数]
        """
        # occ: batch, seq, node -> 转置为 node, seq
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        return output_occ, output_prc, output_label


class CreateFastDataset(Dataset):
    """
    创建快速数据集类，包含伪样本生成，用于元学习或数据增强
    """
    def __init__(self, occ, prc, lb, pt, law, device, adj, num_layers=2, prob=0.6):
        """
        初始化数据集
        参数:
            occ: 占用率数据
            prc: 价格数据
            lb: 回看时间步长
            pt: 预测时间步长
            law: 价格-需求法则参数（负值表示价格与需求负相关）
            device: 设备
            adj: 邻接矩阵
            num_layers: 图传播层数
            prob: 价格变化概率
        """
        occ, label = create_rnn_data(occ, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        self.occ = torch.Tensor(occ)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device
        self.adj = adj
        self.eye = torch.eye(adj.shape[0])  # 单位矩阵
        self.deg = torch.sum(adj, dim=0)  # 度矩阵
        self.num_layers = num_layers
        self.law = -law  # 价格-需求法则参数（取负）

        # 价格变化
        chg = torch.randn(size=[self.occ.shape[2]]) / 2  # 生成随机价格变化
        chg[torch.where(chg < prob)] = 0  # 部分节点价格不变
        self.prc_chg = chg  # [节点数, ]

        # 标签变化（基于价格变化和图传播）
        chg = torch.unsqueeze(chg, dim=1)  # [节点数, 1]
        deg = torch.unsqueeze(self.deg, dim=1)  # [节点数, 1]
        label_chg = [-chg]  # 初始变化（价格与需求负相关）
        hop_chg = chg
        for n in range(self.num_layers):  # 图传播（多跳影响）
            hop_chg = torch.matmul(self.adj - self.eye, hop_chg) * (1 / deg)
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1)  # [节点数, 传播层数]
        label_chg = torch.sum(label_chg, dim=1)  # [节点数, ]
        self.label_chg = torch.squeeze(label_chg, dim=1)

    def __len__(self):
        """返回数据集大小"""
        return len(self.occ)

    def __getitem__(self, idx):
        """
        获取单个样本（包含原始样本和伪样本）
        参数:
            idx: 样本索引
        返回:
            output_occ: 原始占用率序列
            output_prc: 原始价格序列
            output_label: 原始标签
            output_prc_ch: 变化后的价格序列
            output_label_ch: 变化后的标签
        """
        # 伪采样：生成价格变化后的数据
        prc_ch = torch.Tensor(self.prc[idx, :, :] * (1 + self.prc_chg))  # 应用价格变化 [节点数, 序列长度]
        label_ch = torch.tan(torch.Tensor(self.label[idx, :] * (1 + self.label_chg / self.law)))  # 应用需求变化 [节点数, ]

        # 转移到设备
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        output_prc_ch = torch.transpose(prc_ch, 0, 1).to(self.device)
        output_label_ch = label_ch.to(self.device)
        return output_occ, output_prc, output_label, output_prc_ch, output_label_ch


class PseudoDataset(Dataset):
    """
    伪数据集类，用于生成增强数据
    """
    def __init__(self, occ, prc, lb, pt, device, adj, law, num_layers=2, prop=0.4):
        """
        初始化数据集
        参数:
            occ: 占用率数据
            prc: 价格数据
            lb: 回看时间步长
            pt: 预测时间步长
            device: 设备
            adj: 邻接矩阵
            law: 价格-需求法则参数
            num_layers: 图传播层数
            prop: 价格变化节点比例
        """
        occ, label = create_rnn_data(occ, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        self.occ = torch.Tensor(occ)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device
        self.adj = adj
        self.eye = torch.eye(adj.shape[0])
        self.deg = torch.sum(adj, dim=0)
        self.num_layers = num_layers
        self.prop = prop  # 价格变化节点比例
        self.law = -law  # 价格-需求法则参数（取负）

        # 价格变化
        node_score = torch.rand(size=[self.occ.shape[2]])  # 为每个节点生成随机分数
        shred = torch.quantile(node_score, self.prop)  # 计算分位数
        prc_chg = torch.randn_like(node_score) / 2  # 价格变化百分比
        prc_chg[torch.where(node_score > self.prop)] = 0  # 部分节点价格不变
        self.prc_chg = prc_chg

        # 标签变化（基于价格变化和图传播）
        label_chg = self.law * prc_chg  # 占用率变化百分比（价格与需求负相关）
        label_chg = torch.unsqueeze(label_chg, dim=1)  # [节点数, 1]
        hop_chg = -label_chg
        label_chg = [label_chg]
        deg = torch.unsqueeze(self.deg, dim=1)  # [节点数, 1]
        for n in range(self.num_layers):  # 图传播
            hop_chg = torch.matmul(self.adj - self.eye, hop_chg) * (1 / deg)
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1)  # [节点数, 传播层数]
        label_chg = torch.sum(label_chg, dim=1)  # [节点数, ]
        self.label_chg = torch.squeeze(label_chg, dim=1)

    def __len__(self):
        """返回数据集大小"""
        return len(self.occ)

    def __getitem__(self, idx):
        """
        获取单个样本（包含原始样本和伪样本）
        参数:
            idx: 样本索引
        返回:
            output_occ: 原始占用率序列
            output_prc: 原始价格序列
            output_label: 原始标签
            output_pseudo_prc: 伪价格序列
            output_pseudo_label: 伪标签
        """
        # 采样：生成伪数据
        pseudo_prc = torch.Tensor(self.prc[idx, :, :] * (1 + self.prc_chg))  # [节点数, 序列长度]
        pseudo_label = torch.tan(torch.Tensor(self.label[idx, :] * (1 + self.label_chg)))  # [节点数, ]

        # 转移到设备
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        output_pseudo_prc = torch.transpose(pseudo_prc, 0, 1).to(self.device)
        output_pseudo_label = pseudo_label.to(self.device)

        return output_occ, output_prc, output_label, output_pseudo_prc, output_pseudo_label


def meta_division(data, support_rate, query_rate):
    """
    元学习数据划分：支持集和查询集
    参数:
        data: 输入数据
        support_rate: 支持集比例
        query_rate: 查询集比例
    返回:
        supprot_set: 支持集
        query_set: 查询集
    """
    data_length = len(data)
    support_division_index = int(data_length * support_rate)
    supprot_set = data[:support_division_index, :]
    query_set = data[support_division_index:, :]
    return supprot_set, query_set


def zero_init_global_gradient(model):
    """
    初始化全局梯度为零
    参数:
        model: 模型
    返回:
        grads: 初始化为零的梯度字典
    """
    grads = dict()
    for name, param in model.named_parameters():
        param.requires_grad_(True)  # 确保参数需要梯度
        grads[name] = 0  # 初始化梯度为零
    return grads


def data_mix(ori_data, pse_data, mix_ratio):
    """
    混合原始数据和伪数据
    参数:
        ori_data: 原始数据
        pse_data: 伪数据
        mix_ratio: 混合比例
    返回:
        mix_data: 混合后的数据
    """
    shred = int(ori_data.shape[0] * mix_ratio)  # 计算混合点
    mix_data = ori_data  # 复制原始数据
    mix_data[shred:] = pse_data[shred:]  # 在第一个维度（批次）上混合数据
    return mix_data