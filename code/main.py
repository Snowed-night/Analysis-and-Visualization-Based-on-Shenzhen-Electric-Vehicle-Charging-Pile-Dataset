import copy
import baselines
import torch
import numpy as np
import pandas as pd
import functions as fn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条库
import models
import learner

# 系统配置
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")  # 设置设备（GPU优先）
fn.set_seed(seed=2023, flag=True)  # 设置随机种子以确保结果可复现

# 超参数配置
model_name = 'PAG'  # 模型名称
seq_l = 12  # 输入序列长度
pre_l = 6   # 预测序列长度
bs = 512    # 批次大小
p_epoch = 200  # 预训练轮数
n_epoch = 1000  # 微调训练轮数
# 价格需求弹性系数列表（负值表示价格与需求负相关）
law_list = np.array([-1.48, -0.74])  # 电动汽车充电需求的价格弹性。建议：最多5个元素。
is_train = True  # 是否进行训练
mode = 'completed'  # 预训练模式：'simplified'（简化）或 'completed'（完整）
is_pre_train = True  # 是否进行预训练

# 读取输入数据
occ, prc, adj, col, dis, cap, time, inf = fn.read_dataset()
# 处理邻接矩阵
adj_dense = torch.Tensor(adj)  # 转换为张量
adj_dense_cuda = adj_dense.to(device)  # 转移到设备
adj_sparse = adj_dense.to_sparse().to(device)  # 转换为稀疏矩阵并转移到设备

# 数据集划分
train_occupancy, valid_occupancy, test_occupancy = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_price, valid_price, test_price = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

# 创建数据加载器
train_dataset = fn.CreateDataset(train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)  # 训练数据加载器
valid_dataset = fn.CreateDataset(valid_occupancy, valid_price, seq_l, pre_l, device, adj_dense)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occupancy), shuffle=False)  # 验证数据加载器
test_dataset = fn.CreateDataset(test_occupancy, test_price, seq_l, pre_l, device, adj_dense)
test_loader = DataLoader(test_dataset, batch_size=len(test_occupancy), shuffle=False)  # 测试数据加载器

# 训练设置
model = models.PAG(a_sparse=adj_sparse).to(device)  # 初始化模型（物理感知图网络）
# 其他可选模型：
# model = FGN().to(device)  # 傅里叶图网络
# model = baselines.LSTM(seq_l, 2).to(device)  # LSTM基准模型
# model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda).to(device)  # LSTM+GCN基准模型

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)  # Adam优化器，带L2正则化
loss_function = torch.nn.MSELoss()  # 均方误差损失函数
valid_loss = 100  # 初始化验证损失

if is_train is True:
    model.train()  # 设置模型为训练模式
    if is_pre_train is True:  # 预训练阶段
        if mode == 'simplified':  # 简化版的物理信息元学习
            model = learner.fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)

        elif mode == 'completed':  # 完整的物理信息元学习过程
            model = learner.physics_informed_meta_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
        else:
            print("Mode error, skip the pre-training process.")  # 模式错误，跳过预训练

    # 微调训练阶段
    for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
        for j, data in enumerate(train_loader):
            '''
            数据格式:
            occupancy = (batch, seq, node)  # 占用率序列
            price = (batch, seq, node)      # 价格序列
            label = (batch, node)           # 标签
            '''
            model.train()  # 设置模型为训练模式
            occupancy, price, label = data

            optimizer.zero_grad()  # 清零梯度
            predict = model(occupancy, price)  # 前向传播
            loss = loss_function(predict, label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        for j, data in enumerate(valid_loader):
            '''
            数据格式:
            occupancy = (batch, seq, node)
            price = (batch, seq, node)
            label = (batch, node)
            '''
            model.train()  # 注意：这里应该是model.eval()，可能是代码错误
            occupancy, price, label = data
            predict = model(occupancy, price)
            loss = loss_function(predict, label)
            if loss.item() < valid_loss:  # 保存最佳模型
                valid_loss = loss.item()
                torch.save(model, './checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')

# 加载最佳模型
model = torch.load('./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')

# 测试阶段
model.eval()  # 设置模型为评估模式
result_list = []  # 存储结果列表
predict_list = np.zeros([1, adj_dense.shape[1]])  # 初始化预测结果数组
label_list = np.zeros([1, adj_dense.shape[1]])    # 初始化真实标签数组

for j, data in enumerate(test_loader):
    occupancy, price, label = data  # occupancy.shape = [batch, seq, node]
    print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
    with torch.no_grad():  # 禁用梯度计算
        predict = model(occupancy, price)  # 预测
        predict = predict.cpu().detach().numpy()  # 转移到CPU并转换为numpy数组
        label = label.cpu().detach().numpy()      # 转移到CPU并转换为numpy数组
        predict_list = np.concatenate((predict_list, predict), axis=0)  # 拼接预测结果
        label_list = np.concatenate((label_list, label), axis=0)        # 拼接真实标签

# 计算评估指标
output_no_noise = fn.metrics(test_pre=predict_list[1:, :], test_real=label_list[1:, :])
result_list.append(output_no_noise)

# 保存结果到CSV文件
result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'], data=result_list)
result_df.to_csv(model_name + '_' + str(pre_l) + 'bs' + str(bs) + '.csv', encoding='gbk')