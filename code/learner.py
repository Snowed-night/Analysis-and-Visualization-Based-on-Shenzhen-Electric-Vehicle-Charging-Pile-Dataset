import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import functions as fn
import copy
from tqdm import tqdm  # 进度条库


def physics_informed_meta_learning(law_list, global_model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l,
                                   pre_l, device, adj_dense):
    """
    基于物理信息的元学习算法
    参数:
        law_list: 物理定律参数列表（价格-需求关系参数）
        global_model: 全局模型
        model_name: 模型名称
        p_epoch: 预训练轮数
        bs: 批次大小
        train_occupancy: 训练集占用率数据
        train_price: 训练集价格数据
        seq_l: 序列长度
        pre_l: 预测长度
        device: 设备 (CPU/GPU)
        adj_dense: 密集邻接矩阵
    返回:
        global_model: 预训练后的全局模型
    """
    # 划分支持集和查询集（元学习标准设置）
    support_occ, query_occ = fn.meta_division(train_occupancy, support_rate=0.5, query_rate=0.5)
    support_prc, query_prc = fn.meta_division(train_price, support_rate=0.5, query_rate=0.5)

    # 预训练数据生成
    n_laws = len(law_list)  # 物理定律数量
    support_dataset_dict = dict()  # 支持集数据集字典
    query_dataset_dict = dict()  # 查询集数据集字典
    support_dataloader_dict = dict()  # 支持集数据加载器字典
    query_dataloader_dict = dict()  # 查询集数据加载器字典

    # 为每个物理定律创建数据集和数据加载器
    for n in range(n_laws):
        support_dataset_dict[n] = fn.PseudoDataset(support_occ, support_prc, seq_l, pre_l, device, adj_dense,
                                                   law_list[n])
        query_dataset_dict[n] = fn.PseudoDataset(query_occ, query_prc, seq_l, pre_l, device, adj_dense, law_list[n])
        support_dataloader_dict[n] = DataLoader(support_dataset_dict[n], batch_size=bs, shuffle=True, drop_last=True)
        query_dataloader_dict[n] = DataLoader(query_dataset_dict[n], batch_size=query_occ.shape[0], shuffle=False)

    # 保存初始模型
    torch.save(global_model, './checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')
    loss_function = torch.nn.MSELoss()  # 均方误差损失函数

    # 外层循环（元学习循环）
    global_model.train()
    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        query_loss = 100  # 初始化查询损失
        global_grads = fn.zero_init_global_gradient(global_model)  # 初始化全局梯度

        # 内层循环（遍历所有物理定律）
        for n in range(n_laws):
            # 加载当前模型副本
            temp_model = torch.load(
                './checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt').to(device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), weight_decay=0.00001)  # 临时优化器
            temp_model.train()

            # 支持集训练（适应阶段）
            for j, data in enumerate(support_dataloader_dict[n]):
                '''
                数据格式:
                occupancy = (batch, seq, node)  # 占用率序列
                price = (batch, seq, node)      # 价格序列
                label = (batch, node)           # 标签
                pseudo_price = (batch, seq, node)  # 伪价格序列
                pseudo_label = (batch, node)       # 伪标签
                '''
                occupancy, price, label, pseudo_price, pseudo_label = data
                mix_ratio = (j + 1) * occupancy.shape[0] / len(train_occupancy)  # 计算混合比例
                mix_prc = fn.data_mix(price, pseudo_price, mix_ratio)     # 混合价格数据
                mix_label = fn.data_mix(label, pseudo_label, mix_ratio)  # 混合标签数据

                temp_optimizer.zero_grad()  # 清零梯度
                predict = temp_model(occupancy, mix_prc)  # 前向传播
                loss = loss_function(predict, mix_label)  # 计算损失
                loss.backward()  # 反向传播
                temp_optimizer.step()  # 更新参数

            # 查询集评估（元优化阶段）
            for j, data in enumerate(query_dataloader_dict[n]):
                '''
                数据格式:
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, pseudo_price, pseudo_label = data
                temp_optimizer.zero_grad()
                predict = temp_model(occupancy, price)  # 使用原始价格预测
                loss = loss_function(predict, label)  # 计算损失
                loss.backward()  # 反向传播计算梯度

                # 累积梯度到全局梯度
                for name, param in temp_model.named_parameters():
                    if param.grad is not None:
                        global_grads[name] += param.grad

        # 全局更新：批量梯度下降 (BGD)
        for name, param in global_model.named_parameters():
            param = param - 0.02 * global_grads[name] / n_laws  # 平均梯度并更新参数

        # 保存最佳模型
        if query_loss > loss:
            loss = query_loss
            torch.save(global_model,
                       './checkpoints' + '/meta_' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')

    return global_model


def fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device,
                  adj_dense):
    """
    快速学习算法（数据增强预训练）
    参数:
        law_list: 物理定律参数列表
        model: 模型
        model_name: 模型名称
        p_epoch: 预训练轮数
        bs: 批次大小
        train_occupancy: 训练集占用率数据
        train_price: 训练集价格数据
        seq_l: 序列长度
        pre_l: 预测长度
        device: 设备
        adj_dense: 密集邻接矩阵
    返回:
        model: 预训练后的模型
    """
    n_laws = len(law_list)
    fast_datasets = dict()  # 快速学习数据集字典
    fast_loaders = dict()  # 快速学习数据加载器字典

    # 为每个物理定律创建数据集和数据加载器
    for n in range(n_laws):
        fast_datasets[n] = fn.CreateFastDataset(train_occupancy, train_price, seq_l, pre_l, law_list[n], device,
                                                adj_dense)
        fast_loaders[n] = DataLoader(fast_datasets[n], batch_size=bs, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)  # 优化器
    loss_function = torch.nn.MSELoss()  # 损失函数

    # 预训练循环
    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        # 遍历所有物理定律
        for n in range(n_laws):
            # 第一阶段训练：使用变化后的数据
            for j, data in enumerate(fast_loaders[n]):
                '''
                数据格式:
                occupancy = (batch, seq, node)  # 占用率序列
                price = (batch, seq, node)      # 价格序列
                label = (batch, node)           # 标签
                prc_ch = (batch, seq, node)     # 变化后的价格序列
                label_ch = (batch, node)        # 变化后的标签
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()  # 清零梯度
                predict = model(occupancy, prc_ch)  # 使用变化后的价格预测
                loss = loss_function(predict, label_ch)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

            # 第二阶段训练：再次使用变化后的数据（增强训练）
            for j, data in enumerate(fast_loaders[n]):
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

    return model