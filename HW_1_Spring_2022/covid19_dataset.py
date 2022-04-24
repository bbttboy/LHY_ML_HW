from torch.utils.data.dataset import Dataset, T_co, random_split
from torch.utils.data.dataloader import DataLoader
from os.path import join, isfile, isdir
from os import listdir
import pandas as pd
import numpy as np
import torch


class Covid19Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = 'train',
                 valid: float = 0.2, seed: int = 131416, feat_idx: list = None):
        r"""
        李宏毅2022春季hw_1数据集

        默认地址: D:\Machine Learning\05 李宏毅2021_2022机器学习\Lhy HW Data\2022\HW1

        :param dataset_path: 数据集路径
        :param split: 加载测试还是训练集
        :param valid: 训练集中划分多少作为验证集, 仅对train和valid起作用
        :param seed: 随机种子, 仅对train和valid起作用
        :param feat_idx: 选择一条样本中哪些位置作为特征输入, 不指定时视为除开标签的全部
        """
        super(Covid19Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.valid = valid
        self.seed = seed
        self.feat_idx = feat_idx

        # 加载数据集
        self.dataset = self._load_data()
        self.x, self.y = self._generate_sample()

    def _load_data(self):
        r"""
        加载对应数据集

        """
        split_tag = self.split
        if self.split is 'valid':
            split_tag = 'train'
        assert isdir(self.dataset_path), "请检查数据集路径"
        files = [f for f in listdir(self.dataset_path) if isfile(join(self.dataset_path, f))]
        data_file = [f for f in files if split_tag in f][0]
        data = pd.read_csv(join(self.dataset_path, data_file)).values
        data_size = len(data)
        valid_size = int(self.valid * data_size)
        if self.split is 'test':
            dataset = data
        else:
            if self.split is 'valid':
                _, dataset = \
                    random_split(data, [data_size - valid_size, valid_size],
                                 generator=torch.Generator().manual_seed(self.seed))
            else:
                dataset, _ = \
                    random_split(data, [data_size - valid_size, valid_size],
                                 generator=torch.Generator().manual_seed(self.seed))
        return np.array(dataset)

    def get_loader(self, batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def _generate_sample(self):
        y = self.dataset[:, -1]
        if not self.feat_idx:
            self.feat_idx = list(range(0, self.dataset.shape[-1] - 1))
        x = self.dataset[:, self.feat_idx]
        return x, y

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.dataset)
