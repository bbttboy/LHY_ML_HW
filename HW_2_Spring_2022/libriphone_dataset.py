import torch
from torch.utils.data.dataset import Dataset, T_co
from os.path import join
import random


class PhoneDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = "train",
                 seed: int = 131416, concat_n: int = 0):
        super.__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.seed = seed
        self.concat_n = concat_n
        self.X = None
        self.y = None

    def _load_data(self):
        mode = 'train' if self.split in ['train', 'valid'] else 'test'
        # 1. 标签
        labels_dict = {}
        if mode == 'train':
            labels_path = join(self.dataset_path, 'train_labels.txt')
            with open(labels_path) as f:
                lines = f.readlines()
                for line in lines:
                    label = line.strip('\n').split(' ')
                    labels_dict[label[0]] = label[1:]

        # 2. 数据明细
        split_list = []
        split_path = join(self.dataset_path, f'{mode}_split.txt')
        with open(split_path) as f:
            lines = f.readlines()
            for line in lines:
                split_list.append(line.strip('\n'))

        # 3. 数据源
        data_dir = join(self.dataset_path, 'feat', mode)
        # 打乱数据
        random.seed(self.seed)
        random.shuffle(split_list)
        if mode == 'train':
            split_list = self._get_valid_split(split_list)
        # 获取数据
        data = []
        labels = []
        for feat_file in split_list:
            feat_path = join(data_dir, f'{feat_file}.pt')
            feat = torch.load(feat_path)
            # 窗口化数据
            concat_feat = self._process_feat(feat)
            data.append(concat_feat)
            labels.append(torch.tensor(labels_dict[feat_file]))
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        self.X = data
        self.y = labels

    def _get_valid_split(self, split_list):
        train_ratio = 0.8
        if self.split == 'train':
            return split_list[: len(split_list) * train_ratio]
        else:
            return split_list[len(split_list) * train_ratio:]

    def _process_feat(self, feat):
        dim0 = feat.shape[0]
        dim1 = feat.shape[1]
        concat_feat = torch.empty(dim0, 2 * self.concat_n + 1, dim1)
        concat_feat[:, self.concat_n, :] = feat
        mid = self.concat_n
        for i in range(self.concat_n):
            concat_feat[:, mid+i+1, :] = _shift_feat(feat, i+1)
            concat_feat[:, mid-i-1, :] = _shift_feat(feat, -i-1)
        return concat_feat

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]


def _shift_feat(feat: torch.Tensor, n):
    if n == 0:
        return feat
    # 左移n位
    elif n > 0:
        shift_feat = torch.empty(feat.shape)
        shift_feat[:-n] = feat[n:]
        shift_feat[-n:] = feat[-1].repeat([n] + [1] * (feat.dim() - 1))
    # 右移-n位
    else:
        n = -n
        shift_feat = torch.empty(feat.shape)
        shift_feat[n:] = feat[:-n]
        shift_feat[:n] = feat[0].repeat([n] + [1] * (feat.dim() - 1))
    return shift_feat
