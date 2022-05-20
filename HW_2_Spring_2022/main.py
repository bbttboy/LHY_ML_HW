def parse_opt():
    """
    命令行参数设置
    :return: Namespace
    """
    import argparse

    parser = argparse.ArgumentParser()
    resource = r"D:\Machine Learning\05 李宏毅2021_2022机器学习\Lhy HW Data\2022\HW2\libriphone\libriphone"
    parser.add_argument("--model", type=str, default="PhonemeModel")
    parser.add_argument("--dataset_path", type=str, default=resource)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--valid", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=131416)
    parser.add_argument("--feat_idx", type=list, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=100000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./log")

    return parser.parse_args()


def same_seed(seed):
    """
    设置所有的随机函数的随机种子

    设置随机种子其实就是让某一随机函数变成一个固定的迭代器

    :param seed: 随机种子
    """
    import torch.backends.cudnn
    import numpy as np
    import torch

    # 1. cudnn确定性
    torch.backends.cudnn.deterministic = True
    # 2. 自动优化
    torch.backends.cudnn.benchmark = False
    # 3. np随机种子
    np.random.seed(seed)
    # 4. cpu随机种子
    torch.manual_seed(seed)
    # 5. cuda随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_optimizer_and_criterion(opt, input_dim):
    from libriphone_model import PhonemeModel
    import torch

    model = PhonemeModel(input_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion


def get_data_loader(dataset, split, opt):
    from torch.utils.data.dataloader import DataLoader
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True if split == 'train' else False,
    )


def create_logger(opt):
    from torch.utils.tensorboard import SummaryWriter
    from os.path import join
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = join(opt.log_dir, now, "_", opt.model)
    logger = SummaryWriter(log_dir=log_dir)
    for k, v in opt.__dict__.items():
        logger.add_text(k, str(v))
    return logger


def train_loop(opt, train_set, valid_set, model, optimizer, criterion, logger):
    import math
    import os
    import torch
    from tqdm import tqdm

    logger.add_text("train_set_x_shape", str(train_set.X.shape))
    logger.add_text("valid_set_x_shape", str(valid_set.X.shape))

    train_loader = get_data_loader(train_set, "train", opt)
    valid_loader = get_data_loader(valid_set, "valid", opt)

    best_loss, step, early_stop_count = math.inf, 0, 0

    for epoch in range(opt.num_epochs):
        # 训练
        train_pbar = tqdm(train_loader, position=0, leave=True)
        loss_record = []
        model.train()
        for x, y in train_pbar:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1}/{opt.num_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        logger.add_scalar('Loss/train', mean_train_loss, step)

        # 验证
        valid_pbar = tqdm(valid_loader, position=0, leave=True)
        loss_record = []
        model.eval()
        for x, y in valid_pbar:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{opt.num_epochs}]: '
              f'Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        logger.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(
                {"epoch": epoch,
                 "step": step,
                 "model_state_dict": model.state_dict()},
                os.path.join(logger.file_writer.get_logdir(), 'best_loss_checkpoint.ckpt')
            )
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # 保存最后的模型
        torch.save(
            {"epoch": epoch,
             "step": step,
             "model_state_dict": model.state_dict()},
            os.path.join(logger.file_writer.get_logdir(), 'lasted_checkpoint.ckpt')
        )


def main():
    from libriphone_dataset import PhoneDataset
    opt = parse_opt()
    same_seed(opt.seed)
    model, optimizer, criterion = create_model_optimizer_and_criterion(opt, 195)
    logger = create_logger(opt)
    train_set = PhoneDataset(opt.dataset_path, split='train', concat_n=2)
    valid_set = PhoneDataset(opt.dataset_path, split='valid', concat_n=2)
    train_loop(opt, train_set, valid_set, model, optimizer, criterion, logger)
    print('Finished!')


if __name__ == '__main__':
    main()
