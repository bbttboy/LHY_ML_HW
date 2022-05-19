from tqdm import tqdm
import os
import argparse
import math
import torch
import torch.backends.cudnn
import numpy as np
from covid19_dataset import Covid19Dataset
from covid19_logger import Covid19Logger
from covid19_regression_model import Covid19RegressionModel as CModel


def parse_opt():
    parser = argparse.ArgumentParser()
    resource = r"D:\Machine Learning\05 李宏毅2021_2022机器学习\Lhy HW Data\2022\HW1"
    parser.add_argument("--model", type=str, default="Covid19")
    parser.add_argument("--dataset_path", type=str, default=resource)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--valid", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=131416)
    parser.add_argument("--feat_idx", type=list, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=100000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument("--early_stop", type=int, default=400)
    parser.add_argument('--log_dir', type=str, default="./log")

    return parser.parse_args()


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    # cudnn确定性
    torch.backends.cudnn.deterministic = True
    # 自动优化
    torch.backends.cudnn.benchmark = False
    # np随机
    np.random.seed(seed)
    # cpu随机
    torch.manual_seed(seed)
    # gpu随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_optimizer_and_criterion(opt, dim):
    model = CModel(dim).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    criterion = torch.nn.MSELoss(reduction="mean")
    return model, optimizer, criterion


def get_data_loader(opt, dataset):
    shuffle = True
    if opt.split == "test":
        shuffle = False
    return dataset.get_loader(opt.batch_size, shuffle)


def train_loop(opt, train_set, valid_set, model, optimizer, criterion, logger):
    logger.add_text("train_set_x_shape", str(train_set.x.shape))
    logger.add_text("valid_set_x_shape", str(valid_set.x.shape))

    train_loader = train_set.get_loader(batch_size=opt.batch_size, shuffle=True)
    valid_loader = valid_set.get_loader(batch_size=opt.batch_size, shuffle=True)

    best_loss, step, early_stop_count = math.inf, 0, 0

    for epoch in range(opt.num_epochs):

        # 训练
        train_pbar = tqdm(train_loader, position=0, leave=True)
        loss_record = []
        model.train()
        for x, y in train_pbar:
            # TODO 为什么每个batch都要执行zero_grad
            # 因为不执行zero_grad，梯度下降是使用累计梯度
            # Set gradient to zero.
            optimizer.zero_grad()
            x, y = x.float().cuda(), y.float().cuda()
            pred = model(x)
            loss = criterion(pred, y)
            # Compute gradient(backpropagation).
            loss.backward()
            # Update parameters.
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{opt.num_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        logger.add_scalar("Loss/train", mean_train_loss, step)

        # 验证
        valid_pbar = tqdm(valid_loader, position=0, leave=True)
        loss_record = []
        model.eval()
        for x, y in valid_pbar:
            x, y = x.float().cuda(), y.float().cuda()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{opt.num_epochs}]: '
              f'Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        logger.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            # 保存loss最好时的模型
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

        if early_stop_count > opt.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            return


def main():
    opt = parse_opt()
    same_seed(opt.seed)
    train_set = Covid19Dataset(
        dataset_path=opt.dataset_path,
        split="train",
        valid=opt.valid,
        seed=opt.seed,
        feat_idx=opt.feat_idx
    )
    valid_set = Covid19Dataset(
        dataset_path=opt.dataset_path,
        split="valid",
        valid=opt.valid,
        seed=opt.seed,
        feat_idx=opt.feat_idx
    )
    model, optimizer, criterion = create_model_optimizer_and_criterion(opt, train_set.x.shape[1])
    logger = Covid19Logger(opt).get_logger()
    train_loop(opt, train_set, valid_set, model, optimizer, criterion, logger)
    print("Finished!")


if __name__ == '__main__':
    main()

torch.optim.ASGD





