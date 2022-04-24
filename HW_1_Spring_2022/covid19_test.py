from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np


from HW_1_Spring_2022.covid19_dataset import Covid19Dataset
from covid19_regression_model import Covid19RegressionModel as CModel


def predict(test_loader, model):
    model.eval()
    criterion = nn.MSELoss("mean")
    preds = []
    loss_record = []
    test_pbar = tqdm(test_loader, position=0, leave=True)
    origin = []
    for x, y in test_pbar:
        x, y = x.float().cuda(), y.float().cuda()
        origin.append(y.detach().cpu())
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
            loss = criterion(pred, y)
            loss_record.append(loss.detach().item())
    mean_loss = sum(loss_record)/len(loss_record)
    print("Loss/test:", f"{mean_loss: .4f}")
    preds = torch.cat(preds, dim=0).numpy()
    origin = torch.cat(origin, dim=0).numpy()

    plt.title("Test")
    plt.xlabel("x")
    plt.ylabel("y")
    x = np.arange(1, len(preds) + 1)
    plt.plot(x, origin)
    plt.plot(x, preds)
    plt.legend(["origin", "predict"])
    plt.show()

    return preds


def test():
    resource = r"D:\Machine Learning\05 李宏毅2021_2022机器学习\Lhy HW Data\2022\HW1"
    test_dataset = Covid19Dataset(
        dataset_path=resource,
        split="valid",
        valid=0.2,
        seed=131416,
        feat_idx=None
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=256,
        pin_memory=True
    )
    model = CModel(test_dataset.x.shape[1])
    model.load_state_dict(
        torch.load(r"./log/2022-04-23_23_01_59_Covid19/best_loss_checkpoint.ckpt")['model_state_dict'])
    predict(test_loader, model.cuda())


if __name__ == '__main__':
    test()

