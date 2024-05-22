import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ImageDataset
from model.loss import ImageLoss, CSVLoss
from model.module import ImageModule, CSVModule
from model_config import ImageConfig, CSVConfig
from processing import data_factory

from sklearn.utils import resample

from torcheval.metrics import BinaryAUROC
from torcheval.metrics.functional import r2_score

def cal_auroc(y_preds, y_true, batch_size):
    metric = BinaryAUROC(num_tasks=batch_size)
    metric.update(y_preds, y_true)
    vals = metric.compute()
    return vals
    
def cal_reg(y_preds, y_true):
    val = r2_score(y_preds, y_true)
    return val


class ImageModel:
    def __init__(self, config: ImageConfig):
        self.config = config

    def train(self, train_data):
        self.config.training = True
        device = torch.device(self.config.device)

        data_loader = DataLoader(ImageDataset(self.config, train_data), batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        model = ImageModule()
        if self.config.train_from_scratch == False:
            if self.config.model_load_path and os.path.exists(self.config.model_load_path):
                print('load model')
                model = torch.load(self.config.model_load_path)

        model = model.to(device)
        model.train()

        lossFn = ImageLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_epochs = []
        auroc_epochs = []
        acc_epochs = []
        for epoch in range(self.config.epochs):
            loss_batchs = []
            auroc_batchs = []
            acc_batchs = []
            for images, labels in tqdm(data_loader):
                optim.zero_grad()
                images = images.to(device)
                labels[0] = labels[0].to(device)
                labels[1] = labels[1].to(device)
                y = model(images)
                # y -> 预测值 ------- y[0]:回归预测值 y[1]:分类预测值
                # labels -> 真实值 ------- labels[0]:回归真实值 labels[1]:分类真实值
                loss = lossFn(y, labels)
                reg_acc = cal_reg(y[0].cpu().detach(),labels[0].cpu().detach())
                auroc = cal_auroc(y[1].cpu().detach(), labels[1].cpu().detach(), len(y[0])).mean()
                loss.backward()
                optim.step()
                with torch.no_grad():
                    loss_batch = loss.detach().cpu().numpy()
                    loss_batchs.append(loss_batch)
                    acc_batch = reg_acc.detach().cpu().numpy()
                    acc_batchs.append(acc_batch)
                    auroc_batch = auroc.detach().cpu().numpy()
                    auroc_batchs.append(auroc_batch)

            loss_epochs.append(np.mean(loss_batchs))
            acc_epochs.append(np.mean(acc_batchs))
            auroc_epochs.append(np.mean(auroc_batchs))
            print('Epoch: {}/{}'.format(epoch + 1, self.config.epochs))
            print('Loss: {:.3f}, Reg ACC: {:.2f}%, Cls AUROC: {:.2f}%'.format(np.mean(loss_batchs), np.mean(acc_batchs)*100, np.mean(auroc_batchs)*100))

            # torch.save(model, self.config.model_save_path)

        # torch.save(model.cpu(), self.config.model_save_path)

        log_name = f'{self.config.model_name}-loss-epoch({self.config.epochs})-eta({self.config.learning_rate})-{time.strftime("%Y%m%d%H%M")}'
        data_factory.plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss', log_path=self.config.model_log_path)

        return loss_epochs

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        self.config.training = False
        device = torch.device(self.config.device)

        data_loader = DataLoader(ImageDataset(self.config, data), batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=5)

        model = ImageModule()
        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            print('load model')
            model = torch.load(self.config.model_load_path)

        model = model.to(device)
        model.eval()

        # if self.config.model_load_path and os.path.exists(self.config.model_load_path):
        #     model.load_state_dict(torch.load(self.config.model_load_path))

        eval_data: pd.DataFrame = None
        for images, index_data in tqdm(data_loader):
            images = images.to(device)

            y_regression, y_classify = model(images)

            y_regression = y_regression * 100.0
            index = index_data.cpu().detach().numpy().squeeze()
            y_regression = pd.DataFrame(y_regression.cpu().detach().numpy(), columns=[self.config.label_regression],
                                        index=index)
            y_classify = pd.DataFrame(y_classify.cpu().detach().numpy(), columns=[*self.config.label_classify],
                                      index=index)

            feature_data = data.iloc[index, :]

            batch_data = pd.concat([feature_data, y_regression, y_classify], axis=1)
            if eval_data is None:
                eval_data = batch_data
            else:
                eval_data = pd.concat([eval_data, batch_data], join='inner')

        return eval_data


class CSVModel:
    def __init__(self, config: CSVConfig):
        self.config = config

    def train(self, train_data):
        train_data = resample(train_data, replace=False, random_state=2021)

        self.config.training = True
        device = torch.device(self.config.device)

        data_size = len(train_data)
        batch_size = self.config.batch_size
        batch_count = int(data_size / batch_size)
        # if data_size % batch_size != 0:
        #     batch_count += 1

        model = CSVModule(len(self.config.features))

        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            print('load model')
            model = torch.load(self.config.model_load_path)

        model = model.to(device)
        model.train()

        lossFn = CSVLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_epochs = []
        for epoch in range(self.config.epochs):
            loss_batchs = []
            for batch in range(batch_count):
                index_start = batch * batch_size
                index_end = index_start + batch_size
                batch_features = train_data.iloc[index_start: index_end, :][self.config.features]
                batch_label_regression = train_data[[self.config.label_regression]].iloc[index_start: index_end, :]
                batch_label_classify = train_data[self.config.label_classify].iloc[index_start: index_end, :]

                label_regression = torch.Tensor(batch_label_regression.values).float().to(device)
                label_classify = torch.Tensor(batch_label_classify.values).float().to(device)
                features = torch.Tensor(batch_features.values).float().to(device)

                optim.zero_grad()
                y = model(features)
                loss = lossFn(y, (label_regression, label_classify))
                loss.backward()
                optim.step()

                with torch.no_grad():
                    loss_batch = loss.detach().cpu().numpy()
                    loss_batchs.append(loss_batch)

            loss_epochs.append(np.mean(loss_batchs))

            print(f'{epoch + 1}/{self.config.epochs}\tloss:{np.mean(loss_batchs)}')

            if (epoch + 1) % 100 == 0:
                torch.save(model, self.config.model_save_path)

        torch.save(model.cpu(), self.config.model_save_path)

        log_name = f'{self.config.model_name}-loss-epoch({self.config.epochs})-eta({self.config.learning_rate})-{time.strftime("%Y%m%d%H%M")}'
        data_factory.plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss', log_path=self.config.model_log_path)

        return loss_epochs

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        self.config.training = False
        device = torch.device(self.config.device)

        model = CSVModule(len(self.config.features))

        if self.config.model_load_path and os.path.exists(self.config.model_load_path):
            print('load model')
            model = torch.load(self.config.model_load_path)

        model = model.to(device)
        model.eval()

        features = torch.Tensor(data[self.config.features].values).float().to(device)

        y_regression, y_classify = model(features)

        y_regression = pd.DataFrame(y_regression.cpu().detach().numpy(), columns=[self.config.label_regression],
                                    index=data.index)

        y_classify = y_classify.cpu().detach().numpy()[:, 1]
        y_classify = pd.DataFrame(y_classify, columns=[*self.config.label_classify], index=data.index)

        eval_data = pd.concat([data, y_regression, y_classify], axis=1)
        eval_data.reset_index(inplace=True)
        return eval_data
