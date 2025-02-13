"""
@Time ： 2023/2/20 17:50
@Auth ： Sek Hiunam
@File ：pretrain.py
@Desc ：
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from .model import GGNN
from .optim_schedule import ScheduledOptim
import time
import tqdm
import numpy as np
import pandas as pd

class TEGTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, ggnn:GGNN,train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.001, warmup_steps=10000,warm_up_n_epochs=10,
                 with_cuda: bool = True, cuda_devices=None):
        '''

        :param ggnn:
        :param train_dataloader:
        :param valid_dataloader:
        :param lr:
        :param betas:
        :param weight_decay:
        :param warmup_steps:
        :param with_cuda:
        :param cuda_devices:
        '''
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.model = ggnn.to(self.device)
        self.model = self.model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and valid data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warm_up_n_epochs = warm_up_n_epochs
        self.optim = None
        self.optim_schedule = None
        self.optimizer_name = 'adam'
        self.init_optimizer()

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.hyper_criterion = nn.MSELoss()

        # deep SVDD hyperparameters
        self.radius =  torch.tensor(0.1, device=self.device)  # radius initialized with 0 by default.
        self.center = None
        self.nu = 0.05

        self.objective = "soft-boundary"
        # self.objective = None


        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))


    def init_optimizer(self):
        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.model.out_channels, n_warmup_steps=self.warmup_steps)

        # Set optimizer (Adam optimizer for now)
        self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay/2,
                               amsgrad=self.optimizer_name == 'amsgrad') # weight_decay 指定权值衰减率，相当于L2正则化中的lambda,deeptralog 中正则项前面还有一个1/2，所以要除以2


        # Set learning rate scheduler
        self.optim_schedule = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[60,100], gamma=0.1)

    def train(self, epoch):
        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        str_code = "train"
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        data_loader = self.train_data
        self.model.train()

        # Setting the tqdm progress bar
        totol_length = 0.0
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0
        total_dist = []

        self.optim_schedule.step()
        for i, data in data_iter:
            data = data.to(self.device)
            self.optim.zero_grad()

            outputs = self.model.forward(data).to(self.device)

            dist = torch.sum((outputs - self.center.to(self.device)) ** 2, dim=1)
            total_dist+= dist.cpu().tolist()

            # Update network parameters via backpropagation: forward + backward + optimize
            # 在pytorch中进行L2正则化，最直接的方式可以直接用优化器自带的weight_decay选项指定权值衰减率，相当于L2正则化中的lambda
            scores = dist - self.radius ** 2
            loss = self.radius ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)).to(self.device)

            # 3. backward and optimization only in train
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            totol_length += data.num_graphs
            # we optimize 𝜃 with a fixed 𝑅 in the first few epochs
            # after every k(here k=5) epochs we calculate an optimized value for 𝑅 by linear search.
        if epoch >= self.warm_up_n_epochs and epoch%5 == 0:
            self.radius.data = torch.tensor(self.get_radius(torch.tensor(total_dist), self.nu),
                                            device=self.device)
            print("Update Radius...")

        avg_loss = total_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, loss={}".format(epoch, str_code, avg_loss))

        return avg_loss, total_dist

    def valid(self, epoch):
        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        str_code = "valid"
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        data_loader = self.valid_data
        data_iter = enumerate(data_loader)

        total_loss = 0.0
        totol_length = 0.0
        total_dist = []

        self.model.eval()
        with torch.no_grad():
            for i, data in data_iter:
                outputs = self.model.forward(data)

                dist = torch.sum((outputs - self.center.to(self.device)) ** 2, dim=0)
                total_dist.append(dist)

                # Update network parameters via backpropagation: forward + backward + optimize
                if self.objective == 'soft-boundary':
                    scores = dist - self.radius ** 2
                    loss = self.radius ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                total_loss += loss.item()
                totol_length += data.num_graphs

        avg_loss = total_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, loss={}".format(epoch, str_code, avg_loss))

        return avg_loss, total_dist

    def iteration(self, epoch, data_loader, start_train):
        """
        loop over the data_loader for training or validing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or valid
        :return: None
        """
        str_code = "train" if start_train else "valid"

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = 0.0
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0
        total_dist = []
        if start_train:
            self.optim_schedule.step()

        for i, data in data_iter:
            if start_train:
                self.optim.zero_grad()

            outputs = self.model.forward(data)

            dist = torch.sum((outputs - self.center) ** 2, dim=0)
            total_dist.append(dist)

            # Update network parameters via backpropagation: forward + backward + optimize
            if self.objective == 'soft-boundary':
                scores = dist - self.radius ** 2
                loss = self.radius ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)

            total_loss += loss.item()
            totol_length += data.num_graphs

            # 3. backward and optimization only in train
            if start_train:
                loss.backward()
                self.optim.step()

            if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                self.radius.data = torch.tensor(self.get_radius(torch.tensor(total_dist), self.nu), device=self.device)


        avg_loss = total_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, loss={}".format(epoch, str_code, avg_loss))

        return avg_loss, total_dist

    def save_log(self, save_dir, surfix_log):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(save_dir + key + f"_{surfix_log}.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def save(self, save_dir="output/ggnn_trained.pth"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        torch.save(self.model, save_dir)
        print(" Model Saved on:", save_dir)
        return save_dir

    @staticmethod
    def get_radius(dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


