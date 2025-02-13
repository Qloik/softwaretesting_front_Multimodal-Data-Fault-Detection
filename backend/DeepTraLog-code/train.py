"""
@Time ： 2023/2/14 下午6:11
@Auth ： Sek Hiunam
@File ：train.py
@Desc ：
"""
import os
import sys

from predict import find_best_threshold

sys.path.append("../")
dirname = os.path.dirname(__file__)

from torch_geometric.loader import DataLoader

from model.model import GGNN
from model.pretrain import TEGTrainer
from dataset.TEGDataset import TEGDataset
from dataset.trace_event_graph import TEGraph
from dataset.utils import save_parameters,generate_train_valid
import pickle as plk
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm
import gc
import numpy as np

class Trainer():
    def __init__(self, options):
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.output_path = options["output_dir"]
        self.data_dir = options["data_dir"]

        self.event_type = options["edge_type"]
        self.edge_type = options["edge_type"]

        self.train_ratio = options["train_ratio"]
        self.valid_ratio = options["valid_ratio"]

        # self.graph_corpus = options["graph_corpus"]
        # self.logTem2Embed = options["logTem2Embed"]
        # self.spanTem2Embed = options["spanTem2Embed"]
        #
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]

        self.hidden = options["hidden"]
        self.num_layers = options["num_layers"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_weight_decay = options["adam_weight_decay"]
        self.with_cuda = options["with_cuda"]
        self.cuda_devices = options["cuda_devices"]

        self.epochs = options["epochs"]
        self.warm_up_n_epochs = options["warm_up_n_epochs"]
        # self.hidden = options["hidden"] # out_channel
        # self.num_layers = options["num_layers"]
        #
        self.n_epochs_stop = options["n_epochs_stop"]

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")

    def train(self):
        # graph data
        graph_construct = TEGraph(event_type=self.event_type,edge_type=self.edge_type,
                             graph_corpus=None,log_corpus=None,span_corpus=None,out_dir=self.output_path,embed_dir=self.output_path) # 这里仅仅是graph一些参数
        graph_data, test_normal, test_abnormal = graph_construct.load_graphs_from_jsons(filepath=self.data_dir,is_Train=True, test_ratio=1-self.train_ratio)
        self.test_normal = test_normal
        self.abnormal = test_abnormal

        print("DataSet Size: ", len(graph_data))

        train_dataset, valid_dataset = generate_train_valid(graph_data, valid_ratio=self.valid_ratio)

        # print("\nLoading Train Dataset")
        # train_dataset = TEGDataset(train_dataset)
        # print("\nLoading valid Dataset")
        # valid_dataset = TEGDataset(valid_dataset)

        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                            drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                            drop_last=True)

        del train_dataset
        del valid_dataset
        del graph_data
        gc.collect()

        print("Building model")
        ggnn = GGNN(out_channels=self.hidden,hidden=self.hidden,num_layers=self.num_layers,device=self.device)

        print("Creating Trainer")
        self.trainer = TEGTrainer(ggnn,train_dataloader=self.train_data_loader, valid_dataloader=self.valid_data_loader,
                              lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay,
                              with_cuda=self.with_cuda, cuda_devices=self.cuda_devices)

        self.start_iteration(surfix_log="log2")

        self.plot_train_valid_loss("_log2")

    def start_iteration(self, surfix_log):
        print("Training Start")
        best_loss = float('inf')
        epochs_no_improve = 0
        # Initialize hypersphere center c (if c not loaded)
        if self.trainer.center is None:
            print('Initializing center ...')
            self.trainer.center = torch.Tensor([0.1 for _ in range(
                self.hidden)])
            # self.trainer.center = self.calculate_center([self.train_data_loader, self.valid_data_loader])
            print('Center initialized.')

        for epoch in range(self.epochs):
            print("\n")

            _, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            print("radius:{}".format(self.trainer.radius))
            self.trainer.save_log(self.model_dir, surfix_log)

            if epoch < self.warm_up_n_epochs:
                #  The hypersphere center 𝑐 is set to the mean of the vector representations of all the TEGs after an initial forward pass.
                self.trainer.center = self.calculate_center([self.train_data_loader, self.valid_data_loader])

            # save model after 10 warm up epochs
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                best_center = self.trainer.center
                best_radius = self.trainer.radius
                total_dist = train_dist + valid_dist

                if best_center is None:
                    raise TypeError("center is None")

                print("best radius", best_radius)
                best_center_path = self.model_dir + "best_center.pt"
                print("Save best center", best_center_path)
                torch.save({"center": best_center, "radius": best_radius}, best_center_path)

                total_dist_path = self.model_dir + "best_total_dist.pt"
                print("save total dist: ", total_dist_path)
                torch.save(total_dist, total_dist_path)
                # self.predict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping")
                break

    def calculate_center(self, data_loader_list,eps=0.1):
        print("start calculate center")
        # model = torch.load(self.model_path)
        # model.to(self.device)
        # 𝑐 is the center of the learned hypersphere
        total_samples = 0
        center = torch.zeros(self.trainer.model.out_channels, device=self.device)

        with torch.no_grad():
            for data_loader in data_loader_list:
                totol_length = len(data_loader)
                data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)

                for i, data in data_iter:
                    outputs = self.trainer.model.forward(data)

                    center += torch.sum(outputs.detach().clone(), dim=0)
                    total_samples += data.num_graphs

        center = center / total_samples

        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps


        return center

    def plot_train_valid_loss(self, surfix_log):
        train_loss = pd.read_csv(self.model_dir + f"train{surfix_log}.csv")
        valid_loss = pd.read_csv(self.model_dir + f"valid{surfix_log}.csv")
        sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
        sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
        plt.title("epoch vs train loss vs valid loss")
        plt.legend()
        plt.savefig(self.model_dir + "train_valid_loss.png")
        plt.show()
        print("plot done")


    def helper(self, model, test_data, is_abnormal):
        total_results = []
        anomaly_scores = []

        data_loader =DataLoader(test_data, batch_size=1, num_workers=2,
                                            drop_last=True)

        for idx, data in enumerate(data_loader):
            outputs = model.forward(data)
            ans = torch.sum((outputs - self.trainer.center) ** 2, dim=1) - self.trainer.radius ** 2
            anomaly_scores+= ans.cpu().tolist()
        return total_results, anomaly_scores

    def predict(self):
        normal = self.test_normal
        abnormal = self.abnormal
        print('starting testing...')
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))
        print("test normal predicting")
        test_normal_results, test_normal_ans = self.helper(model, normal, is_abnormal=False)

        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_ans = self.helper(model, abnormal, is_abnormal=True)

        print("Total Traces: ", len(normal) + len(abnormal), "Total Anomaly: ", len(abnormal))

        best_th, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(test_normal_ans,
                                                                             test_abnormal_ans,
                                                                             thres=False,
                                                                             th_range=np.arange(100),
                                                                             seq_range=np.arange(0, 1, 0.01),
                                                                             normal=len(test_normal_ans),
                                                                             abnormal=len(test_abnormal_ans))

        # print("best threshold: {}, best threshold ratio: {}".format(best_th, best_seq_th))
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))

#
# if __name__ == '__main__':
#     options = dict()
#     options["output_dir"] = './output/'
#     options["model_dir"] = options["output_dir"]+'ggnn/'
#     options["model_path"] = options["model_dir"] + "best_model.pth"
#
#     options["on_memory"] = True
#     options["num_workers"] = 2
#     options["batch_size"] = 8
#
#     options["epochs"] = 100
#     options["n_epochs_stop"] =15
#     options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
#     options["lr"] = 1e-4
#     options["adam_beta1"] = 0.9
#     options["adam_beta2"] = 0.999
#     options["adam_weight_decay"] = 0.00
#     options["with_cuda"] = True
#     options["cuda_devices"] = None
#
#     output_dir = options["output_dir"]
#     with open (output_dir + 'traces/traTem2embed.pkl', "rb") as f:
#         spanTem2embed = plk.load(f)
#     with open (output_dir + "logs/logTem2embed.pkl", "rb") as f:
#         logTem2embed = plk.load(f)
#
#     t = Trainer(options=options).train()

if __name__ == '__main__':
    model_dir = './output/ggnn/'
    surfix_log = '_log2'
    train_loss = pd.read_csv(model_dir + f"train{surfix_log}.csv")
    valid_loss = pd.read_csv(model_dir + f"valid{surfix_log}.csv")
    train_loss = train_loss['loss'].tolist()
    valid_loss = valid_loss['loss'].tolist()
    epoch = list(range(len(train_loss)))
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, valid_loss, label='val_loss')
    plt.title("epoch vs train loss vs valid loss")
    plt.legend()
    plt.savefig(model_dir + "train_valid_loss.png")
    plt.show()
    print("plot done")


