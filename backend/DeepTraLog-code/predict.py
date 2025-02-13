"""
@Time ： 2023/2/14 下午6:11
@Auth ： Sek Hiunam
@File ：predict.py
@Desc ：
"""

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import time
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset.trace_event_graph import TEGraph


def compute_anomaly(results, seq_threshold=0.0):
    total_errors = 0
    for ans in results:
        if ans>seq_threshold:
            total_errors+=1
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, thres, th_range, seq_range,normal,abnormal):
    best_result = [0] * 9
    if thres:
        for seq_th in seq_range:
            test_normal_anomaly = compute_anomaly(test_normal_results, seq_th)
            test_abnormal_anomaly = compute_anomaly(test_abnormal_results, seq_th)

            # 所有日志的总数为:
            FP = test_normal_anomaly # 将正常的错误判断为异常的
            TP = test_abnormal_anomaly # 将异常的正确判断为异常的

            if TP == 0:
                continue

            TN = normal - FP
            FN = abnormal - TP
            P = 100.0 * TP / (TP + FP)
            R = 100.0 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)


            if F1 > best_result[-1]:
                best_result = [0, seq_th, FP, TP, TN, FN, P, R, F1]
                print("GroundTruth Anomaly:",abnormal)
                print("Predicting Anomaly:", test_abnormal_anomaly+test_normal_anomaly)
    else:
        test_normal_anomaly = compute_anomaly(test_normal_results, seq_threshold=0)
        test_abnormal_anomaly = compute_anomaly(test_abnormal_results, seq_threshold=0)

        FP = test_normal_anomaly  # 将正常的错误判断为异常的
        TP = test_abnormal_anomaly  # 将异常的正确判断为异常的

        TN = normal - FP
        FN = abnormal - TP
        P = 100.0 * TP / (TP + FP)
        R = 100.0 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        best_result = [0, 0, FP, TP, TN, FN, P, R, F1]

    return best_result


class Predictor():
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
        self.hidden = options["hidden"]

        self.epochs = options["epochs"]

        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]

        self.output_dir = options["output_dir"]
        self.model_dir = options["model_dir"]

        self.center = None
        self.radius = None
        self.test_ratio = options["test_ratio"]

        self.event_type = {"LogEvent": 0, "ServerRequest": 1, "ServerResponse": 2, "ClientRequest": 3, "ClientResponse": 4,
                      "Consumer": 5, "Producer": 6}
        self.edge_type = {'Sequence': 0, 'SynRequest': 1, 'SynResponse': 2, 'AsynRequest': 3}

    def helper(self, model, test_data, is_abnormal):
        total_results = []
        anomaly_scores = []

        # use 1/10 test data
        # if self.test_ratio != 1:
        #     _, test_data = train_test_split(test_data, test_size=self.test_ratio)


        data_loader =DataLoader(test_data, batch_size=1, num_workers=2,
                                            drop_last=True)


        for idx, data in enumerate(data_loader):
            outputs = model.forward(data)
            ans = torch.sum((outputs - self.center) ** 2, dim=1) - self.radius ** 2

            # anomaly_scores.append(ans)
            # total_results.append(int(ans>0)) # For a TEG of a trace,if its anomaly score is greater than 0 it is treated as anomalous.

            anomaly_scores+= ans.cpu().tolist()
            # dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
            # when visualization no mask
            # continue

            # # loop though each session in batch
            # if idx < 10 or idx % 1000 == 0:
            #     print(
            #         "TraceBatch {}, #anomaly score: {} # deepSVDD_label: {} \n".format(
            #             idx,
            #             ans,
            #             is_abnormal
            #         )
            #     )

        # for hypersphere distance
        return total_results, anomaly_scores

    def predict(self):
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))

        start_time = time.time()

        center_dict = torch.load(self.model_dir + "best_center.pt")
        self.center = center_dict["center"]
        self.radius = center_dict["radius"]

        normal, abnormal = TEGraph(event_type=self.event_type,edge_type=self.edge_type,
                             graph_corpus=None,log_corpus=None,span_corpus=None,out_dir=self.output_path,embed_dir=self.output_path).load_graphs_from_jsons(filepath=self.data_dir,is_Train=False, test_ratio=self.test_ratio)


        print("test normal predicting")
        test_normal_results, test_normal_ans = self.helper(model, normal,is_abnormal=False)

        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_ans = self.helper(model, abnormal,is_abnormal=True)

        # print("Saving test normal results")
        # with open(self.model_dir + "test_normal_results", "wb") as f:
        #     pkl.dump(test_normal_results, f)
        #
        # print("Saving test abnormal results")
        # with open(self.model_dir + "test_abnormal_results", "wb") as f:
        #     pkl.dump(test_abnormal_results, f)

        print("Saving test normal errors")
        with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
            pkl.dump(test_normal_ans, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
            pkl.dump(test_abnormal_ans, f)

        print("Total Traces: ", len(normal)+len(abnormal), "Total Anomaly: ", len(abnormal))

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
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
