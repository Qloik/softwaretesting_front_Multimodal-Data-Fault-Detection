"""
@Time ： 2023/2/19 16:42
@Auth ： Sek Hiunam
@File ：DeepTraLog.py
@Desc ：
"""
import sys

import torch

sys.path.append("../")

import argparse
from train import Trainer
from predict import Predictor
from dataset.utils import seed_everything

options = dict()
options[
    'device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # donot forget to set options["with_cuda"] correspondently

options["output_dir"] = '../../workspace/weight/rtd/'
options["model_dir"] = options["output_dir"] + 'DTL/'
options["model_path"] = options["model_dir"] + "best_model.pth"
options["data_dir"] = './data/all-no-response-relationship-no-trace-no-dependency.jsons'
# options["data_dir"] = './workspace/multimodal/data/DeepTraLog/v4/0-5-11-79-82-86.jsons'  # graph dir
# options["data_dir"] = './workspace/multimodal/data/DeepTraLog/v4/0_process_0-F01-01--parsed_SUCCESSF0101_SpanData2021-08-14_10-22-48.jsons'
# options["data_dir"] = './workspace/multimodal/data/DeepTraLog/v2/GraphData/0_process_0_parsed_ERROR_F012_SpanData2021-08-14_01-52-43.jsons'  # graph dir

event_type = {"LogEvent": 4, "ServerRequest": 0, "ServerResponse": 2, "ClientRequest": 1,
              "ClientResponse": 3, "Consumer": 7,
              "Producer": 6, "MongoDB": 5, 'Internal': 8}
options["edge_type"] = {'Sequence': 0, 'SynRequest': 1, 'SynResponse': 2, 'AsynRequest': 3, 'MongoDB': 4, 'Other': -1,
                        "Invalid": -2}

options["train_ratio"] = 0.8
options["valid_ratio"] = 0.15  # 这里的valid_ratio是train set中的比例
options["test_ratio"] = 1 - options["train_ratio"]

# model
options["hidden"] = 50  # embedding size
options["num_layers"] = 3

options["epochs"] = 60
options["n_epochs_stop"] = 40
options["warm_up_n_epochs"] = 1
options["batch_size"] = 100

options["on_memory"] = True
options["num_workers"] = 2
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.001
options["with_cuda"] = True
options["cuda_devices"] = None

seed_everything(seed=1234)
print("device", options["device"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./anrr', help='Trained Model Weight')
    parser.add_argument('--data', type=str,
                        default='./test.jsons',
                        help='Data for Train or Test')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')

    args = parser.parse_args()
    print("arguments", args)
                # Trainer(options).train()
    # Predictor(options).predict()

    if args.mode == 'train':
        Trainer(options).train()

    elif args.mode == 'predict':
        options["model_dir"] = args.weight
        options["model_path"] = options["model_path"] = options["model_dir"] + "best_model.pth"
        options["data_dir"] = args.data
        options["train_ratio"] = 0.2
        options["test_ratio"] = 0.8
        Predictor(options).predict()
