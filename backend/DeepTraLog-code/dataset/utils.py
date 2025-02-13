import random
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def generate_train_valid(data, valid_ratio):
    # num_trace = int(len(data) * sample_ratio)
    # # only even number of samples, or drop_last=True in DataLoader API
    # # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # # num_trace += num_trace % 2
    #
    # test_size = int(min(num_trace, len(data)) * valid_size)
    # train_size = int(num_trace - test_size)
    # # only even number of samples
    # # test_size += test_size % 2

    trainset, validset = train_test_split(data, train_size=1-valid_ratio, test_size=valid_ratio, random_state=1234)

    print("train size ", len(trainset))
    print("valid size ", len(validset))
    print("=" * 40)

    return trainset, validset
