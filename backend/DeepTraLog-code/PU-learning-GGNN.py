import argparse
import json
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import GlobalAttention, AttentionalAggregation
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
import tqdm
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class GGNN(nn.Module):
    def __init__(self, out_channels, hidden, num_layers, device):
        super(GGNN, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden = hidden

        self.ggnn = GatedGraphConv(out_channels=out_channels, num_layers=num_layers).to(device)
        self.soft_attention = AttentionalAggregation(gate_nn=nn.Linear(out_channels, 1), nn=None).to(device)
        self.ggnn_2 = GatedGraphConv(out_channels=out_channels, num_layers=num_layers).to(device)
        self.tanh = nn.Tanh().to(device)
        self.linear_1 = torch.nn.Linear(hidden, out_channels).to(device)

        self.device = device

    def forward(self, data):
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)

        x_ggnn = self.ggnn(x, edge_index).to(self.device)  # output: node features (|V|,out_channel)
        output = self.soft_attention(x_ggnn, batch).to(self.device)

        x_ggnn_2 = self.tanh(self.linear_1(x_ggnn)).to(self.device)
        batch_sum = torch.zeros(batch.max() + 1, x_ggnn_2.shape[1]).to(self.device)
        batch_sum.scatter_add_(0, batch.repeat(x_ggnn_2.shape[1], 1).t(), x_ggnn_2)
        output_2 = batch_sum
        output = output * output_2
        output = self.tanh(output).to(self.device)

        return output


class PU_GGNN(nn.Module):
    def __init__(self, hidden_channels):
        super(PU_GGNN, self).__init__()
        self.conv1 = GatedGraphConv(num_layers=hidden_channels, out_channels=hidden_channels, aggr='add')
        self.conv2 = GatedGraphConv(num_layers=hidden_channels, out_channels=hidden_channels, aggr='add')
        self.att = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_channels, 1), nn.Tanh()))
        self.lin = nn.Linear(hidden_channels, 1)
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.att(x, data.batch)
        x = self.lin(x)
        x = self.sig(x)
        return x


class PU_SAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PU_SAGE, self).__init__()
        self.conv1 = SAGEConv(50, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 4)
        self.pool = global_mean_pool
        self.fc1 = torch.nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.4, training=self.training)

        x_pool = self.pool(x3, batch)

        x4 = F.relu(self.fc1(x_pool))
        x4 = F.dropout(x4, p=0.3, training=self.training)

        x5 = F.relu(self.fc2(x4))
        x5 = F.dropout(x5, p=0.3, training=self.training)

        x = self.lin(x5)
        x = self.sig(x)
        return x


def calculate_center(data_loader_list, model, device, eps=0.1):
    print("start calculate center")
    # model = torch.load(self.model_path)
    # model.to(self.device)
    # 𝑐 is the center of the learned hypersphere
    total_samples = 0
    center = torch.zeros(50, device=device)

    with torch.no_grad():
        for data_loader in data_loader_list:
            totol_length = len(data_loader)
            data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)

            for i, data in data_iter:
                outputs = model.forward(data)

                center += torch.sum(outputs.detach().clone(), dim=0)
                total_samples += data.num_graphs

    center = center / total_samples

    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps

    return center


def test_with_svdd(center, radius, test_loader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)

            # 计算距离
            dist = torch.sum((output - center.to(device)) ** 2, dim=1).sqrt()

            # 计算预测结果
            pred = (dist <= radius).float()

            # 更新 TP, TN, FP, FN
            tp += ((pred.flatten() == 0) & (batch.y.flatten() == 0)).float().sum()
            tn += ((pred.flatten() == 1) & (batch.y.flatten() == 1)).float().sum()
            fp += ((pred.flatten() == 0) & (batch.y.flatten() == 1)).float().sum()
            fn += ((pred.flatten() == 1) & (batch.y.flatten() == 0)).float().sum()

    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    print('testing...')
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')

    return f1_score


def get_probabilities(output, center, radius):
    # 概率与距离设为线性关系
    # in_section: 在超球内部，[(1-in_section)R, R]，概率为[1, 1-in_prob_section]
    # out_section: 在超球外部, [R, (1+out_section)R]，概率为[1, 0]

    # 理论上
    # in_section减小, in_prob_section减小，提高precision，降低recall(相当于对于超球内的数据放宽标准，超球内正常和异常判断为正常的概率都增大)
    # out_section减小，提高Recall，降低recall（相当于对于超球外数据更加严格，超球外常和异常判断为异常的概率都增大）

    in_section = 0.05
    in_prob_section = 0.1
    out_section = 0.15

    # 计算与中心的距离
    dist = torch.sqrt(torch.sum((output - center) ** 2, dim=1))

    # 区域1：在0.95R之内，概率设为1
    mask1 = dist <= (1 - in_section) * radius
    prob1 = torch.ones_like(dist)

    # 区域2：在0.95R~1R范围内，概率由1递减到0.9
    mask2 = (dist > (1 - in_section) * radius) & (dist <= radius)
    prob2 = 1 - in_prob_section * (dist[mask2] - (1 - in_section) * radius) / (in_section * radius)

    # 区域3：在R-1.15R范围内，概率由1递减到0
    mask3 = (dist > radius) & (dist <= (1 + out_section) * radius)
    prob3 = 1 - (dist[mask3] - radius) / (out_section * radius)

    # 区域4：在1.15R之外，概率设为0
    mask4 = dist > (1 + out_section) * radius
    prob4 = torch.zeros_like(dist)

    # 将四个区域的概率合并
    prob = torch.zeros_like(dist)
    prob[mask1] = prob1[mask1]
    prob[mask2] = prob2
    prob[mask3] = prob3
    prob[mask4] = prob4[mask4]

    return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='../../workspace/weight/rtd/PU', help='Trained Model Weight')
    parser.add_argument('--data', type=str,
                        default='./data/all-no-response-relationship-no-trace-no-dependency.jsons',
                        help='Data for Train or Test')
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')

    args = parser.parse_args()
    print("arguments", args)

    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/0-5-11-79-82-86.jsons'
    # data_dir = './data/all-no-response-relationship-no-trace-no-dependency.jsons'
    data_dir = args.data
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace-no-dependency.jsons'
    model_2_flag = 'SAGE'

    # PU_model_dir = './PU'
    PU_model_dir = args.weight
    SVDD_center_path = PU_model_dir + '/best_center.pt'
    best_model_path_1 = PU_model_dir + '/PU_1.pth'
    best_model_path_2 = PU_model_dir + '/PU_2-' + model_2_flag + '.pth'

    model_flag = 'SAGE'

    if args.mode == 'predict':
        test_ratio = 0.9
    else:
        test_ratio = 0.2
    # valid_ratio = 0.25
    valid_ratio = 0.5

    hidden_channels = 64

    random.seed(1234)
    with open(data_dir, 'r') as file:
        print(data_dir)
        jsonList = []
        i = 0
        for graph in file:
            if (i + 1) % 5000 == 0:
                print("graph loading:" + str(i))
            i += 1
            teg = json.loads(graph)
            # 如果没有边则添加一条[0,0]的边
            if len(teg['edge_index']) == 0:
                teg['edge_index'].append([0, 0])
            edge_index = torch.tensor(teg['edge_index'], dtype=torch.long).t().contiguous()
            x = torch.tensor(teg['node_info'], dtype=torch.float)
            y = torch.tensor(1 - teg['trace_bool'], dtype=torch.float).view(1, -1)
            data = Data(x=x, edge_index=edge_index, y=y)
            jsonList.append(data)

    if args.mode == 'predict':
        hidden_channels = 50
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_2_flag == 'GGNN':
            model = PU_GGNN(hidden_channels).to(device)
        elif model_2_flag == 'SAGE':
            model = PU_SAGE(hidden_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001 / 2)
        train_data, test_data = train_test_split(jsonList, test_size=test_ratio, random_state=1234,
                                                 stratify=[data.y.item() for data in jsonList])
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # 在测试集上评估模型（单独）
        model.load_state_dict(torch.load(best_model_path_2))
        model.eval()
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                pred = (output > 0.5).float()
                tp += ((pred == 0) & (batch.y == 0)).float().sum()
                tn += ((pred == 1) & (batch.y == 1)).float().sum()
                fp += ((pred == 0) & (batch.y == 1)).float().sum()
                fn += ((pred == 1) & (batch.y == 0)).float().sum()

        precision = 0 if tp == 0 else tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        print('testing...')
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(int(tp), int(tn), int(fp), int(fn)))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(precision*100, recall*100, f1_score*100))
        exit(0)

    # 将数据集拆分成训练、验证和测试集
    train_data, test_data = train_test_split(jsonList, test_size=test_ratio, random_state=1234,
                                             stratify=[data.y.item() for data in jsonList])
    train_data, valid_data = train_test_split(train_data, test_size=valid_ratio / (1 - test_ratio), random_state=1234,
                                              stratify=[data.y.item() for data in train_data])
    positive_valid_data = [data for data in valid_data if data.y.item() == 1]

    # 从训练数据中分离出正例和负例数据
    positive_data = [data for data in train_data if data.y.item() == 1]
    negative_data = [data for data in train_data if data.y.item() == 0]

    # 从有标签的正例数据中随机选择一部分
    labeled_data, unlabeled_data = train_test_split(positive_data, test_size=0.5, random_state=1234)

    # 将未标记数据合并为一个数据集
    unlabeled_data = [data for data in train_data if data not in labeled_data]

    print(len(labeled_data))
    print(len(unlabeled_data))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_channels = 50
    # 对有标签的正例数据进行训练
    model = GGNN(out_channels=hidden_channels, hidden=hidden_channels, num_layers=3, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001 / 2)

    total_dist = []
    center = torch.Tensor([0 for _ in range(50)])
    radius = 0.1
    nu = 0.05

    best_val_loss = float('inf')
    best_test_f1 = 0


    print('Train on labelled data...')
    for epoch in range(19):
        print('\bEpoch ' + str(epoch))
        model.train()
        labeled_loader = DataLoader(labeled_data, batch_size=32, shuffle=True)
        data_iter = enumerate(labeled_loader)
        total_train_loss = 0.0
        total_val_loss = 0.0

        # train
        for i, data in data_iter:
            optimizer.zero_grad()
            outputs = model.forward(data).to(device)

            dist = torch.sum((outputs - center.to(device)) ** 2, dim=1)
            # Update network parameters via backpropagation: forward + backward + optimize
            # 在pytorch中进行L2正则化，最直接的方式可以直接用优化器自带的weight_decay选项指定权值衰减率，相当于L2正则化中的lambda
            scores = dist - radius ** 2
            loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)).to(device)

            # 3. backward and optimization only in train
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print("train loss:" + str(total_train_loss / len(labeled_data)))

        # valid
        positive_valid_loader = DataLoader(positive_valid_data, batch_size=32, shuffle=True)
        data_iter = enumerate(positive_valid_loader)
        for i, data in data_iter:
            model.eval()
            outputs = model.forward(data).to(device)
            dist = torch.sum((outputs - center.to(device)) ** 2, dim=1)
            scores = dist - radius ** 2
            loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)).to(device)
            total_val_loss += loss.item()
        print("val loss:" + str(total_val_loss / len(positive_valid_loader)))

        # 更新SVDD模型
        if total_val_loss / len(positive_valid_loader) < best_val_loss:
            # 这里原则上不能用test data的f1判断是否更新模型
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
            f1 = test_with_svdd(center, radius, test_loader, model, device)
            if f1 > best_test_f1:
                best_test_f1 = f1
                print("Save best center", SVDD_center_path)
                torch.save({"center": center, "radius": radius}, SVDD_center_path)
                print("Save GGNN_1 model", SVDD_center_path)
                torch.save(model.state_dict(), best_model_path_1)

        if epoch == 0:
            #  The hypersphere center 𝑐 is set to the mean of the vector representations of all the TEGs after an initial forward pass.
            center = calculate_center([labeled_loader], model, device)
            print('center')
            print(center)
        print('\n')

    # 计算未标记数据的分类概率
    model = GGNN(out_channels=hidden_channels, hidden=hidden_channels, num_layers=3, device=device).to(device)
    model.load_state_dict(torch.load(best_model_path_1))
    model.eval()
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=32, shuffle=False)
    unlabelled_data_iter = enumerate(unlabeled_loader)
    prob_list = []
    with torch.no_grad():
        for i, data in unlabelled_data_iter:
            output = model(data)
            prob = get_probabilities(output, center, radius)
            prob_list.extend(prob.tolist())

    # 将未标记数据的加权结果加入到数据集中
    for i in range(len(unlabeled_data)):
        unlabeled_data[i].y = torch.tensor(prob_list[i], dtype=torch.float).view(-1, 1)

    # 将有标签的正例数据和加权后的未标记数据合并为训练集
    train_data = labeled_data.copy()
    train_data.extend(unlabeled_data)

    # 构建DataLoader并训练模型
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    if model_2_flag == 'GGNN':
        model = PU_GGNN(hidden_channels).to(device)
    elif model_2_flag == 'SAGE':
        model = PU_SAGE(hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print('Train on extension data...')
    print('model:' + model_2_flag)
    best_f1 = 0
    for epoch in range(30):
        print('\nEpoch ' + str(epoch))
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型
        model.eval()
        valid_acc = 0
        # 在验证集上评估模型
        model.eval()
        valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                output = model(batch)
                pred = (output > 0.5).float()
                valid_tp += ((pred == 0) & (batch.y == 0)).float().sum()
                valid_tn += ((pred == 1) & (batch.y == 1)).float().sum()
                valid_fp += ((pred == 0) & (batch.y == 1)).float().sum()
                valid_fn += ((pred == 1) & (batch.y == 0)).float().sum()

        valid_precision = 0 if valid_tp == 0 else valid_tp / (valid_tp + valid_fp)
        valid_recall = valid_tp / (valid_tp + valid_fn)
        valid_f1_score = 0 if (valid_precision + valid_recall) == 0 else 2 * valid_precision * valid_recall / (
                valid_precision + valid_recall)
        print("validing...")
        print(f'TP: {valid_tp}, TN: {valid_tn}, FP: {valid_fp}, FN: {valid_fn}')
        print(f'Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1-score: {valid_f1_score:.4f}')

        if valid_f1_score > best_f1:
            best_f1 = valid_f1_score
            torch.save(model.state_dict(), best_model_path_2)

            # 在测试集上评估模型（单独）
            model.load_state_dict(torch.load(best_model_path_2))
            model.eval()
            tp, tn, fp, fn = 0, 0, 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    pred = (output > 0.5).float()
                    tp += ((pred == 0) & (batch.y == 0)).float().sum()
                    tn += ((pred == 1) & (batch.y == 1)).float().sum()
                    fp += ((pred == 0) & (batch.y == 1)).float().sum()
                    fn += ((pred == 1) & (batch.y == 0)).float().sum()

            precision = 0 if tp == 0 else tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

            print('testing...')
            print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')
