import argparse
import os
import random
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(50, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 4)
        self.pool = global_mean_pool
        self.fc1 = torch.nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 15)

    def forward(self, x, edge_index, batch):
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

        x = self.classifier(x5)
        return x


def count_samples_by_class(dataset):
    class_counts = {i: 0 for i in range(15)}
    for data in dataset:
        class_counts[data.y.item()] += 1
    return class_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='../../workspace/weight/rtd/classify/best_model-50d.pth', help='Trained Model Weight')
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

    data_dir = args.data
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/process_all.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace-no-dependency.jsons'
    # data_dir = './data/jsons/all-no-response-relationship-no-trace.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-log.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/0-5-11-79-82-86.jsons'

    # best_model_path = 'best_error_classify_model-no-response-relationship-no-trace.pth'
    best_model_path = args.weight
    if args.mode == 'predict':
        test_ratio = 0.9
    else:
        test_ratio = 0.2
    valid_ratio = 0.25

    hidden_channels = 64

    random.seed(1234)
    with open(data_dir, 'r') as file:
        jsonList = []
        i = 0
        with open(data_dir) as f:
            for graph in f:
                if i % 5000 == 0:
                    print("graph loading:" + str(i))
                i += 1
                teg = json.loads(graph)
                if teg['trace_bool'] == 0:
                    continue
                edge_index = torch.tensor(teg['edge_index'], dtype=torch.long).t().contiguous()
                x = torch.tensor(teg['node_info'], dtype=torch.float)
                y = torch.tensor(teg['error_type'], dtype=torch.long).view(1)
                data = Data(x=x, edge_index=edge_index, y=y)
                jsonList.append(data)
    train_dataset, test_dataset = train_test_split(jsonList, test_size=test_ratio, random_state=1234)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=valid_ratio, random_state=1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        # Count samples by class in jsonList
        samples_by_class_jsonList = count_samples_by_class(jsonList)
        print(f'Samples by class in dataset: {samples_by_class_jsonList}')

        # Count samples by class in train_dataset
        samples_by_class_train_dataset = count_samples_by_class(train_dataset)
        print(f'Samples by class in train_dataset: {samples_by_class_train_dataset}')

        # Count samples by class in test_dataset
        samples_by_class_test_dataset = count_samples_by_class(test_dataset)
        print(f'Samples by class in test_dataset: {samples_by_class_test_dataset}')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = GCN(hidden_channels=hidden_channels).to(device)

        # Training and validation
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        for epoch in range(1, 101):
            print(f'\nEpoch: {epoch:03d}')
            # train
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)

            # validate
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y.long())
                    total_loss += loss.item()
            val_loss = total_loss / len(val_loader)

            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('saving best model...')
                torch.save(model.state_dict(), best_model_path)
                model.eval()
                with torch.no_grad():
                    y_true = []
                    y_pred = []

                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.batch)
                        _, pred = out.max(dim=1)
                        y_true.extend(data.y.cpu().numpy())
                        y_pred.extend(pred.cpu().numpy())

                    cm = confusion_matrix(y_true, y_pred)
                    print("Confusion Matrix:\n", cm)
                    report = classification_report(y_true, y_pred)
                    print("Classification Report:\n", report)

    elif args.mode == 'predict':
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = GCN(hidden_channels=hidden_channels).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []

            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                _, pred = out.max(dim=1)
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:\n", cm)
            report = classification_report(y_true, y_pred)
            print("Classification Report:\n", report)
            # 解析报告，提取平均值
            lines = report.split('\n')
            avg_line = lines[-2]  # 倒数第二行是平均值行

            # 提取平均值的 Precision、Recall 和 F1-score
            precision_avg = float(avg_line.split()[2])
            recall_avg = float(avg_line.split()[3])
            f1_score_avg = float(avg_line.split()[4])
            # precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(precision_avg*100, recall_avg*100, f1_score_avg*100))
            # print("Classification Report:\n", report)