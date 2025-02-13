import argparse
import os
import random
import json
import torch
import torch.nn.functional as F
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
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        # x2 = self.conv2(x1, edge_index)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, p=0.4, training=self.training)

        # x3 = self.conv3(x2, edge_index)
        # x3 = F.relu(x3)
        # x3 = F.dropout(x3, p=0.4, training=self.training)

        x_pool = self.pool(x1, batch)

        # x4 = F.relu(self.fc1(x_pool))
        # x4 = F.dropout(x4, p=0.3, training=self.training)

        # x3 = F.relu(self.fc2(x_pool))
        # x3 = F.dropout(x3, p=0.3, training=self.training)

        x = self.classifier(x_pool)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')

    args = parser.parse_args()
    print("arguments", args)

    data_dir = './data/jsons/all-no-response-relationship-no-trace.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-no-trace-no-span_dependency-no-log_sequence.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace-no-dependency.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-log.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/0-5-11-79-82-86.jsons'

    best_model_path = 'best_model-50d.pth'

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
                # 如果没有边则添加一条[0,0]的边
                if len(teg['edge_index']) == 0:
                    teg['edge_index'].append([0, 0])
                edge_index = torch.tensor(teg['edge_index'], dtype=torch.long).t().contiguous()
                x = torch.tensor(teg['node_info'], dtype=torch.float)
                y = torch.tensor(teg['trace_bool'], dtype=torch.long).view(1)
                data = Data(x=x, edge_index=edge_index, y=y)
                jsonList.append(data)
    train_dataset, test_dataset = train_test_split(jsonList, test_size=test_ratio, random_state=1234)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=valid_ratio, random_state=1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        # Count positive and negative samples in jsonList
        positive_samples_jsonList = sum([1 for data in jsonList if data.y.item() == 1])
        negative_samples_jsonList = sum([1 for data in jsonList if data.y.item() == 0])
        print(f'Positive samples in dataset: {positive_samples_jsonList}')
        print(f'Negative samples in dataset: {negative_samples_jsonList}')
        # Count positive and negative samples in train_dataset
        positive_samples_train_dataset = sum([1 for data in train_dataset if data.y.item() == 1])
        negative_samples_train_dataset = sum([1 for data in train_dataset if data.y.item() == 0])
        print(f'Positive samples in train_dataset: {positive_samples_train_dataset}')
        print(f'Negative samples in train_dataset: {negative_samples_train_dataset}')

        # Count positive and negative samples in test_dataset
        positive_samples_test_dataset = sum([1 for data in test_dataset if data.y.item() == 1])
        negative_samples_test_dataset = sum([1 for data in test_dataset if data.y.item() == 0])
        print(f'Positive samples in test_dataset: {positive_samples_test_dataset}')
        print(f'Negative samples in test_dataset: {negative_samples_test_dataset}')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = GCN(hidden_channels=hidden_channels).to(device)

        # Training and validation
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        positive_count = sum([data.y.item() for data in train_dataset])
        negative_count = len(train_dataset) - positive_count
        positive_weight = negative_count / (positive_count + negative_count)
        negative_weight = positive_count / (positive_count + negative_count)
        weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

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
                correct = 0
                total = 0
                with torch.no_grad():
                    TP, TN, FP, FN = 0, 0, 0, 0
                    total_positive, total_negative = 0, 0
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.batch)
                        _, pred = out.max(dim=1)

                        total_positive += (data.y == 1).sum().item()
                        total_negative += (data.y == 0).sum().item()

                        TP += ((pred == 1) & (data.y == 1)).sum().item()
                        TN += ((pred == 0) & (data.y == 0)).sum().item()
                        FP += ((pred == 1) & (data.y == 0)).sum().item()
                        FN += ((pred == 0) & (data.y == 1)).sum().item()

                    print(f'Total Positive Samples: {total_positive}')
                    print(f'Total Negative Samples: {total_negative}')
                    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

                    precision = TP / (TP + FP) if TP + FP > 0 else 0
                    recall = TP / (TP + FN) if TP + FN > 0 else 0
                    F1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {F1_score:.4f}')

    elif args.mode == 'predict':
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = GCN(hidden_channels=hidden_channels).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            TP, TN, FP, FN = 0, 0, 0, 0
            total_positive, total_negative = 0, 0
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                _, pred = out.max(dim=1)

                total_positive += (data.y == 1).sum().item()
                total_negative += (data.y == 0).sum().item()

                TP += ((pred == 1) & (data.y == 1)).sum().item()
                TN += ((pred == 0) & (data.y == 0)).sum().item()
                FP += ((pred == 1) & (data.y == 0)).sum().item()
                FN += ((pred == 0) & (data.y == 1)).sum().item()

            print(f'Total Positive Samples: {total_positive}')
            print(f'Total Negative Samples: {total_negative}')
            print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            F1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {F1_score:.4f}')