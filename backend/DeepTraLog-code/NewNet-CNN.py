import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class CNN(nn.Module):
    def __init__(self, hidden_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.fc2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data_list, max_size):
        self.data_list = data_list
        self.max_size = max_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        # Zero-pad the tensor along the first dimension
        padded_x = torch.zeros(self.max_size, x.size(1))
        padded_x[:x.size(0), :] = x
        return padded_x, y


def custom_collate_fn(batch):
    max_size = max(x.size(0) for x, _ in batch)
    padded_x_batch = torch.stack([torch.cat((x, torch.zeros(max_size - x.size(0), x.size(1))), dim=0) for x, _ in batch])
    y_batch = torch.tensor([y for _, y in batch])
    return padded_x_batch, y_batch


def count_pos_neg_samples(dataset):
    num_pos = sum(1 for _, y in dataset if y == 1)
    num_neg = sum(1 for _, y in dataset if y == 0)
    return num_pos, num_neg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='../../workspace/weight/rtd/CNN/best_model-50d.pth', help='Trained Model Weight')
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
    # data_dir = './data/jsons/all-no-response-relationship.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-no-trace-no-span_dependency-no-log_sequence.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace-no-dependency.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-trace.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship-no-log.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all-no-response-relationship.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/all.jsons'
    # data_dir = './workspace/multimodal/data/DeepTraLog/v4/0-5-11-79-82-86.jsons'

    # best_model_path = 'CNN-best_model-50d.pth'
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
            origin_graph_data = False

            # 原始7维图数据trace_bool=1代表normal
            for graph in f:
                teg = json.loads(graph)
                if len(teg['node_info'][0]) == 7:
                    origin_graph_data = True
                break

            for graph in f:
                if i % 5000 == 0:
                    print("graph loading:" + str(i))
                i += 1
                teg = json.loads(graph)
                x = torch.tensor(teg['node_info'], dtype=torch.float)
                if origin_graph_data:
                    y = torch.tensor(0 if teg['trace_bool'] else 1, dtype=torch.long).view(1)
                else:
                    y = torch.tensor(teg['trace_bool'], dtype=torch.long).view(1)
                jsonList.append((x, y))
    train_dataset, test_dataset = train_test_split(jsonList, test_size=test_ratio, random_state=1234)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=valid_ratio, random_state=1234)

    train_pos, train_neg = count_pos_neg_samples(train_dataset)
    val_pos, val_neg = count_pos_neg_samples(val_dataset)
    test_pos, test_neg = count_pos_neg_samples(test_dataset)
    print(f"Train set: Positive={train_pos}, Negative={train_neg}")
    print(f"Validation set: Positive={val_pos}, Negative={val_neg}")
    print(f"Test set: Positive={test_pos}, Negative={test_neg}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the maximum size for the first dimension
    max_size = max(x.size(0) for x, _ in jsonList)
    if args.mode == 'train':
        train_loader = DataLoader(CustomDataset(train_dataset, max_size), batch_size=32, shuffle=True,
                                  collate_fn=custom_collate_fn)
        val_loader = DataLoader(CustomDataset(val_dataset, max_size), batch_size=32, shuffle=False,
                                collate_fn=custom_collate_fn)
        test_loader = DataLoader(CustomDataset(test_dataset, max_size), batch_size=32, shuffle=False,
                                 collate_fn=custom_collate_fn)

        model = CNN(hidden_channels=hidden_channels).to(device)
        # Compute class weights based on the number of samples
        train_targets = [y for _, y in train_dataset]
        negative_count = sum(1 for y in train_targets if y == 0)
        positive_count = sum(1 for y in train_targets if y == 1)
        negative_weight = positive_count / (positive_count + negative_count)
        positive_weight = negative_count / (positive_count + negative_count)
        weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device)

        # Create a new weighted cross entropy loss with the computed weights
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        num_epochs = 50

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            model.eval()
            val_running_loss = 0.0
            y_true = []
            y_pred = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(-1))

                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    y_true.extend(labels.view(-1).cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            val_loss = val_running_loss / len(val_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print('Saved best model to {}'.format(best_model_path))

                test_loader = DataLoader(CustomDataset(test_dataset, max_size), batch_size=32, shuffle=False,
                                         collate_fn=custom_collate_fn)

                print('result on test set: ')
                model.eval()
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        y_true.extend(labels.view(-1).cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

                print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, '
                      f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')
            print('\n')

    elif args.mode == 'predict':
        model = CNN(hidden_channels).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loader = DataLoader(CustomDataset(test_dataset, max_size), batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.view(-1).cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(precision*100, recall*100, f1_score*100))
