import os
import re
import subprocess

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tasks

app = Flask(__name__)
CORS(app)  # 允许所有跨域访问

log_file = open("./workspace/log/console_output.txt", "w")

# 路由：接收数据并处理
@app.route('/process', methods=['POST'])
def process_data():
    # 获取POST请求中的原始数据
    form_data = request.get_json()
    print(form_data)

    data = form_data['data']
    model = form_data['model']
    train_data = form_data['train_data']
    print(model)

    pth_dir=get_pth_path(train_data)
    data_dir=get_data_path(data)
    if model == 'DTL':
        result = DTL_predict(pth_dir, data_dir)
    elif model == 'CNN':
        result = CNN_predict(pth_dir,data_dir)
    elif model == 'Classify':
        result = Classify_predict(pth_dir, data_dir)
    elif model == 'PU':
        result = PU_predict(pth_dir, data_dir)
    else :
        result = 0
    return jsonify(result)

@app.route('/process_mul', methods=['POST'])
def process_mul_data():
    # 获取POST请求中的原始数据
    form_data = request.get_json()  # 将表单数据转换为字典
    print(form_data)

    results = []  # 存储每个组合的处理结果

    for item in form_data:
        model = item['model']
        data = item['data']
        train_data = item['train_data']

        pth_dir = get_pth_path(train_data)
        data_dir = get_data_path(data)

        if model == 'DTL':
            result = DTL_predict(pth_dir, data_dir)
        elif model == 'CNN':
            result = CNN_predict(pth_dir, data_dir)
        elif model == 'Classify':
            result = Classify_predict(pth_dir, data_dir)
        elif model == 'PU':
            result = PU_predict(pth_dir, data_dir)
        else:
            result = 0

        results.append(result)

    return jsonify(results)


def get_log():
    global log_file
    log_file_path = './workspace/log/console_output.txt'  # 替换为实际的日志文件路径

    start_pattern = r'Saving test abnormal results'
    data_pattern1 = r'TP:\s+(\d+), TN:\s+(\d+), FP:\s+(\d+), FN:\s+(\d+)\n'
    data_pattern2 = r'Precision:\s+([\d.]+)%, Recall:\s+([\d.]+)%, F1-measure:\s+([\d.]+)%'
    data = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            print(line)
            match1 = re.search(data_pattern1, line)
            match2 = re.search(data_pattern2, line)
            if match1:
                data['TP'] = int(match1.group(1))
                data['TN'] = int(match1.group(2))
                data['FP'] = int(match1.group(3))
                data['FN'] = int(match1.group(4))
            if match2:
                data['P'] = float(match2.group(1))
                data['R'] = float(match2.group(2))
                data['F1'] = float(match2.group(3))
                break
    log_file = open("./workspace/log/console_output.txt", "w")

    print(data)
    return data


def DTL_predict(weight, data):
    # 调用预测脚本使用相应权重和数据进行预测
    process_cmd = ['python', './model/DeepTraLog-code/DeepTraLog.py', '--weight', weight+'DTL/', '--data', data, 'predict']
    process = subprocess.Popen(process_cmd,stdout=log_file, stderr=subprocess.STDOUT)
    process.wait()  # 等待进程执行完毕

    # 在预测完成后获取recall等结果
    log_data = get_log()  # 假设有一个获取recall结果的函数

    # 组织结果并返回
    result = log_data

    return result

def CNN_predict(weight, data):
    # 调用预测脚本使用相应权重和数据进行预测
    process_cmd = ['python', './model/DeepTraLog-code/NewNet-CNN.py', '--weight', weight+'CNN/best_model-50d.pth', '--data', data, 'predict']
    process = subprocess.Popen(process_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    process.wait()  # 等待进程执行完毕

    log_file.close()  # 关闭文件

    # 在预测完成后获取recall等结果
    log_data = get_log()  # 假设有一个获取recall结果的函数

    # 组织结果并返回
    result = log_data

    return result

def Classify_predict(weight, data):
    # 调用预测脚本使用相应权重和数据进行预测
    process_cmd = ['python', './model/DeepTraLog-code/error_classify.py', '--weight', weight+'classify/best_model-50d.pth', '--data', data, 'predict']
    process = subprocess.Popen(process_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    process.wait()  # 等待进程执行完毕

    log_file.close()  # 关闭文件

    # 在预测完成后获取recall等结果
    log_data = get_log()  # 假设有一个获取recall结果的函数

    # 组织结果并返回
    result = log_data

    return result

def PU_predict(weight, data):
    # 调用预测脚本使用相应权重和数据进行预测
    process_cmd = ['python', './model/DeepTraLog-code/PU-learning-GGNN.py', '--weight', weight+'PU/', '--data', data, 'predict']
    process = subprocess.Popen(process_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    process.wait()  # 等待进程执行完毕

    log_file.close()  # 关闭文件

    # 在预测完成后获取recall等结果
    log_data = get_log()  # 假设有一个获取recall结果的函数

    # 组织结果并返回
    result = log_data

    return result


def get_pth_path(file_name):
    mapping = {
        'r': './workspace/weight/r/',
        'rt':'./workspace/weight/rt/',
        'rtd':'./workspace/weight/rtd/',
        # Add more mappings as needed
    }

    if file_name in mapping:
        return mapping[file_name]

    # Handle the case when the file name is not found in the mapping
    raise ValueError(f"File name '{file_name}' not found in the mapping.")


def get_data_path(file_name):
    mapping = {
        'r-s': './workspace/data/r/info-shortest.jsons',
        'r-m': './workspace/data/r/info-median.jsons',
        'r-l': './workspace/data/r/info-longest.jsons',
        'rt-s': './workspace/data/rt/info-shortest.jsons',
        'rt-m': './workspace/data/rt/info-median.jsons',
        'rt-l': './workspace/data/rt/info-longest.jsons',
        'rtd-s': './workspace/data/rtd/info-shortest.jsons',
        'rtd-m': './workspace/data/rtd/info-median.jsons',
        'rtd-l': './workspace/data/rtd/info-longest.jsons',
        # Add more mappings as needed
    }

    if file_name in mapping:
        return mapping[file_name]

    # Handle the case when the file name is not found in the mapping
    raise ValueError(f"File name '{file_name}' not found in the mapping.")


if __name__ == '__main__':
    app.run(port=5001)
