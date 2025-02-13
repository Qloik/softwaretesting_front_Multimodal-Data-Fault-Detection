import json
import random

jsons_dir = 'all-no-response-relationship-no-trace-no-dependency.jsons'


def sort_jsons_by_node_info_length(jsons_data):
    sorted_jsons = sorted(jsons_data, key=lambda x: len(x['node_info']))
    return sorted_jsons


def split_jsons(sorted_jsons, output_dir):
    total_count = len(sorted_jsons)
    middle_index = total_count // 2

    # Extracting longest JSONs
    longest_jsons = sorted_jsons[-100:]
    write_jsons_to_file(longest_jsons, output_dir, 'info-longest.jsons')

    # Extracting shortest JSONs
    shortest_jsons = sorted_jsons[:100]
    write_jsons_to_file(shortest_jsons, output_dir, 'info-shortest.jsons')

    # Extracting JSONs with median lengths
    # median_jsons = sorted_jsons[middle_index-50:middle_index+50]#改随机
    median_jsons = random.sample(sorted_jsons, 100)
    write_jsons_to_file(median_jsons, output_dir, 'info-median.jsons')


def write_jsons_to_file(jsons, output_dir, filename):
    with open(f'{output_dir}/{filename}', 'w') as f:
        print(f'{output_dir}/{filename}')
        for json_data in jsons:
            json.dump(json_data, f)
            f.write('\n')


# 读取包含多个JSON数据的文件
with open(jsons_dir) as f:
    jsons_data = f.readlines()

parsed_jsons = []
for json_data in jsons_data:
    json_obj = json.loads(json_data)
    # print(len(json_obj['node_info']))
    parsed_jsons.append(json_obj)

# 根据 `node_info` 值的长度对 JSON 数据进行排序
sorted_jsons = sort_jsons_by_node_info_length(parsed_jsons)

# 指定输出目录
output_directory = '../../../workspace/data/rtd'

# 分割并输出 JSON 数据到不同文件
split_jsons(sorted_jsons, output_directory)
