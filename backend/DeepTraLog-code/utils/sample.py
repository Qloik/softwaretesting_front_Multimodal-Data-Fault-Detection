"""
@Time ： 2023/3/7 15:52
@Auth ： Sek Hiunam
@File ：sample.py.py
@Desc ：
"""
import json


def sample(filenames,n_normal,n_abnormal):
    normals = []
    abnormals = []

    for filepath in filenames:
        normal = []
        abnormal = []
        jsonList = []
        with open(filepath) as f:
            for graph in f:
                teg = json.loads(graph)
                jsonList.append(teg)

            for data in jsonList:
                if data['trace_bool']:  # trace_bool==True means normal
                    normal.append(data)
                else:
                    abnormal.append(data)

            normals += normal
            abnormals += abnormals

    nor_total = len(normals)
    abnor_total = len(abnormals)



    pass


if __name__ == '__main__':
    jsons_num = 9
    data_dir = '/workspace/multimodal/data/DeepTraLog/GraphData/process{}.jsons'

    filenames = []
    for i in range(jsons_num):
        filename = data_dir.format(i)
        filenames.append(filename)
    sample(filenames)