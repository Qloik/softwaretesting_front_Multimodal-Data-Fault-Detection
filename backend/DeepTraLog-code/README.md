# DeepTraLog
DeepTraLog Re-implmentation

## Run

在运行程序之前，首先生成span和log模板的嵌入表示，在embedding.py这里：存放模板的文件地址、预训练语言模型GloVe的地址修改成你们自己电脑上的地址。然后run一遍embedding.py。

```
templates = pd.read_csv('/workspace/multimodal/data/DeepTraLog/GraphData/id_url+type.csv') # 存放模板的文件地址
temEmbed = Embedding(pretrainModel_dir='/workspace/plelog/datasets/glove.6B.300d.txt')# 指定预训练语言模型
```

整个项目入口在DeepTraLog.py，在运行整个项目之前现确保

```
options["data_dir"] = '/workspace/multimodal/data/DeepTraLog/GraphData/process0.jsons'  # graph dir
```

存放了graphs的process0.jsons等等。

然后，训练模型则在文件目录下：

```bash
python DeepTraLog.py train
```

测试模型则下文件目录下：

```
python DeepTraLog.py predict
```

