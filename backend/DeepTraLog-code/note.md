| 关系                 | 左                     | 右                      |
| -------------------- | ---------------------- | ----------------------- |
| Synchronous Request  | parent: client request | child: server request   |
| Synchronous Response | child: server response | parent: client response |
| Asynchronous Request | parent: producer       | child: consumer         |

对于**Synchronous Request**和**Asynchronous Request**，都是parent->child；**Synchronous Response**相反



- 父子结点的URL不一定相同

- 一个span可以作为多个span的父节点
  - 一个同步操作，可能是一个MongoDB操作和一个同步操作的父结点
    - 代码中，这种情况还可以处理，因为如果MongoDB是子节点，就按照MongDB的方式处理
    - 问题：如果同步操作是子节点，父节点的client request和client response是定义不了的
      - 目前不知道会不会出现这种情况



**特殊关系**

| ParentSpan URL的EventType | CurrentSpan URL的EventType |
| ------------------------- | -------------------------- |
| Client/Server             | Producer                   |
| Consumer                  | Client/Server              |
|                           |                            |



**关于SpanId与ParentSpan**

- 在Local中二者不同
- Entry中二者不同（ParentSpan可以取到-1）
- Exit中二者相同（ParentSpan可以取到-1）



**Entry, Exit, Local**

|               | Entry | Exit | Local |
| ------------- | ----- | ---- | ----- |
| extension_URL | N     | Y    | Y     |
| common URL    | Y     | Y    | N     |
|               |       |      |       |

- extension_URL

  ```python
  extension_list = ["MongoDB/FindOperation", "MongoDB/UpdateOperation","SpringAsync","Thread/travel.service.TravelServiceImpl$MyCallable/call","Thread/rebook.service.RebookServiceImpl$AsyncDrawBackMoney/call","Thread/rebook.service.RebookServiceImpl$AsyncUpdateOrder/call"]
  ```

  - "MongoDB/FindOperation", "MongoDB/UpdateOperation"只在Exit Span中出现
  - 其余只在Local Span中出现



**span与log模板的嵌入表示**

- 读取glove.6B.300d.txt生成tem2embed
  - tem2embed主要是记录4万个单词与对应300维向量的字典
- 读取id_url+type.csv获得templates
  - templates前1359种为log部分（对应了日志模板？）
  - 后面为trace部分，应该是所有span的对应的内容
    - trace部分相同temp的只会保留最后一条
- 将templates的前1350种log部分与trace部分，分别进行如下操作
  - 计算每个单词的TF score: 词w在事件e种出现的频率${\frac{L_{we}(e中w数量)}{L_e(e总单词数)}}$
  - 计算每个单词的IDF score: log(总事件数量/包含w的事件数量) $\log\frac{L}{L_w}$
  - 对于每个template转化为数字（每个单词向量\*TF\*IDF）
    - 代码直接将总事件数量用对应templates的数量代替
    - 代码中好像没有除以不同单词数量？



**load_graphs_from_jsons**

- tem2Emed采用主成分分析（100个主成分），将Tensor由(1513, 300)转为(1513, 100)，这是新的tem2Emd

- 对于每一条TEG数据

  ```python
  graph = Data(x=tem2Emed,                     edge_index=torch.tensor(data['edge_index']).t().contiguous(),                  edge_attr=torch.tensor(data['edge_attr']).reshape(-1, 1))
  # 第一个参数Node feature matrix with shape：[num_nodes, num_node_features]
  # 第二个参数 edge_index: [2, num_deges]
  # 第三个参数 edge_attr(edge feature matrix) [num_edges, num_edge_features]
  ```

  - 第一个参数为什们是tem2Emed?
    - 实际上每一条原始json都有node_info，应该用这个？



**train**

- 训练时，好像只用了一个网络？



**predict**

新的数据经过forward之后输出，若为负数（代码是小于0.5）则为异常



### 修改之处

1. 代码进行embedding中好像没有除以不同单词数量？
2. 主成分分析数量
3. hidded=7?，修改graph生成参数
4. 两个网络?
   - soft_attention中的线性层函数是否有误？
5. warm up epoch后再确定center初始值？文中是初次
6. num_layer增多？
7. soft_attention替换？
8. 损失函数是否需要修改？
   - 是否需要考虑anomaly数据的影响？



### 各次结果

- options["adam_weight_decay"] 默认为0.001
- nu默认为0.05
- options["warm_up_n_epochs"]默认为10
- 数据集默认为process0

**初始**（oringinal）

```
Total Traces:  4967 Total Anomaly:  872
TP: 873, TN: 0, FP: 4095, FN: 0
Precision: 17.57%, Recall: 100.00%, F1-measure: 29.89%
elapsed_time: 56.947227478027344
```

<img src=".\note-pic\1.png" style="zoom:60%;" />



**除以不同单词数量**

结果和之前完全一样。。。



**hidden改为7（使用json中数据）**

和之前完全一样



**改变soft attention，使其输入只为ggnn的输出（不包括ggnn的输入）**

```
Total Traces:  4967 Total Anomaly:  872
TP: 77, TN: 4019, FP: 76, FN: 796
Precision: 50.33%, Recall: 8.82%, F1-measure: 15.01%
```



**加上一个网络**

<img src=".\note-pic\2.png" style="zoom:60%;" />

threshold 0和0.5结果一样均如下

```
Total Traces:  4967 Total Anomaly:  872
TP: 160, TN: 4072, FP: 23, FN: 713
Precision: 87.43%, Recall: 18.33%, F1-measure: 30.30%
```



**num_layer改为4**

```
Total Traces:  4967 Total Anomaly:  872
TP: 182, TN: 4074, FP: 21, FN: 691
Precision: 89.66%, Recall: 20.85%, F1-measure: 33.83%
```



**加了tanh（num_layer=4）**

> 最后的输出加tanh

```
Total Traces:  4967 Total Anomaly:  872
TP: 75, TN: 4095, FP: 0, FN: 798
Precision: 100.00%, Recall: 8.59%, F1-measure: 15.82%
```



**无tanh（num_layer=3）**

```
Total Traces:  4967 Total Anomaly:  872
TP: 316, TN: 4017, FP: 78, FN: 557
Precision: 80.20%, Recall: 36.20%, F1-measure: 49.88%
```



**1**
$$
h_g=\tanh \left\{ 
\phi\left[\sum_{v \in V}f_i\left(h_v^{(T)}, x_v\right)\right] 
\odot 
\sum_{v \in V}\tanh \left[f_j\left(h_v^{(T)}, x_v\right)\right]
\right\}
$$


```
num_layer=3
Total Traces:  4967 Total Anomaly:  872
TP: 385, TN: 3947, FP: 148, FN: 488
Precision: 72.23%, Recall: 44.10%, F1-measure: 54.77%
第二次
Total Traces:  4967 Total Anomaly:  872
TP: 308, TN: 3938, FP: 157, FN: 565
Precision: 66.24%, Recall: 35.28%, F1-measure: 46.04%
```

**1-4**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=2 
训练不全
Total Traces:  4967 Total Anomaly:  872
TP: 873, TN: 0, FP: 4095, FN: 0
Precision: 17.57%, Recall: 100.00%, F1-measure: 29.89%

训练全了
Total Traces:  4967 Total Anomaly:  872
TP: 27, TN: 4094, FP: 1, FN: 846
Precision: 96.43%, Recall: 3.09%, F1-measure: 5.99%
```

**1-5**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["adam_weight_decay"] = 0.01
TP: 141, TN: 3856, FP: 239, FN: 732
Precision: 37.11%, Recall: 16.15%, F1-measure: 22.51%
elapsed_time: 76.58447265625
```

**1-6**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["adam_weight_decay"] = 0.005
TP: 222, TN: 4089, FP: 6, FN: 651
Precision: 97.37%, Recall: 25.43%, F1-measure: 40.33%
elapsed_time: 74.88221502304077
```

**1-7**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["adam_weight_decay"] = 0.003
Total Traces:  4967 Total Anomaly:  872
TP: 10, TN: 4089, FP: 6, FN: 863
Precision: 62.50%, Recall: 1.15%, F1-measure: 2.25%
```

**warmup-5**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 5
TP: 340, TN: 4003, FP: 92, FN: 533
Precision: 78.70%, Recall: 38.95%, F1-measure: 52.11%
elapsed_time: 79.04714918136597
注：这里第99个epoch模型还在更新，考虑增加epoch?
```

**warmup-3**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 3
Total Traces:  4967 Total Anomaly:  872
TP: 321, TN: 3996, FP: 99, FN: 552
Precision: 76.43%, Recall: 36.77%, F1-measure: 49.65%
```

**warmup-1**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 1
Total Traces:  4967 Total Anomaly:  872
TP: 432, TN: 3936, FP: 159, FN: 441
Precision: 73.10%, Recall: 49.48%, F1-measure: 59.02%
```

**warmup-1-train-ratio-0.8**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 1
train_ratio = 0.8
Total Traces:  4967 Total Anomaly:  872
TP: 410, TN: 3912, FP: 183, FN: 463
Precision: 69.14%, Recall: 46.96%, F1-measure: 55.93%
```

**warmup-1-all**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 1
所有数据
Total Traces:  39745 Total Anomaly:  7000
TP: 2135, TN: 32135, FP: 611, FN: 4866
Precision: 77.75%, Recall: 30.50%, F1-measure: 43.81%
```

**warmup-1-all-1**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
options["warm_up_n_epochs"] = 1
所有数据,训练集比例调整为0.8
29%
```



**warmup-1-1**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=2 
options["warm_up_n_epochs"] = 1
Total Traces:  4967 Total Anomaly:  872
TP: 151, TN: 4076, FP: 19, FN: 722
Precision: 88.82%, Recall: 17.30%, F1-measure: 28.95%
```

**warmup-1-2**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=4 
options["warm_up_n_epochs"] = 1
注：epoch94为最优模型
Total Traces:  4967 Total Anomaly:  872
TP: 383, TN: 3987, FP: 108, FN: 490
Precision: 78.00%, Recall: 43.87%, F1-measure: 56.16%
```

**warmup-1-2-alll**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=4 
options["warm_up_n_epochs"] = 1
所有数据
Total Traces:  39745 Total Anomaly:  7000
TP: 1338, TN: 32087, FP: 659, FN: 5663
Precision: 67.00%, Recall: 19.11%, F1-measure: 29.74%
```





**2**

```
x_ggnn_2 = self.ggnn(x,edge_index)
num_layer=3 
nu = 0.07(1全是0.05)
Total Traces:  4967 Total Anomaly:  872
TP: 178, TN: 3965, FP: 130, FN: 695
Precision: 57.79%, Recall: 20.39%, F1-measure: 30.14%
```



**1-1**

```
x_ggnn_2 = self.ggnn_2(x,edge_index)
num_layer = 2
Total Traces:  4967 Total Anomaly:  872
TP: 72, TN: 3936, FP: 159, FN: 801
Precision: 31.17%, Recall: 8.25%, F1-measure: 13.04%
```



**1-2**

```
x_ggnn_2 = self.ggnn_2(x,edge_index)
num_layer = 3
Total Traces:  4967 Total Anomaly:  872
TP: 293, TN: 3918, FP: 177, FN: 580
Precision: 62.34%, Recall: 33.56%, F1-measure: 43.63%
```



**1-2-1**

```
x_ggnn_2 = self.ggnn_2(x,edge_index)
num_layer = 3
options["warm_up_n_epochs"] = 1
Total Traces:  4967 Total Anomaly:  872
TP: 162, TN: 4063, FP: 32, FN: 711
Precision: 83.51%, Recall: 18.56%, F1-measure: 30.37%
```



**1-2-warmup-1-all**

```
x_ggnn_2 = self.ggnn_2(x,edge_index)
num_layer = 3
options["warm_up_n_epochs"] = 1
所有数据
Total Traces:  39745 Total Anomaly:  7000
TP: 1357, TN: 32089, FP: 657, FN: 5644
Precision: 67.38%, Recall: 19.38%, F1-measure: 30.11%
注：best model在epoch91取到
```





**1-3**

```
x_ggnn_2 = self.ggnn_2(x,edge_index)
num_layer = 4
Total Traces:  4967 Total Anomaly:  872
TP: 62, TN: 4093, FP: 2, FN: 811
Precision: 96.88%, Recall: 7.10%, F1-measure: 13.23%
```





**2**
$$
h_g=\tanh \left\{ 
\sum_{v \in V}
\left\{ 
\phi\left[f_i\left(h_v^{(T)}, x_v\right)\right] 
\odot 
\tanh \left[f_j\left(h_v^{(T)}, x_v\right)\right]
\right\}
\right\}
$$

- 遇到了莫名其妙的bug暂未成功



### Attention

https://zhuanlan.zhihu.com/p/91839581

<img src=".\note-pic\3.png" style="zoom:40%;" />

<img src=".\note-pic\4.png" style="zoom:40%;" />

<img src=".\note-pic\5.png" style="zoom:50%;" />

<img src=".\note-pic\6.png" style="zoom:40%;" />



### grapha construction

