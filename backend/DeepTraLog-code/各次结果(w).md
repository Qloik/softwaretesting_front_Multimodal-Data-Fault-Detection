**no-response-relationship**2

1、DeepTraLog

```
Total Traces:  34451 Total Anomaly:  20254
TP: 18771, TN: 13897, FP: 300, FN: 1483
Precision: 98.43%, Recall: 92.68%, F1-measure: 95.47%
```

2、GCN

```
Total Positive Samples: 4085
Total Negative Samples: 14163
TP: 3981, TN: 14162, FP: 1, FN: 104
Precision: 0.9997, Recall: 0.9745, F1-score: 0.9870
```

3、GCN-class

```
Confusion Matrix:
 [[ 13   0   0   0   0   0   0  10   0   0   0   0   0   0   0]
 [  0 404   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0 446   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 324   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 227   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0  80   0  25   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0 303   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0 372   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 437   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 181   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   1 168   0   0   0   0]
 [  0   0   0   0   0   0   0   0   1   0   0 352   0   0   0]
 [  0   0   0   0   0   0   0   0   1   0   0   0 181   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  66 299   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 160]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.57      0.72        23
           1       1.00      1.00      1.00       404
           2       1.00      1.00      1.00       446
           3       1.00      1.00      1.00       324
           4       1.00      1.00      1.00       227
           5       1.00      0.76      0.86       105
           6       1.00      1.00      1.00       303
           7       0.91      1.00      0.96       372
           8       1.00      1.00      1.00       437
           9       0.99      1.00      1.00       181
          10       1.00      0.99      1.00       169
          11       1.00      1.00      1.00       353
          12       0.73      0.99      0.84       182
          13       1.00      0.82      0.90       365
          14       1.00      1.00      1.00       160

    accuracy                           0.97      4051
   macro avg       0.98      0.94      0.95      4051
weighted avg       0.98      0.97      0.97      4051
```

4、CNN

```
Train set: Positive=12079, Negative=42662
Validation set: Positive=4090, Negative=14157
Test set: Positive=4084, Negative=14164
TP: 3966, TN: 13752, FP: 412, FN: 118, Precision: 0.9059, Recall: 0.9711, F1-score: 0.9374
```

5、GCN-2

```
Total Positive Samples: 4085
Total Negative Samples: 14163
TP: 3981, TN: 14162, FP: 1, FN: 104
Precision: 0.9997, Recall: 0.9745, F1-score: 0.9870
```

6、GCN-1

```
Total Positive Samples: 4085
Total Negative Samples: 14163
TP: 3979, TN: 14121, FP: 42, FN: 106
Precision: 0.9896, Recall: 0.9741, F1-score: 0.9817
```

no-response-relationship-no-trace**

1、DeepTraLog

```
Total Traces:  33333 Total Anomaly:  19352
TP: 17201, TN: 13957, FP: 24, FN: 2151
Precision: 99.86%, Recall: 88.88%, F1-measure: 94.05%
```
2、GCN

```
Total Positive Samples: 3922
Total Negative Samples: 13930
TP: 3826, TN: 13930, FP: 0, FN: 96
Precision: 1.0000, Recall: 0.9755, F1-score: 0.9876
```

3、GCN-class

```
Confusion Matrix:
 [[ 11   0   0   0   0   0   0  13   0   0   0   0   0   0   0]
 [  0 403   0   0   0   0   0   0   1   0   0   0   0   0   0]
 [  0   0 416   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 285   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 189   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0  68   0  12   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0 309   0   0   0   0   0   0   0   0]
 [  1   0   0   0   0   1   0 275   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 395   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 204   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 192   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 358   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0 200   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  63 314   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 161]]
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.46      0.61        24
           1       1.00      1.00      1.00       404
           2       1.00      1.00      1.00       416
           3       1.00      1.00      1.00       285
           4       1.00      1.00      1.00       189
           5       0.99      0.85      0.91        80
           6       1.00      1.00      1.00       309
           7       0.92      0.99      0.95       277
           8       1.00      1.00      1.00       395
           9       1.00      1.00      1.00       204
          10       1.00      1.00      1.00       192
          11       1.00      1.00      1.00       358
          12       0.76      1.00      0.86       200
          13       1.00      0.83      0.91       377
          14       1.00      1.00      1.00       161

    accuracy                           0.98      3871
   macro avg       0.97      0.94      0.95      3871
weighted avg       0.98      0.98      0.98      3871
```

4、CNN

```
Train set: Positive=11558, Negative=41995
Validation set: Positive=3880, Negative=13971
Test set: Positive=3913, Negative=13939
TP: 3817, TN: 13539, FP: 400, FN: 96, Precision: 0.9051, Recall: 0.9755, F1-score: 0.9390
```
5、GCN-2

```
Total Positive Samples: 3922
Total Negative Samples: 13930
TP: 3826, TN: 13930, FP: 0, FN: 96
Precision: 1.0000, Recall: 0.9755, F1-score: 0.9876
```
6、GCN-1

```
Total Positive Samples: 3922
Total Negative Samples: 13930
TP: 3826, TN: 13930, FP: 0, FN: 96
Precision: 1.0000, Recall: 0.9755, F1-score: 0.9876
```
能否结合数据集、代码复现

**PU-learning**

内容和格式见截图

| GGNN_epoch | A.no-response-relationship | B.no-response-relationship-no-trace | C.no-response-relationship-no-trace-no-dependency |
| ---------- | -------------------------- | ----------------------------------- | ---------------------------------------- |
| 2          | √                          | √                                   | √                                        |
| 4          | √                          | √                                   | √                                        |
| 6          | √                          | √                                   | √                                        |
| 8          | √                          | √                                   | √                                        |
| 10         | √                          | √                                   | √                                        |
| 12         | √                          | √                                   | √                                        |
| 14         | √                          | √                                   | √                                        |
| 16         | √                          | √                                   | √                                        |
| 18         | √                          | √                                   | √                                        |

1、B epoch=2

```
Epoch 2
train loss:0.0054316130605176136
val loss:0.05473681872496058
testing...
TP: 3775.0, TN: 11150.0, FP: 2832.0, FN: 102.0
Precision: 0.5714, Recall: 0.9737, F1-score: 0.7201

Epoch 29
validing...
TP: 4847.0, TN: 0.0, FP: 17476.0, FN: 0.0
Precision: 0.2171, Recall: 1.0000, F1-score: 0.3568
```

2、B epoch=4

```
Epoch 4
train loss:0.002082120430464552
val loss:0.021504786662362392
testing...
TP: 3761.0, TN: 12749.0, FP: 1233.0, FN: 116.0
Precision: 0.7531, Recall: 0.9701, F1-score: 0.8479

Epoch 21
validing...
TP: 4679.0, TN: 17470.0, FP: 6.0, FN: 168.0
Precision: 0.9987, Recall: 0.9653, F1-score: 0.9817
```

3、B epoch=6

```
Epoch 6
train loss:0.0007810792143249812
val loss:0.015611772679160238
testing...
TP: 3654.0, TN: 13225.0, FP: 757.0, FN: 223.0
Precision: 0.8284, Recall: 0.9425, F1-score: 0.8818

Epoch 29
validing...
TP: 4413.0, TN: 17466.0, FP: 10.0, FN: 434.0
Precision: 0.9977, Recall: 0.9105, F1-score: 0.9521
```

4、B epoch=8

```
Epoch 8
train loss:0.0007195407354651235
val loss:0.023607804976281125
testing...
TP: 3687.0, TN: 12167.0, FP: 1815.0, FN: 190.0
Precision: 0.6701, Recall: 0.9510, F1-score: 0.7862

Epoch 29
validing...
TP: 4630.0, TN: 17474.0, FP: 2.0, FN: 217.0
Precision: 0.9996, Recall: 0.9552, F1-score: 0.9769
```

5、B epoch=10

```
Epoch 10
train loss:0.0005320205197632793
val loss:0.01390595772938711
testing...
TP: 3688.0, TN: 13412.0, FP: 570.0, FN: 189.0
Precision: 0.8661, Recall: 0.9513, F1-score: 0.9067

Epoch 29
validing...
TP: 4390.0, TN: 17407.0, FP: 69.0, FN: 457.0
Precision: 0.9845, Recall: 0.9057, F1-score: 0.9435
```

6、B epoch=12

```
Epoch 12
train loss:0.0052064886401538574
val loss:0.023161135241048753
testing...
TP: 3721.0, TN: 12272.0, FP: 1710.0, FN: 156.0
Precision: 0.6851, Recall: 0.9598, F1-score: 0.7995

Epoch 29
validing...
TP: 4256.0, TN: 17467.0, FP: 9.0, FN: 591.0
Precision: 0.9979, Recall: 0.8781, F1-score: 0.9342
```

7、B epoch=14

```
Epoch 14
train loss:0.000628800594207617
val loss:0.012099448574444711
testing...
TP: 3096.0, TN: 13748.0, FP: 234.0, FN: 781.0
Precision: 0.9297, Recall: 0.7986, F1-score: 0.8592

Epoch 19
validing...
TP: 4278.0, TN: 17472.0, FP: 4.0, FN: 569.0
Precision: 0.9991, Recall: 0.8826, F1-score: 0.9372
```

8、B epoch=16

```
Epoch 16
train loss:0.02136518533173907
val loss:0.0896925741748738
testing...
TP: 3534.0, TN: 12429.0, FP: 1553.0, FN: 343.0
Precision: 0.6947, Recall: 0.9115, F1-score: 0.7885

Epoch 29
validing...
TP: 4414.0, TN: 17472.0, FP: 4.0, FN: 433.0
Precision: 0.9991, Recall: 0.9107, F1-score: 0.9528
```

9、B epoch=18

```
Epoch 18
train loss:0.0005027012042854341
val loss:0.012086309724141736
testing...
TP: 3475.0, TN: 13847.0, FP: 135.0, FN: 402.0
Precision: 0.9626, Recall: 0.8963, F1-score: 0.9283

Epoch 29
validing...
TP: 4313.0, TN: 17469.0, FP: 7.0, FN: 534.0
Precision: 0.9984, Recall: 0.8898, F1-score: 0.9410
```
10、A epoch=2

```
Epoch 2
train loss:0.004107279548293636
val loss:0.08554630593841409
testing...
TP: 3923.0, TN: 11557.0, FP: 2640.0, FN: 128.0
Precision: 0.5977, Recall: 0.9684, F1-score: 0.7392

Epoch 29
validing...
TP: 5059.0, TN: 941.0, FP: 16805.0, FN: 5.0
Precision: 0.2314, Recall: 0.9990, F1-score: 0.3757
```

11、A epoch=4

```
Epoch 4
train loss:0.0025376587688205306
val loss:0.06669097324950737
testing...
TP: 3922.0, TN: 12911.0, FP: 1286.0, FN: 129.0
Precision: 0.7531, Recall: 0.9682, F1-score: 0.8472

Epoch 29
validing...
TP: 4925.0, TN: 17353.0, FP: 393.0, FN: 139.0
Precision: 0.9261, Recall: 0.9726, F1-score: 0.9488
```

12、A epoch=6

```
Epoch 6
train loss:0.002446378831518814
val loss:0.0611628712616391
testing...
TP: 3905.0, TN: 13001.0, FP: 1196.0, FN: 146.0
Precision: 0.7655, Recall: 0.9640, F1-score: 0.8534

Epoch 29
validing...
TP: 4772.0, TN: 17735.0, FP: 11.0, FN: 292.0
Precision: 0.9977, Recall: 0.9423, F1-score: 0.9692
```

13、A epoch=8

```
Epoch 8
train loss:0.0020177762990733457
val loss:0.04386284583088782
testing...
TP: 3661.0, TN: 13645.0, FP: 552.0, FN: 390.0
Precision: 0.8690, Recall: 0.9037, F1-score: 0.8860

Epoch 29
validing...
TP: 4491.0, TN: 17659.0, FP: 87.0, FN: 573.0
Precision: 0.9810, Recall: 0.8868, F1-score: 0.9315
```

14、A epoch=10

```
Epoch 10
train loss:0.0009829982913854974
val loss:0.01812856723260772
testing...
TP: 3771.0, TN: 13401.0, FP: 796.0, FN: 280.0
Precision: 0.8257, Recall: 0.9309, F1-score: 0.8751

Epoch 29
validing...
TP: 4520.0, TN: 17729.0, FP: 17.0, FN: 544.0
Precision: 0.9963, Recall: 0.8926, F1-score: 0.9416
```

15、A epoch=12

```
Epoch 12
train loss:0.0011129687132779508
val loss:0.022964457964507846
testing...
TP: 3802.0, TN: 12584.0, FP: 1613.0, FN: 249.0
Precision: 0.7021, Recall: 0.9385, F1-score: 0.8033

Epoch 29
validing...
TP: 4931.0, TN: 17730.0, FP: 16.0, FN: 133.0
Precision: 0.9968, Recall: 0.9737, F1-score: 0.9851
testing...
TP: 3947.0, TN: 14187.0, FP: 10.0, FN: 104.0
Precision: 0.9975, Recall: 0.9743, F1-score: 0.9858
```

16、A epoch=14

```
Epoch 10
train loss:0.0012271366007225283
val loss:0.019965723353261883
testing...
TP: 3890.0, TN: 13875.0, FP: 322.0, FN: 161.0
Precision: 0.9236, Recall: 0.9603, F1-score: 0.9415
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4850.0, TN: 17543.0, FP: 203.0, FN: 214.0
Precision: 0.9598, Recall: 0.9577, F1-score: 0.9588
```
17、A epoch=16

```
Epoch 11
train loss:0.0006022453789511452
val loss:0.014007369130178615
testing...
TP: 3710.0, TN: 13836.0, FP: 361.0, FN: 341.0
Precision: 0.9113, Recall: 0.9158, F1-score: 0.9136
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt
Epoch 16
train loss:0.0007386880460649454
val loss:0.02532709785275631
testing...
TP: 3841.0, TN: 12457.0, FP: 1740.0, FN: 210.0
Precision: 0.6882, Recall: 0.9482, F1-score: 0.7975

Epoch 29
validing...
TP: 4528.0, TN: 17736.0, FP: 10.0, FN: 536.0
Precision: 0.9978, Recall: 0.8942, F1-score: 0.9431
```

18、A epoch=18

```
Epoch 7
train loss:0.0026930525331647272
val loss:0.05451393364551099
testing...
TP: 3819.0, TN: 13686.0, FP: 511.0, FN: 232.0
Precision: 0.8820, Recall: 0.9427, F1-score: 0.9113
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt
Epoch 18
train loss:0.0005234546147832708
val loss:0.011106014303728803
testing...
TP: 3375.0, TN: 14099.0, FP: 98.0, FN: 676.0
Precision: 0.9718, Recall: 0.8331, F1-score: 0.8971

Epoch 29
validing...
TP: 4630.0, TN: 17653.0, FP: 93.0, FN: 434.0
Precision: 0.9803, Recall: 0.9143, F1-score: 0.9462
```
19、C epoch=2

```
Epoch 2
train loss:0.0020146473169273295
val loss:0.03736686795759245
testing...
TP: 3765.0, TN: 11720.0, FP: 2261.0, FN: 106.0
Precision: 0.6248, Recall: 0.9726, F1-score: 0.7608

Epoch 29
validing...
TP: 4697.0, TN: 17382.0, FP: 95.0, FN: 141.0
Precision: 0.9802, Recall: 0.9709, F1-score: 0.9755
```

20、C epoch=4

```
Epoch 3
train loss:0.0013019038989856247
val loss:0.027133593756290087
testing...
TP: 3761.0, TN: 12622.0, FP: 1359.0, FN: 110.0
Precision: 0.7346, Recall: 0.9716, F1-score: 0.8366
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4689.0, TN: 17390.0, FP: 87.0, FN: 149.0
Precision: 0.9818, Recall: 0.9692, F1-score: 0.9755
testing...
TP: 3754.0, TN: 13916.0, FP: 65.0, FN: 117.0
Precision: 0.9830, Recall: 0.9698, F1-score: 0.9763
```

21、C epoch=6

```
Epoch 4
train loss:0.0008057563402448963
val loss:0.020426007101765172
testing...
TP: 3739.0, TN: 12984.0, FP: 997.0, FN: 132.0
Precision: 0.7895, Recall: 0.9659, F1-score: 0.8688
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4605.0, TN: 17449.0, FP: 28.0, FN: 233.0
Precision: 0.9940, Recall: 0.9518, F1-score: 0.9724
```

22、C epoch=8

```
Epoch 6
train loss:0.0005296875961992703
val loss:0.015566103281265965
testing...
TP: 3563.0, TN: 13651.0, FP: 330.0, FN: 308.0
Precision: 0.9152, Recall: 0.9204, F1-score: 0.9178
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4476.0, TN: 17466.0, FP: 11.0, FN: 362.0
Precision: 0.9975, Recall: 0.9252, F1-score: 0.9600
```

23、C epoch=10

```
Epoch 8
train loss:0.0006183994679512104
val loss:0.015765021459780076
testing...
TP: 3529.0, TN: 13721.0, FP: 260.0, FN: 342.0
Precision: 0.9314, Recall: 0.9117, F1-score: 0.9214
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4362.0, TN: 17469.0, FP: 8.0, FN: 476.0
Precision: 0.9982, Recall: 0.9016, F1-score: 0.9474
```

24、C epoch=12

```
Epoch 8
train loss:0.0006329591682960103
val loss:0.03438542328155182
testing...
TP: 3680.0, TN: 13549.0, FP: 432.0, FN: 191.0
Precision: 0.8949, Recall: 0.9507, F1-score: 0.9220
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4460.0, TN: 17464.0, FP: 13.0, FN: 378.0
Precision: 0.9971, Recall: 0.9219, F1-score: 0.9580
```

25、C epoch=14

```
Epoch 10
train loss:0.0004733044793952865
val loss:0.013167039315868339
testing...
TP: 3658.0, TN: 13899.0, FP: 82.0, FN: 213.0
Precision: 0.9781, Recall: 0.9450, F1-score: 0.9612
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4561.0, TN: 17461.0, FP: 16.0, FN: 277.0
Precision: 0.9965, Recall: 0.9427, F1-score: 0.9689
```

26、C epoch=16

```
Epoch 7
train loss:0.0007086706685981448
val loss:0.014820710333639677
testing...
TP: 3620.0, TN: 13691.0, FP: 290.0, FN: 251.0
Precision: 0.9258, Recall: 0.9352, F1-score: 0.9305
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4499.0, TN: 17468.0, FP: 9.0, FN: 339.0
Precision: 0.9980, Recall: 0.9299, F1-score: 0.9628
```

27、C epoch=18

```
Epoch 12
train loss:0.0004912141568759927
val loss:0.014427610414215045
testing...
TP: 3508.0, TN: 13893.0, FP: 88.0, FN: 363.0
Precision: 0.9755, Recall: 0.9062, F1-score: 0.9396
Save best center ./PU/best_center.pt
Save GGNN_1 model ./PU/best_center.pt

Epoch 29
validing...
TP: 4234.0, TN: 17462.0, FP: 15.0, FN: 604.0
Precision: 0.9965, Recall: 0.8752, F1-score: 0.9319
```