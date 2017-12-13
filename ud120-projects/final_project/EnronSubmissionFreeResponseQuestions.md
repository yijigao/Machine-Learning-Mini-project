## 安然提交开放式问题
1. 向我们总结此项目的目标以及机器学习对于实现此目标有何帮助。作为答案的部分，提供一些数据背景信息以及这些信息如何用于回答项目问题。你在获得数据时它们是否包含异常值，你是如何处理的？
* 项目目标：通过机器学习识别并提取有用特征，构建算法，通过公开的安然财务和邮件数据集，找出有欺诈嫌疑的安然雇员
* 异常值：有些异常值是由于报表造成，如"TOTAL"； 而有些则是数据本身缺失，如“NaN”；还有些则是正常数据
* 异常值处理：对于因报表和缺失造成的异常值，将其找出并去掉；而正常数据则将其保留，并需要加以关注
2. 你最终在你的POI标识符中使用了什么特征，你使用了什么筛选过程来挑选他们？你是否需要进行任何缩放？为什么？作为任务的一部分，你应该设计自己的特征，而非使用数据集中现成的--解释你尝试创建的特征及其基本原理。（你不一定要在最后的分析中使用它，而只是设计并测试它）。在你的特征选择步骤，如果你使用了算法（如决策树），请也给出所使用的特征重要性；如果你使用了自动特征选择函数（如SelectBest)，请在报告特征得分及你所选的参数的原因

3. 你最终选用了什么算法？你还尝试了其他什么算法？不同算法之间的模型性能有何差异？

4. 调整算法的参数是什么意思，如果你不这样做会发生什么？你是如何调整特定算法的参数的?（一些算法没有需要调整的参数，指明并简要解释对于你最终为选择的模型或需要调整的不同模型，例如决策树分类器，你会怎么做）。

5. 什么是验证？未执行情况下的典型错误是什么？你是如何验证你的分析的

6. 给出至少两个评估变量并说明每个的平均性能。解释对用简单的语言表明算法性能的度量的解读



```commandline
KNeighborsClassifier(algorithm='ball_tree', leaf_size=50, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=4, p=2,
           weights='distance')
	Accuracy: 0.80873	Precision: 0.46433	Recall: 0.33850	F1: 0.39156	F2: 0.35790
	Total predictions: 11000	True positives:  677	False positives:  781	False negatives: 1323	True negatives: 8219
```
```commandline
features_list = ["poi","bonus", "exercised_stock_options", "director_fees"]

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform')
	Accuracy: 0.88364	Precision: 0.65868	Recall: 0.38500	F1: 0.48596	F2: 0.41989
	Total predictions: 14000	True positives:  770	False positives:  399	False negatives: 1230	True negatives: 11601
Test finished in 0.834 s
```