# Mnist-Classifier
**Using Minst from TensorFlow to do classification**
- Visualization of Mnist Dataset, namely CV
- Indentify handwritten numbers
![train_image_preview_100](https://user-images.githubusercontent.com/84648756/146300522-01a127da-518c-41da-b302-7f9a53096b05.png)


  - [x] Random Forest
   **随机森林算法思想:**
    - 将多个不同的决策树进行组合，利用这种组合降低单一决策树有可能带来的片面性和判断不准确性。
    具体来讲，随机森林是用随机的方式建立一个森林。随机森林是由很多的决策树组成，但每一棵决策树之间是没有关联的。在得到森林之后，当对一个新的样本进行判断或预测的时候，让森林中的每一棵决策树分别进行判断，看看这个样本应该属于哪一类（对于分类算法），然后看看哪一类被选择最多(*多数表决制，李航《统计学习方法》*），就预测这个样本为那一类。
----
  - [x] KNN
  **K近邻算法思想：**
    - 对未知类别属性的数据集中的每个点依次执行以下操作：
     1. 计算已知类别数据集中的离散的点与当前点之间的距离；
     2. 按照距离递增次序排序；
     3. 选取与当前点距离最小的 k 个点；
     4. 确定前 k 个点所在类别的出现频率；
     5. 返回前k个点出现频率最高的类别(*多数表决制，李航《统计学习方法》*）作为当前点的预测分类。

---
  - [x] Kmeans
   **K均值聚类算法思想：**
   - 在数据集中根据一定策略选择K个点作为每个簇的初始中心，然后观察剩余的数据，将数据划分到距离这K个点最近的簇中，也就是说将数据划分成K个簇完成一次划分，但形成的新簇并不一定是最好的划分，因此生成的新簇中，重新计算每个簇的中心点，然后在重新进行划分，直到每次划分的结果保持不变。（*多次迭代*）
   - 在实际应用中往往经过很多次迭代仍然达不到每次划分结果保持不变，甚至因为数据的关系，根本就达不到这个终止条件，实际应用中往往采用变通的方法设置一个最大迭代次数，当达到最大迭代次数时，终止计算。
    - 具体的算法步骤如下：
    1. 随机选择K个中心点（*随机初始化中心点，因此每次结果可能会有略微的区别*）
    2. 把每个数据点分配到离它最近的中心点；
    3. 重新计算每类中的点到该类中心点距离的平均值
    4. 分配每个数据到它最近的中心点；
    5. 重复步骤3和4，直到所有的观测值不再被分配或是达到最大的迭代次数（R把10次作为默认迭代次
数）。

---

  - [x] Single Gradient Descent Classifier
  **随机梯度下降分类算法思想：**
    1. 正向传递：数据传入模型，得到预测值，再计算出损失值
    2. 损失值：帮助我们判断现有模型的好坏，需要改进多少（*和梯度相关的模型都会涉及LossFunction*）
    3. 优化算法： 帮助我们从损失值出发，一步一步更新参数，完善模型（*区别于手动调参*）
    4. brute force：比如随机选择1000个值，依次作为某个参数的值，得到1000个损失值，选择其中那个让损失值最小的值，作为最优的参数值。因此，产生了随机梯度下降算法，基于损失值，去自动更新参数，且要大幅降低计算次数。
    
---
![360截图17860607298025](https://user-images.githubusercontent.com/84648756/146300493-9f0035a8-7da7-4ba2-8c58-262b1c154fad.png)
- Upcoming:
  - [ ] Neural Network
  - [ ] Convolutional Neural Network
  - [ ] Logistic Regression

