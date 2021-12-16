# Mnist-Classifier
**Using Minst from TensorFlow to do classification**
- Visualization of Mnist Dataset, namely CV
- Indentify handwritten numbers
  - [x] Random Forest
   **随机森林算法思想:**
    - 将多个不同的决策树进行组合，利用这种组合降低单一决策树有可能带来的片面性和判断不准确性。
    具体来讲，随机森林是用随机的方式建立一个森林。随机森林是由很多的决策树组成，但每一棵决策树之间是没有关联的。在得到森林之后，当对一个新的样本进行判断或预测的时候，让森林中的每一棵决策树分别进行判断，看看这个样本应该属于哪一类（对于分类算法），然后看看哪一类被选择最多(*多数表决制，李航《统计学习方法》*），就预测这个样本为那一类。
  - [x] KNN
    - 
  - [x] Kmeans
  - [x] Single Gradient Descent Classifier
- Upcoming:
  - [ ] Neural Network
  - [ ] Convolutional Neural Network
