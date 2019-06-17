# Handwritten-digit-recognition-model-with-keras


1)Increasing the number of layers might increase the accuracy. But if we increase the number of hidden layers much more than the sufficient number of layers the accuracy of test set will decrease. Increasing the number of hidden layers means increasing the complexity of the hypothesis function , this will cause the network to overfit to the training set , that is it will learn the training data ,but it won't be able to generalize to the new unseen data. The model hasn't learnt the trend instead it memorizes the training set data. So, the accuracy of the training will be greater and that of the test set will be less.

2)For the same number of epochs , overfitting starts to occur earlier for a model having more number of hidden units than that having comparatively lower number of them.there for an optimum number of hidden neurons should be considered.

3)Batch size denotes the number of training examples using in one iteration.Using a large batchsize there is a degradation in the quality of the model,as measured by its ability to generalize to new examples.That is, model with large batch size may have low accuracy of test set.

3)We may have have an overfitted model if we train so much on the training data. Number of epochs is the number of complete passes through the training dataset. Large number of epochs indticates training data so much which may lead to overfitting. So, optimum number of epochs should be considered.
