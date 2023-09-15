import sys
# sys.path.append("/home/sharwin/Documents/course_deep_learning/python_examples/demonstrations_during_course/2/2_5_candle/")

from sklearn.datasets import load_iris
iris_data = load_iris()

data = iris_data.data
data = [data[i] for i in range(len(data))]

targets = iris_data.target
targets = [targets[i] for i in range(len(data))]
targets = [[0]*i + [1] + [0]*(2-i) for i in targets]
#targets = [[1] if i==0 else [0] for i in targets]

from random import sample
test_indices = sample(range(len(data)), int(0.25*len(data)))

x_train = [data[i] for i in range(len(data)) if i not in test_indices]
y_train = [targets[i] for i in range(len(data)) if i not in test_indices]

x_test = [data[i] for i in range(len(data)) if i in test_indices]
y_test = [targets[i] for i in range(len(data)) if i in test_indices]

import numpy as np
class_balance = np.sum(y_train, 0) / len(x_train)


from candle import NeuralNet, Linear, Logistic, ReLU, SGD, SELoss

mynet = NeuralNet()
mynet.add_layer(Linear(4, 100))
mynet.add_layer(ReLU())
#mynet.add_layer(Linear(100, 100))
#mynet.add_layer(ReLU())
#mynet.add_layer(Linear(100, 100))
#mynet.add_layer(ReLU())
#mynet.add_layer(Linear(100, 100))
#mynet.add_layer(ReLU())
mynet.add_layer(Linear(100, 3))
mynet.add_layer(Logistic())
mynet.print()

mytraining = SGD(x_data=x_train, y_data=y_train, net=mynet, loss=SELoss())
mytraining.train(num_epochs=500, lr_start=0.2, gamma=1)
mytraining.plot_loss()

#analyze predictions on test data
predicted_test = [mynet.forward(eg) for eg in x_test]
choice_test = [np.argmax(pred) for pred in predicted_test]
choice_true = [np.argmax(truth) for truth in y_test]
#computing accuracy
accuracy = sum(map(lambda a,b: a==b, choice_test, choice_true)) / len(choice_test)

