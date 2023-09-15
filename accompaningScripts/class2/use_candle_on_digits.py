import sys
# sys.path.append("/home/sharwin/Documents/course_deep_learning/python_examples/demonstrations_during_course/2/2_5_candle/")

#import original data
from sklearn.datasets import load_digits
X = [list(x) for x in load_digits(return_X_y=True)[0]]
y = list(load_digits(return_X_y=True)[1])
Y = [[0]*yy + [1] + [0]*(10-yy-1) for yy in y]

from random import sample
test_indices = sample(range(len(X)), int(0.3*len(X)))

X_train = [X[i] for i in range(len(X)) if i not in test_indices]
Y_train = [Y[i] for i in range(len(X)) if i not in test_indices]

X_test = [X[i] for i in range(len(X)) if i in test_indices]
Y_test = [Y[i] for i in range(len(X)) if i in test_indices]

def plot_pixelseq (pixelseq):
    from math import isqrt
    import matplotlib.pyplot as plt
    n = isqrt(len(pixelseq))
    array = [pixelseq[i*n:(i+1)*n] for i in range((n**2+n-1)//n)]
    plt.imshow(array, cmap="binary")
    plt.show()

#look at data
plot_pixelseq(X_train[0])
Y_train[0]

from candle import NeuralNet, Linear, Logistic, ReLU, SGD, SELoss

mynet = NeuralNet()
mynet.add_layer(Linear(64, 100))
mynet.add_layer(Logistic())
#mynet.add_layer(Linear(100, 100))
#mynet.add_layer(ReLU())
mynet.add_layer(Linear(100, 10))
mynet.add_layer(Logistic())
mynet.print()

mytraining = SGD(x_data=X_train, y_data=Y_train, net=mynet, loss=SELoss())
mytraining.train(num_epochs=200, lr_start=0.4)
mytraining.plot_loss()

#compute loss on test
SELoss().on_sample(Y_test, [mynet.forward(x) for x in X_test])

