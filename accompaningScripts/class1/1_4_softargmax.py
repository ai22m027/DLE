from math import exp
import matplotlib.pyplot as plt

def softargmax (x:list[float]):
    x = [exp(xx) for xx in x]
    x = [xx/sum(x) for xx in x]
    return x

def plot_softargmax (x:list[float]):
    plt.title("Input (gray)="+str(x)+", softargmax output (black)")
    plt.ylabel("Value")
    plt.xlabel("Index")
    plt.xticks([i for i in range(len(x))])
    plt.bar(range(len(x)), x, color="gray")
    plt.bar([i+0.15 for i in range(len(x))], softargmax(x), color="black")
    plt.hlines(1,-0.5,len(x)-0.5, color="black", ls="dashed")
    plt.show()

#effect of scaling
plot_softargmax(x=[0.5,2.5,1.5])
plot_softargmax(x=[1,5,3])
plot_softargmax(x=[2,10,6])
