import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from arff import Arff


if __name__ == '__main__':
    xVal = [6, 6, 6, 8, 6]
    yVal = [93, 96, 95, 92, 95]
    plt.scatter(xVal[:], yVal[:], color = 'blue')
    # plt.scatter(xVal[4:], yVal[4:], color = 'red')
    # x = np.linspace(-2,2,100)
    # y = 0.8*x-1
    plt.title('Epoch vs Misclass')
    plt.plot()
    
