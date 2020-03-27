import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron


class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.finalWeights = []
        self.weightsArray = []
    
    def computeOutput(self, pattern, weights):
        output = np.dot(pattern, weights)
        if output > 0:
            return 1
        else:
            return 0
    
    def changeWeights(self, currWeights, currPat, target, output):
        self.weightsArray.append(currWeights)
        diff = self.lr * (target - output)
        newWeights = []
        for i in range(currWeights.size):
            columnDiff = diff * currPat[i]
            newWeights.append((columnDiff + currWeights[i]))
        
        return np.array(newWeights)

    def fit(self, X, y, initial_weights=None, shuffle=False):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        #including bias
        patternSize = X.shape[1] + 1
        trainingSize = X.shape[0]
        currWeights = self.initialize_weights(patternSize) if not initial_weights else initial_weights    
        Fitted = False
        epoch = 0
        missesArray = []

        while(not Fitted and epoch < 500):
            Fitted = True
            for i in range(trainingSize):
                misses = 0
                #currPat = np.array([X[i][0], X[i][1], X[i][2], X[i][3], 1])
                #wiping currPat and create new pattern
                currPat = np.zeros(patternSize)
                for j in range(patternSize-1):
                    currPat[j] = X[i][j]

                #adding bias
                currPat[patternSize-1] = 1
                currOutput = self.computeOutput(currPat, currWeights)
                if currOutput != y[i]:
                    currWeights = self.changeWeights(currWeights, currPat, y[i], currOutput)
                    Fitted = False
                    misses += 1
            missesArray.append(misses)
            epoch +=1
            if shuffle == True:
                X, y = self._shuffle_data(X, y)
            #stop criteria if same accuracy for 2 epoch in a row, will break. if epoch > 10
            if epoch > 5:
                if missesArray[epoch - 1] == missesArray[epoch - 2] and missesArray[epoch - 2] == missesArray[epoch - 3]:
                    break
        print("num of epoch ", epoch)
        self.finalWeights = currWeights
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        outputVec = []
        weights = self.finalWeights
        size = X.shape[0]
        patternSize = X.shape[1] + 1
        for i in range(size):
            # pattern = np.array([X[i][0], X[i][1], 1])
            currPat = np.zeros(patternSize)
            for j in range(patternSize-1):
                currPat[j] = X[i][j]

                #adding bias
            currPat[patternSize-1] = 1
            output = self.computeOutput(currPat, weights)
            outputVec.append(output)

        return np.array(outputVec)
        

    def initialize_weights(self, size):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        weights = np.zeros(size)
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        size = X.shape[0]
        score = 0
        predicted = self.predict(X)
        for i in range(size):
            if predicted[i] == y[i]:
                score += 1

        #print("score is ", score/size)
        return score/size

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        size = X.shape[0]
        a = np.array(X)
        b = np.array(y)
        combined = np.append(a,b, axis=1)
        np.random.shuffle(combined)
        labelIndex = combined.shape[1] - 1
        newLabels = np.empty([size, 1])
        #grabbing all the labels.
        for i in range(size):
            newLabels[i] = combined[i][labelIndex]
        
        #deleting labels column
        newData = np.delete(combined, labelIndex, axis=1)  

        return newData, newLabels
    
    def split(self, X, y):
        splitIndex = round((X.shape[0]/10)*7)
        X, y = self._shuffle_data(X, y)
        X_train = X[0:splitIndex]
        X_test = X[splitIndex:]
        y_train = y[0:splitIndex]
        y_test = y[splitIndex:]

        return X_train, X_test, y_train, y_test



    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.finalWeights
    
    def get_all_weights(self):
        return self.weightsArray


if __name__ == '__main__':
    pc = PerceptronClassifier(0.01)
    #mat = Arff("linsep2nonorigin.arff",label_count=1)
    #mat = Arff("data_banknote_authentication.arff",label_count=1)
    mat = Arff("voter.arff", label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    
    # pc.fit(data, labels)
    # accuracy = pc.score(data, labels)
    X_train, X_test, y_train, y_test = pc.split(data, labels)
    
    # data = pd.read_csv("DataSet1.txt", sep=',').to_numpy()
    # labels = pd.read_csv("Labels1.txt").to_numpy()



    pc.fit(X_train, y_train, shuffle=True)
    accuracy1 = pc.score(X_train, y_train)
    accuracy = pc.score(X_test, y_test)
    print("Test Accuracy= [{:.2f}]".format(accuracy1))
    print("Final Accuracy = [{:.2f}]".format(accuracy))

    #print("Final Weights =",pc.get_weights())

    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    print("score of SKLearn model is ", clf.score(X_test, y_test))

    