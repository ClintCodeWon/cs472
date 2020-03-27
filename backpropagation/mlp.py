import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
from math import e
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=False):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.input_weights = []
        self.hidden_weights = []
        self.changed_hidden_weights = []
        self.changed_input_weights = []
        self.MSEArraypred = []
        self.MSEArraytrue = []
        self.MSELossArray = []
        self.testPred = []
        self.testTruth = []
        self.testMSE = []



    def fit(self, input_data, input_truths, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        X, X_test, y, y_test = train_test_split(input_data, input_truths, test_size=0.25, random_state=42)
        patternSize = X.shape[1]
        batchSize = X.shape[0]
        # print(inputWeights.shape)
        # print(hiddenWeights.shape)
        bias = np.array([1.0])
        self.input_weights , self.hidden_weights = self.initialize_random_weights(patternSize)
        self.changed_input_weights, self.changed_hidden_weights = self.init_zero_weights(patternSize)
        misses = 0
        correct = 0
        num_epoch = 0
        while(True and num_epoch < 50):
            for i in range(batchSize):
                currPat = np.zeros(patternSize)
                for j in range(patternSize):
                    currPat[j] = X[i][j]
                input_layer = np.concatenate((currPat, bias))
                #print("input layer ", input_layer)
                #print("input weights", self.input_weights)
                hidden_netArray = self.computeNet(self.input_weights, input_layer)
                #print("hidden net ", hidden_netArray)
                hidden_outputArray = self.computeOutput(hidden_netArray)
                #print("output hidden ", hidden_outputArray)
                hidden_layer = np.concatenate((hidden_outputArray, bias))
                #print("hidden layer ", hidden_layer)
                outputNet = self.computeNet(self.hidden_weights, hidden_layer)
                #print("output final ", outputNet)
                my_pred = self.computeOutput(outputNet)

                self.recordTestMSE(my_pred, y[i])
                
                if my_pred < .333333:
                    round_pred = 0
                elif my_pred < .6666666 and my_pred > .3333333:
                    round_pred = 1
                else:
                    round_pred = 2

                if round_pred != y[i]:
                    self.changeWeight(round_pred, y[i], hidden_layer, input_layer)
                    misses += 1
                else:
                    correct += 1

                # if(round(my_pred[0], 1) != y[i]/10):
                #     self.changeWeight(my_pred, y[i]/10, hidden_layer, input_layer)
                #     misses += 1
                # else:
                #     correct +=1

                #
                # if(round(my_pred[0], 0) != y[i]):
                #     self.changeWeight(my_pred, y[i], hidden_layer, input_layer)
                #     misses += 1
                # else:
                #     correct +=1
            self.calculateTestMSE()
            breakCondition = self.validation_set(X_test, y_test)
            if breakCondition:
                break

            if self._shuffle_data:
                X, y = self._shuffle_data(X, y)
            
            num_epoch += 1


        
        print("num of epoch ", num_epoch)
        print("num of misses ", misses)
        print("corrects ", correct)
        #print("MSE loss ", self.MSELossArray[-1])

        
        return self

    def validation_set(self, X, y):
        patternSize = X.shape[1]
        batchSize = X.shape[0]
        bias = np.array([1.0])
        self.MSEArraypred = []
        self.MSEArraytrue = []
        for i in range(batchSize):
            currPat = np.zeros(patternSize)
            for j in range(patternSize):
                    currPat[j] = X[i][j]
            input_layer = np.concatenate((currPat, bias))
            #print("input layer ", input_layer)
            #print("input weights", self.input_weights)
            hidden_netArray = self.computeNet(self.input_weights, input_layer)
            #print("hidden net ", hidden_netArray)
            hidden_outputArray = self.computeOutput(hidden_netArray)                
            hidden_layer = np.concatenate((hidden_outputArray, bias))
            #print("hidden layer ", hidden_layer)
            outputNet = self.computeNet(self.hidden_weights, hidden_layer)
            #print("output final ", outputNet)
            my_pred = self.computeOutput(outputNet)
            self.recordMSE(my_pred, y[i])
        
        return self.calculateMSE()

    def train_deterministic(self, X, y, epoch_limit):
        patternSize = X.shape[1]
        batchSize = X.shape[0]
        # print(inputWeights.shape)
        # print(hiddenWeights.shape)
        bias = np.array([1.0])
        self.input_weights , self.hidden_weights = self.init_zero_weights(patternSize)
        self.changed_input_weights, self.changed_hidden_weights = self.init_zero_weights(patternSize)
        misses = 0
        correct = 0
        for num_epoch in range(epoch_limit):
            for i in range(batchSize):
                currPat = np.zeros(patternSize)
                for j in range(patternSize):
                    currPat[j] = X[i][j]
                input_layer = np.concatenate((currPat, bias))
                #print("input layer ", input_layer)
                #print("input weights", self.input_weights)
                hidden_netArray = self.computeNet(self.input_weights, input_layer)
                #print("hidden net ", hidden_netArray)
                hidden_outputArray = self.computeOutput(hidden_netArray)
                #print("output hidden ", hidden_outputArray)
                hidden_layer = np.concatenate((hidden_outputArray, bias))
                #print("hidden layer ", hidden_layer)
                outputNet = self.computeNet(self.hidden_weights, hidden_layer)
                #print("output final ", outputNet)
                my_pred = self.computeOutput(outputNet)

                
                #two outputs. 
                if(round(my_pred[0], 1) != y[i]/10):
                    self.changeWeight(my_pred, y[i], hidden_layer, input_layer)
                    misses += 1
                else:
                    correct +=1
        
        print("num of epoch ", num_epoch)
        print("num of misses ", misses)
        print("corrects ", correct)
            


        return self
    
    def recordTestMSE(self, pred, true):
        self.testPred.append(pred)
        self.testTruth.append(true)

    def calculateTestMSE(self):
        loss = mean_squared_error(self.testTruth, self.testPred)
        self.testMSE.append(loss)

    def recordMSE(self, predict, true):
        self.MSEArraypred.append(predict)
        self.MSEArraytrue.append(true)
    
    def calculateMSE(self):
        loss = mean_squared_error(self.MSEArraytrue, self.MSEArraypred)
        print("MSE loss curr is ", loss)
        self.MSELossArray.append(loss)
        if len(self.MSELossArray) > 15:
            if self.MSELossArray[-1] >= self.MSELossArray[-2]:
                return True
            else:
                return False
        
        return False

    def computeNet(self, weights, outputs):
        return np.matmul(weights, outputs)
    def computeOutput(self, net):
        outputArray = np.zeros(net.size)
        for i in range(net.size):
            outputArray[i] = 1/(1 + e**(-net[i]))

        return outputArray
    def changeWeight(self, pred, truth, outputArray, input_layer):
        backpropHidden = self.changeOutlayerWeight(pred, truth, outputArray)
        backpropInput = self.changeHiddenlayerWeight(pred, truth, input_layer, outputArray)
        #print("backpropHidden ", backpropHidden)
        #print("backpropInput", backpropInput)
        self.changed_input_weights = backpropInput
        self.changed_hidden_weights = backpropHidden
        self.hidden_weights = self.hidden_weights + backpropHidden
        self.input_weights = self.input_weights + backpropInput
        #print("new input weights ", self.input_weights)
        #print("new hidden weights ", self.hidden_weights)

    def changeHiddenlayerWeight(self, pred, truth, input_layer, hidden_layer):
        #print("pred ", pred)
        #print("truth ", truth)
        #print("Prev change wieghts ", self.changed_input_weights)
        # print(input_layer.shape)
        # print(hidden_layer.shape)
        #print(self.hidden_weights)
        changeWeightsArray = np.zeros((hidden_layer.shape[0] - 1, input_layer.shape[0]))
        for i in range(self.hidden_layer_widths):
            #print("hidden i ", hidden_layer[i])
            for j in range(input_layer.shape[0]):
                #print("input j ", input_layer[j])
                delta = (self.lr * input_layer[j] * (hidden_layer[i] * (1- hidden_layer[i]) * (truth - pred) * (pred * (1 - pred))) * self.hidden_weights[0][i]) + (self.momentum * self.changed_input_weights[i][j])
                #print("delta ", delta)
                changeWeightsArray[i][j] = delta

        return changeWeightsArray

    def changeOutlayerWeight(self, pred, truth, outputArray):
        # print("pred ", pred)
        # print("truth ", truth)
        #print("hidden Layer out ", outputArray)
        #print("Prev change wieghts ", self.changed_hidden_weights)
        size = self.hidden_layer_widths + 1
        changeWeightsArray = np.zeros((1, size))
        for i in range(size):
            delta = ((self.lr) * (outputArray[i]) * (truth - pred) * (pred * (1 - pred))) + (self.momentum * self.changed_hidden_weights[0][i])
            #print("delta ", delta)
            changeWeightsArray[0][i] = delta
        #print("hidden changeWeightsArray ", changeWeightsArray)
        return changeWeightsArray

        
    def initialize_random_weights(self, patternSize):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        num_hidden_nodes = self.hidden_layer_widths
        num_input = patternSize + 1
        inputWeight = np.random.rand(num_hidden_nodes, num_input)
        inputWeight = inputWeight - inputWeight.mean()
        inputWeight = inputWeight / inputWeight.max()
        hiddenWeights = np.random.rand(1, num_hidden_nodes+1)
        hiddenWeights = hiddenWeights- hiddenWeights.mean()
        hiddenWeights = hiddenWeights/ hiddenWeights.max()
        
        return inputWeight, hiddenWeights
    
    def init_zero_weights(self, patternSize):
        num_hidden_nodes = self.hidden_layer_widths
        num_input = patternSize + 1
        InputWeight = np.zeros((num_hidden_nodes, num_input))
        HiddenWeights = np.zeros((1, num_hidden_nodes+1))

        return InputWeight, HiddenWeights
       

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
            if predicted[i] < .333333:
                round_pred = 0
            elif predicted[i] < .6666666 and predicted[i] > .3333333:
                round_pred = 1
            else:
                round_pred = 2
            
            if round_pred == y[i]:
                score += 1

            # for 10 outputs
            # if(round(predicted[i], 1) == y[i]/10):
            #     score += 1

            #for two outputs
            # if(round(predicted[i], 0) == y[i]):
            #     score += 1

        #print("score is ", score/size)
        return score/size
        

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        outputVec = []
        size = X.shape[0]
        patternSize = X.shape[1]
        bias = np.array([1.0])

        for i in range(size):
            currPat = np.zeros(patternSize + 1)
            for j in range(patternSize):
                currPat[j] = X[i][j]
            currPat[patternSize] = 1
            #print("curr Pat ", currPat)
            # print("input weights", self.input_weights)
            hidden_netArray = self.computeNet(self.input_weights, currPat)
            # print("hidden net ", hidden_netArray)
            hidden_outputArray = self.computeOutput(hidden_netArray)
            # print("output hidden ", hidden_outputArray)
            hidden_layer = np.concatenate((hidden_outputArray, bias))
            # print("hidden layer ", hidden_layer)
            outputNet = self.computeNet(self.hidden_weights, hidden_layer)
            #print("output final ", outputNet)
            my_pred = self.computeOutput(outputNet)
            outputVec.append(my_pred[0])

        #print("outputVec ", outputVec)
        return outputVec

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

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.input_weights, self.hidden_weights

    def get_MSELoss(self):
        #print("MSE loss Array ", self.MSELossArray)
        return self.MSELossArray, self.testMSE



if __name__ == "__main__":
    bp = MLPClassifier(8, 0.4)
    #mat = Arff("linsep2nonorigin.arff",label_count=1)
    #mat = Arff("data_banknote_authentication.arff",label_count=1)
    mat = Arff("iris.arff",label_count=1)
    #mat = Arff("vowel.arff",label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    #bp.train_deterministic(data, labels, 10)
    bp.fit(data, labels)
    accuracy = bp.score(data, labels)
    print("accuracy ", accuracy)
    mseLoss, testLoss = bp.get_MSELoss()
    print("testLoss ", testLoss[-1])
    print("MSE less ", mseLoss[-1])
    #pyplot.plot(mseLoss, testLoss)
    #pyplot.show()
    #print("weights", bp.get_weights())
   
