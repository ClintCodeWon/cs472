import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
import numpy.ma as ma
import math
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class Node():
    def __init__(self, count, num_of_classes, value):
        self.count = count
        self.num_of_classes = num_of_classes
        self.value = value
        # print(self.count)
        # print(self.num_of_classes)
        self.childrenArray = []
        self.mostLikey = 0
        self.chosenColumn = None


    def createChildren(self, data, labels, count):
        self.mostLikey = self.calculateMostLikely(labels)
        # print("most likely ", self.mostLikey)
    
        if self.checkPurity(labels) or data.shape[1] == 0 or data.shape[0] == 0 or count == []: # need to add condition where out of data 
            # print("data is now ", data.shape[1])
            self.chosenColumn = None
            return self
        else:
            entropyArray = self.calculateEntropy(data, count, labels)
            # print(entropyArray)
            minPos = entropyArray.index(min(entropyArray))
            self.chosenColumn = minPos
            print("chosen column in creating child ", self.chosenColumn)
            # print("min pos is ", minPos)
            # print(self.count[minPos])

            #removing attribute from count
            num_ofChildren = self.count[minPos]
            # print("children count is ", self.count)
            childDataArray = []
            childLabelArray = []
            combined = np.concatenate((data,labels), axis=1)

            #creatin data set and labels for each attribute 
            for i in range(num_ofChildren):
                filterCombined = combined[np.where(combined[:,minPos] == i)]
                childLabels = filterCombined[:, -1].reshape(-1, 1)
                childData = np.delete(filterCombined, -1, 1)
                childData = np.delete(childData, minPos, 1)
                childDataArray.append(childData)
                childLabelArray.append(childLabels)

            #removing the count
            del count[minPos]
                        
            #for every attribute in count self.count[minPos]
            #creating child for each attribute
            for j in range(num_ofChildren):
                # print("creatin child with count ", count)
                child = Node(count, self.num_of_classes, j)
                newChild = child.createChildren(childDataArray[j], childLabelArray[j], count)
                self.childrenArray.append(newChild)
            
            # print("size of child array ", len(self.childrenArray))

            return self
        

    
    def checkPurity(self, labels):
        uniue_classes = np.unique(labels)

        if len(uniue_classes) == 1:
            # print("pure")
            return True
        else:
            #print("not pure")
            return False
    
    def calculateEntropy(self, data, counts, labels):
        entropyArray = []
        combined = np.concatenate((data, labels), axis=1)
        total = data.shape[0]

        #columns data.shape[1]
        for i in range(data.shape[1]):
            # every attribute of each column counts[i]
            allAttributeEntropy = 0

            for j in range(counts[i]):
                attributeArr = combined[np.where(combined[:,i] == j)]
                attributeNum = attributeArr.shape[0]
                singularEntropy = 0

                #each classification self.num_of_classes
                for k in range(self.num_of_classes):
                    classificationArr = attributeArr[np.where(attributeArr[:, -1] == k)]
                    # print(classificationArr)
                    count = classificationArr.shape[0]
                    if attributeNum == 0:
                        prob = 0
                    else:
                        prob = count/attributeNum
                    # print("prob is ", prob)
                    if prob == 0:
                        singularEntropy += 0
                    else:
                        singularEntropy += prob * -np.log2(prob)
                        # print("entorpy single ", singularEntropy)
                
                allAttributeEntropy += (attributeNum/total) * singularEntropy
                # print("all atrr entropy ", allAttributeEntropy)

            entropyArray.append(allAttributeEntropy)


        return entropyArray

    def calculateMostLikely(self, labels):
        (values,count) = np.unique(labels,return_counts=True)
        ind=np.argmax(count)
        return values[ind]

    def getPred(self, row):
        curr = self
        while(curr.getChildrenArray() != []):
            column = curr.getChosenColumn()
            # print("chosen column is ", column)
            value = row[column]
            int_val = np.int_(value)
            # print("getting ", int_val)
            curr = curr.getChild(int_val)
            # print("curr is type ", type(curr))
        # print("ans is ", curr.mostLikey)
        return curr.mostLikey

    def getValue(self):
        return self.value

    def getChildrenArray(self):
        return self.childrenArray

    def getMostLikely(self):
        return self.mostLikey
    
    def getChosenColumn(self):
        return self.chosenColumn
    
    def getChild(self, childIndex):
        node =  self.childrenArray[childIndex]
        return node
    

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, counts=None):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset = 
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]
        """
        self.counts = counts
        self.root = None




    def fit(self, X, y, count):
        """ Fit the data; Make the Desicion tree
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """ 
        num_of_classes = counts[-1]
        counts.pop()
        self.root = Node(counts, num_of_classes, -1)
        self.root.createChildren(X, y, counts)
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
        size = X.shape[0]
        for i in range(size):
            # print("predicting for ", X[i])
            pred = self.root.getPred(X[i])
            outputVec.append(pred)

        return np.array(outputVec)


    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        size = X.shape[0]
        score = 0
        predicted = self.predict(X)
        for i in range(size):
            if predicted[i] == y[i]:
                score += 1

        #print("score is ", score/size)
        return score/size

if __name__ == "__main__":
    mat = Arff("car.arff",label_count=1)
    counts = [] ## this is so you know how many types for each column
    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    #print(counts)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)

    #removing nans
    median = np.median(data)
    roundedMedian = np.round_(median, 0)
    data[np.isnan(data)]= roundedMedian

    DTClass = DTClassifier(counts)
    DTClass.fit(data,labels, counts)
    acc = DTClass.score(data, labels)
    print("acc ", acc)

    clf = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(clf, data, labels, cv=10)
    print(scores)
    tree.export_graphviz(clf)

    # params = {}
    # params[counts] = counts
    # cv_results = cross_validate(DTClass, data, labels, cv=10, fit_params=params)
    # sorted(cv_results.keys())
    # print(cv_results['test_score'])

    # kf = KFold(n_splits=10)
    # for train_index, test_index in kf.split(data):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = data[train_index], data[test_index]
    #     y_train, y_test = labels[train_index], labels[test_index]
    #     DTClass = DTClassifier(counts.copy())
    #     DTClass.fit(X_train,y_train, counts.copy())
    #     acc = DTClass.score(X_test, y_test)
    #     print("acc ", acc)






    # data[np.isnan(data)]= 1
    #arr1 = data[np.where(data[:,1] == 0)]
    # print(data)
    # median = np.median(data)
    # print(np.round_(median, 0))

    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         if np.isnan(data[i][j]):
    #             med = median[j]
    #             data[i][j] = med
    #             print("replace")



    # combined = np.concatenate((data, labels), axis=1)
    # print(combined.shape)
    # newLabels = combined[:, -1].reshape(-1, 1)
    # print(newLabels)
    # newCombined = np.delete(combined, -1, 1)
    # newCombined = np.delete(newCombined, 0, 1)
    # print(newCombined.shape)
    