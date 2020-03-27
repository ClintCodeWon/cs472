import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
from sklearn.preprocessing import normalize
import sklearn as sk
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,columntype=[],weight_type='inverse_distance', normalize = False, linear_regression=False, k=3, hasReal=False): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype
        self.weight_type = weight_type
        self.data = []
        self.labels = []
        self.normalize = normalize
        self.linear_regression = linear_regression
        self.k = k
        self.hasReal = False

    def score(self, X, y):
        result = 0
        size = X.shape[0]
        predicted = self.predict(X)

        if self.linear_regression:
            error = np.sum(np.square(np.subtract(y, predicted)))/size
            result = error
        else:
            score = 0
            for i in range(size):
                if predicted[i] == y[i]:
                    score += 1
        
            result = score/size

        return result

    
    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if self.normalize:
            normalData = sk.preprocessing.normalize(data, axis=0)
            self.data = normalData
            self.labels = labels
            print("data is normalized")
        else:
            self.data = data
            self.labels = labels

       

        
        
        

        return self

    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        outputArray = []
        size = data.shape[0]
        for i in range(size):
            predClass = self.getPred(data[i])
            outputArray.append(predClass)
        
        return outputArray

    def getDistance(self, row):
        distanceArray = []
        inverseDistanceArray = []

        for i in range(self.data.shape[0]):
        # for i in range(1):
            A = self.data[i]
            B = row
            distance = np.subtract(A, B)

            # print("distance is ", distance)
            if self.hasReal:
                for i in range(len(self.columntype)):
                    if self.columntype[i] == 'nominal':
                        if distance[i] != 0:
                            distance[i] = 1
            np.nan_to_num(distance, nan=0.0)
            euclid_distance = np.sqrt(np.sum(np.square(distance)))
            # print("euclid distance ", euclid_distance)
            distanceArray.append(euclid_distance)
            if euclid_distance == 0:
                inverseDistanceArray.append(0)
            else:
                inverseDistanceArray.append(1/(euclid_distance ** 2))

        return np.array(distanceArray), np.array(inverseDistanceArray)
        


    def getPred(self, row):
        finalPred = 0
        distanceArray, inverseDistanceArray = self.getDistance(row)
        # print(inverseDistanceArray, distanceArray)
        k = self.k
        partitionArray = np.argpartition(distanceArray, k)
        lowestValArray = distanceArray[partitionArray[:k]]
        
        if self.weight_type == 'inverse_distance':
            indexVal = np.argmax(inverseDistanceArray)
            classification = self.labels[indexVal]
            # print(classification, "inverse")
            finalPred = classification
        else:
            classificationArray = []
            lowestValInverse = []
            for i in range(k):
                indexVal = np.where(distanceArray == lowestValArray[i])
                classification = self.labels[indexVal][0]
                myInverse = inverseDistanceArray[indexVal][0]
                classificationArray.append(classification)
                # print("class append ", classification)
                lowestValInverse.append(myInverse)

            # classificationArray = np.array(classificationArray)
            # lowestValInverse = np.array(lowestValInverse)
            # print("knn class", classificationArray)
            # print("lowestval inverse", lowestValInverse)

            if self.linear_regression:
                # regression label * 1/distance^2
                A = np.matmul(classificationArray, lowestValInverse)
                # # print("matmul", A)
                B = np.sum(lowestValInverse)
                # # print("sum of lowest", B)
                finalPred = A/B
                # print(finalPred, "linear regression pred")
                
                
                
            else:
                print("class array", classificationArray[0])
                counts = np.bincount(classificationArray)
                finalPred = np.argmax(counts)

        # print(finalPred, "final pred")
        return finalPred


if __name__ == "__main__":
    # mat = Arff("magic_telescope_train.arff",label_count=1)
    # mat2 = Arff("magic_telescope_test.arff",label_count=1)
    # mat = Arff("diabetes.arff",label_count=1)
    # mat2 = Arff("diabetes_test.arff",label_count=1)
    mat = Arff("seismic-bumps_train.arff",label_count=1)
    mat2 = Arff("seismic-bumps_test.arff",label_count=1)
    # mat = Arff("house_train.arff",label_count=1)
    # mat2 = Arff("house_test.arff",label_count=1)
    # mat = Arff("credit.arff",label_count=1)
    raw_data = mat.data
    h,w = raw_data.shape
    train_data = raw_data[:,:-1]
    train_labels = raw_data[:,-1]

    raw_data2 = mat2.data
    h2,w2 = raw_data2.shape
    test_data = raw_data2[:,:-1]
    test_labels = raw_data2[:,-1]

    # neigh = KNeighborsClassifier(n_neighbors=15)
    # neigh = KNeighborsRegressor(n_neighbors=3)
    # neigh.fit(train_data, train_labels)
    # print(neigh.score(test_data, test_labels))



    # start = timeit.default_timer()
    
    # print("number 5")
    # X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)

    # KNN = KNNClassifier(columntype = mat.attr_types,
    #                     weight_type='invese_distanc', 
    #                     linear_regression=False, 
    #                     normalize=False, 
    #                     k=1)
    # KNN.fit(X_train,y_train)
    # # normal_data = sk.preprocessing.normalize(train_data, axis=0)
    # score = KNN.score(X_test,y_test)
    # print("1", score)

    KNN = KNNClassifier(columntype ='classification',
                        weight_type='invese_distanc', 
                        linear_regression=False, 
                        normalize=False, 
                        k=3)
    KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    score = KNN.score(test_data,test_labels)
    print("distance weighting normal training normal test 3", score)


    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distance', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=5)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("distance weighting normal training normal test 5", score)

    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distance', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=7)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("distance weighting normal training normal test 7", score)

    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distance', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=9)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("distance weighting normal training normal test 9", score)

    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distanc', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=11)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("not distance weighting normal training normal test 11", score)

    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distance', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=13)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("distance weighting normal training normal test 13", score)
   

    # KNN = KNNClassifier(columntype ='classification',
    #                     weight_type='invese_distance', 
    #                     linear_regression=True, 
    #                     normalize=True, 
    #                     k=15)
    # KNN.fit(train_data,train_labels)
    # normal_data = sk.preprocessing.normalize(test_data, axis=0)
    # score = KNN.score(normal_data,test_labels)
    # print("distance weighting normal training normal test 15", score)
    
    # stop = timeit.default_timer()
    # print('Time: ', stop - start) 








    # KNN.getPred([1.0, 5.0, 3.0, 4.5, 32.4, 12.4, 3.0, 8.9])
    # KNN.getPred([1.0, 5.0, 3.0, 4.5, 32.4, 12.4, 3.0, 8.9, 10, 5.6, 12.4, 16.7, 19.8])


    # A = np.array([4, 4, 4, 4])
    # B = np.array([1, 1, 1, 1])
    # C = np.sqrt(np.square(np.subtract(A, B)))
    # print(C)
    # C = C.T
    # print(C.shape)
    # A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
    # k = 3

    # idx = np.argpartition(A, k)
    # myindex = A[idx[:k]]
    # print(myindex[0])

