import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
def gmmclassify(X, mu1, sigmasq1, wt1, mu2, sigmasq2,wt2, p1):
    classifiedData = np.zeros((1,len(X)))
#% Input
#THE PROGRAM TAKES THE NAME OF THE CSV FILES AS AN ARGUMENT    


#% - X : N 1-dimensional data points (a 1-by-N vector)
#% - mu1 : means of Gaussian components of the 1st class
#% (a 1-by-K1 vector)
#% - sigmasq1 : variances of Gaussian components of the 1st class
#% (a 1-by-K1 vector)
#% - wt1 : weights of Gaussian components of the 1st class
#% (a 1-by-K1 vector, sums to 1)
#% - mu2 : means of Gaussian components of the 2nd class
#% (a 1-by-K2 vector)
#% - sigmasq2 : variances of Gaussian components of the 2nd class
#% (a 1-by-K2 vector)
#% - wt2 : weights of Gaussian components of the 2nd class
#% (a 1-by-K2 vector, sums to 1)
#% - p1 : the prior probability of class 1.

    for index,value in enumerate(X):
        model1 = []
        model2 = []
        
        for i in range(len(mu1)):
            model1.append(p1 * norm.pdf(value,mu1[i],math.sqrt(sigmasq1[i])))
            model2.append((1-p1) * norm.pdf(value, mu2[i], math.sqrt(sigmasq2[i])))
        model1Probability = sum(model1)
        model2Probability = sum(model2)
        if model1Probability > model2Probability:
            classifiedData[0][index] = 1
        elif model2Probability > model1Probability:
            classifiedData[0][index] = 2
    return classifiedData
    
    
if __name__ == "__main__":
    datafile = sys.argv[1]
    csvfile = open(datafile, 'rb')
    dat = csv.reader(csvfile)
    
    X = []
    Y = []

    dat.next()
    for row in dat:
        X.append(float(row[0]))
        Y.append(float(row[1]))
    
    X_test = np.array(X)
    Y_test = np.array(Y)
    mu1 = np.array((9.7749,29.5825))
    sigmasq1 = np.array((21.9228,9.7838))
    wt1 = np.array((0.5977,0.4023))
    
    mu2 = np.array((-24.8228,-5.0602, 49.6244))
    sigmasq2 = np.array((7.9473,23.3227,100.0243))
    wt2 = np.array((0.2037,0.4988,0.2975))
    
    class1 = X_test[np.nonzero(Y_test ==1)[0]]
    p1 = float(len(class1))/float(len(X))

    classifiedData = gmmclassify(X, mu1, sigmasq1, wt1, mu2, sigmasq2,wt2, p1)
    error = 0
    class1Marker = []
    class2Marker = []
    markerYAxis1 = []
    markerYAxis2 = []
    for i,v in enumerate(classifiedData[0]):
        if v != Y[i]:
            error += 1
        if v == 1:
            class1Marker.append(X[i])
            markerYAxis1.append(-1)
        if v == 2:
            class2Marker.append(X[i])
            markerYAxis2.append(-1)
    accuracy = 1 - (float(error)/float(len(X)))
    class2 = X_test[np.nonzero(Y_test ==2)[0]]
    bins = 50 
    plt.hist(class1, bins, alpha = 0.5, label = 'Class 1', color = 'blue')
    plt.hist(class2, bins,alpha = 0.5, label = 'Class 2', color = 'green')
    plt.plot(class1Marker,markerYAxis1, 'b|', alpha = 0.5, markersize = 15)
    plt.plot(class2Marker,markerYAxis2,'g|', alpha = 0.5, markersize = 15)
    plt.legend(loc='upper right')
    plt.title('Histogram of observed data for each class')
    plt.show()
    print 'The Accuracy of this GMM is: ', accuracy