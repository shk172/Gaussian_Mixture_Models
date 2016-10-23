import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
def gmmest(X, mu_init, sigmasq_init, wt_init, its):
#% Input
#THE PROGRAM TAKES THE NAME OF THE CSV FILES AS AN ARGUMENT, AND WILL
#ITERATE 20 TIMES UNLESS THE ITS VALUE IS CHANGED IN THE MAIN FUNCTION



#% - X : N 1-dimensional data points (a 1-by-N vector)
#% - mu_init : initial means of K Gaussian components
#% (a 1-by-K vector)
#% - sigmasq_init: initial variances of K Gaussian components
#% (a 1-by-K vector
#% - wt_init : initial weights of k Gaussian components
#% (a 1-by-K vector that sums to 1)
#% - its : number of iterations for the EM algorithm
#%
#% Output
#% - mu : means of Gaussian components (a 1-by-K vector)
#% - sigmasq : variances of Gaussian components (a 1-by-K vector)
#% - wt : weights of Gaussian components (a 1-by-K vector, sums
#% to 1)
#% - L : log likelihood


    wt = np.array(wt_init)
    mu = np.array(mu_init)
    sigmasq = np.array(sigmasq_init)
    responsibilityArray = np.zeros((len(mu),len(X)))
    LList = []
    for iterations in range(int(its)):
        for index, value in enumerate(X):
        #E-Step, finding the responsibilities for each data points
            rsumlist = []
            r = 0
            for i in range(len(mu_init)):
                rsumlist.append((wt[i] * norm.pdf(value,mu[i],math.sqrt(sigmasq[i]))))
            rsum = sum(rsumlist)
            for j in range(len(mu_init)):
                r = (wt[j] * norm.pdf(value,mu[j],math.sqrt(sigmasq[j])))/rsum
                responsibilityArray[j][index] = r 
        #M-Step, finding the best fit parameters
                
        for w in range(len(wt_init)):
            #New Weight
            wt[w] = sum(responsibilityArray[w])/len(X)
        for m in range(len(mu_init)):
            #New Mu
            musumlist = []
            for data in range(len(X)):
                musumlist.append(((responsibilityArray[m][data])*X[data]))
            muSum = sum(musumlist)
            newMu = (muSum/sum(responsibilityArray[m]))
            mu[m] = (newMu)
        for s in range(len(sigmasq_init)):
            #New Sigma Square
            sigmaSumList = []
            for data2 in range(len(X)):
                sigmaSumList.append(((responsibilityArray[s][data2])*(math.pow((X[data2] - mu[s]),2))))
            sigmaSum = sum(sigmaSumList)
            newSigma = (sigmaSum/sum(responsibilityArray[s]))
            sigmasq[s] = (newSigma)
        likelihoodLogList = []
        for log in X:
            likelihoodList = []
            for log2 in range(len(mu_init)):
                likelihoodList.append(wt[log2] * norm.pdf(log,mu[log2],math.sqrt(sigmasq[log2])))
            likelihoodLogList.append(math.log(sum(likelihoodList)))
        L = sum(likelihoodLogList)
        LList.append(L)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood for Each Iteration')
    plt.plot(range(len(LList)),LList, '-r')
    plt.show()
    return mu, sigmasq, wt, L
    
    
if __name__ == "__main__":
    datafile = sys.argv[1]
    csvfile = open(datafile, 'rb')
    dat = csv.reader(csvfile)
    X = []
    Y = []
    wt_init = [0.6,0.4]
    mu_init = [10.0,30.0]
    sigmasq_init = [5.0,4.0]
    
    wt_init2 = [0.2,0.5,0.3]
    mu_init2 = [-24.0,-9.0,50.0]
    sigmasq_init2 = [2.0,4.0,6.0]
    dat.next()
    for row in dat:
        X.append(float(row[0]))
        Y.append(float(row[1]))
    
    X_test = np.array(X)
    Y_test = np.array(Y)
    
    class1 = X_test[np.nonzero(Y_test ==1)[0]]
    class2 = X_test[np.nonzero(Y_test ==2)[0]]
    its = 20
    mu1, sigmasq1, wt1, L1 = gmmest(class1,mu_init,sigmasq_init,wt_init,its)
    mu2, sigmasq2, wt2, L2 = gmmest(class2,mu_init2,sigmasq_init2,wt_init2,its)
    print 'The Log-Likelihood of class 1 is: ', L1
    print 'For Class 1 - ', 'Mu: ', mu1, ' Sigmasq: ', sigmasq1, ' Weights: ',wt1
    print 'The Log-Likelihood of class 1 is: ', L2
    print 'For Class 2 - ', 'Mu: ', mu2, ' Sigmasq: ', sigmasq2, ' Weights: ',wt2