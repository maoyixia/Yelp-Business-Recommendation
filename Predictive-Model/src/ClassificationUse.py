##  This script builds and applies classification models in Python, using
## the data we preprocessed with DataLoading.py

## Imported packages need to have been installed on your computer -- see tutorial

import numpy as np
import scipy as sp
import sklearn
import os
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle
import csv
import sys


def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj

dataMat = pickleLoad('myDataMat')
target = pickleLoad('myTarget')
headers = pickleLoad('myHeaders')
valDict = pickleLoad('myValues') 
num_names = pickleLoad('numericalVariables')
cat_names = pickleLoad('categoricalVariables')

target = np.ravel(target)

## Three ways to partition a set into training and test files.
#1 use KFold, #2 do train/test percentage split, #3 use pre-provided test set, won't cover here

def splitFunc(target, optNum):
    
    #do k-fold cross val
    if optNum>1:
        return KFold(len(target), int(optNum), indices = False, shuffle = True)
    
    #do percent based train/test split
    elif optNum<1:
        return ShuffleSplit(n=len(target), n_iter=1, test_size = optNum, indices = False)
    else:
        print 'Error, do not set opt num to 1!'
        return 0

def supClassOptions(type):
    if type == 'Tree':
        return tree.DecisionTreeClassifier(min_samples_leaf = 200, max_depth=3)
    if type == 'NB':
        return GaussianNB()
    if type == 'Logistic':
        return LogisticRegression(penalty='l2',C=1.0)
    if type == 'SVM':
        return SVC(C=1.0, kernel='rbf', probability=True)
    else:
        print 'error, type unknown'
        return 0

#initializing list of lists for storing performance metrics
def metricInit():
    metricList = []
    for i in xrange(9):
        metricList.append([])
    return metricList

#can use scikit learn metrics for some of these
def metricFunc(targTrue, targPreds, targProbs, metricList):
    correctClass = accuracy_score(targTrue, targPreds)
    incorrectClass = 1-correctClass
    correctClassNum = accuracy_score(targTrue, targPreds, normalize=False)
    incorrectClassNum = targTrue.shape[0] - correctClassNum

    print targTrue
    print targProbs
    #sys.exit()
    AUC = roc_auc_score(targTrue, targProbs)
    
    #confusion matrix has row i => known to be in class i, column j => predicted to be in class j
    confusion = confusion_matrix(targTrue, targPreds)
    TP = confusion[1,1]
    FP = confusion[0,1]
    
    #need to calculate these specially since TP+FP might be = 0
    if TP+FP>0:
        precision = (TP*1.0)/(TP+FP)
        TPR = (confusion[1,1]*1.0)/(confusion[1,1]+confusion[1,0])
        FPR = (confusion[0,1]*1.0)/(confusion[0,0]+confusion[0,1])
    else:
        precision = 0
        TPR = 0
        FPR = 0

    #append each value to appropriate list
    metricList[0].append(correctClass)
    metricList[1].append(incorrectClass)
    metricList[2].append(correctClassNum)
    metricList[3].append(incorrectClassNum) 
    metricList[4].append(precision)
    metricList[5].append(AUC)
    metricList[6].append(TPR)
    metricList[7].append(FPR)
    metricList[8].append(confusion)

    return metricList

#takes the average of the different metrics across all folds (or one fold if doing % split)
def printMeanMetrics(classMetrics):
    means = []
    metricLabels = ['Percent Correctly Classified', 'Percent Incorrectly Classified', 'Number Correctly Classified', 'Number Incorrectly Classified', 'Precision', 'AUC', 'True Positive Rate', 'False Positive Rate', 'Confusion Matrix']
    
    #calculate means of metrics
    for i in xrange(len(classMetrics)-1):
        means.append(sum(classMetrics[i])/len(classMetrics[i]))
        print metricLabels[i]+' is: '+repr(means[i]) 
    
    #calculate mean of confusion matrix separately since its not a scalar but an array
    mean_confusion = np.zeros((2,2))
    confusions = classMetrics[8]
    for i in xrange(len(confusions)):
        mean_confusion = np.add(mean_confusion, confusions[i])
    mean_confusion = mean_confusion/len(confusions)
        
    
    means.append(mean_confusion)
    print metricLabels[8]+' is: '+repr(means[8])

#this function saves results to a file as well as printing some results 
def interpretResults(howMany, targ_test, predictions, probs, inds, filename):
    
    #figure out which indices correspond with test file
    whichInds = np.zeros(len(inds))
    for i in xrange(len(inds)):
        whichInds[i] = i
    whichInds = whichInds[inds]

    #print headers
    if howMany > 0:
        print('Instance, actual, predicted, error, class0prob, class1prob')

    #saves to filename
    with open(filename, 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in xrange(predictions.shape[0]):
            actual = targ_test[i]
            predicted = predictions[i]
            error = 0
            if actual == predicted:
                error = 1
            class0prob = round(probs[i][0],2)
            class1prob = round(probs[i][1],2)
        
            #this does the writing to the file
            file_writer.writerow([repr(whichInds[i])+','+repr(actual)+','+repr(predicted)+','+repr(error)+','+repr(class0prob)+','+repr(class1prob)])
        
            #printing
            if howMany >0 and i < howMany:
                print(repr(whichInds[i])+', '+repr(actual)+', '+repr(predicted)+', '+repr(error)+', '+repr(class0prob)+', '+repr(class1prob))


#main function
#optNum is either < 1 for %train/test split or > 1 for k-Fold cross val
#type is the type of supervised model you'd like to build, can be 'Tree', 'NB', 'logistic'
#howmany to print is a nonnegative integer denoting how many classified instances you'd like to see information about.

def main(optNum, type, howManyToPrint):

    #initialize split and model
    mySplit = splitFunc(target,optNum)
    clf = supClassOptions(type)
    classMetrics = metricInit()

    #train for each train/test split
    counter = 0
    for train, test in mySplit:
        
        #separates features and target into train and test
        feat_train = dataMat[train]
        feat_test = dataMat[test]
        targ_train = target[train]
        targ_test = target[test]
        
        #trains the model
        thisFit = clf.fit(feat_train, targ_train)
        
        #get predicted values and class probabilities
        myPredictions = thisFit.predict(feat_test)
        probs = thisFit.predict_proba(feat_test)
        pos_probs = probs[:,1] # get first column of probs array

        print probs
        print probs[:,0]
        #sys.exit()
        
        #compute the metrics of interest
        classMetrics = metricFunc(targ_test, myPredictions, pos_probs, classMetrics)
        
        #printing
        print ' '
        print 'Some results for test set number '+repr(counter+1)
        
        #writing to a file
        #interpretResults(howManyToPrint, targ_test, myPredictions, probs, test, 'mailing_results_'+repr(counter)+'.csv')
        counter = counter+1

    #printing average results across all train/test splits
    print ' '
    print 'Average metrics for classification model across '+repr(len(mySplit))+' test sets:'
    print ' '
    printMeanMetrics(classMetrics)
    print ' '


#change these values to experiment!!!
main(10, 'Logistic', 10)

