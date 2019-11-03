import sys
import time
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row

conf = SparkConf().setMaster("local").setAppName("logistic")
sc=SparkContext(conf=conf)
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn import datasets



def ScikitLearnAccuracy(X,Y, X_test,Y_test):
	X1 = (np.array(X).astype(float))
	Y1 = (np.array(Y).astype(float))
	
	X_test1 = (np.array(X_test).astype(float))
	Y_test1 = (np.array(Y_test).astype(float))
	log_reg = LogisticRegression()
	log_reg.fit(X1,Y1)
	log_reg_soc = log_reg.score(X_test1,Y_test1)
	return log_reg_soc
	


def Sigmoid(z):
  	
	G_of_Z = float(1.0 / float((1.0 + np.math.exp(-1.0 * z))))
	return G_of_Z

def Predict(weights,bias, X):
    z = 0.0
    z=np.dot(np.asarray(X,dtype=np.float64),np.asarray(weights,dtype=np.float64)) 
    z = z + bias
    return Sigmoid(z)

def Cost_Function(X,Y,weights,bias,lambda_reg):  
	sumOfErrors = 0.0  
	error=0.0
	for i in xrange(len(X[0])):    
    		predicted = Predict(weights,bias,X[i]) 
		#print Y[i] 
    		if Y[i] == 1:  
        		error = math.log(predicted)  
    		if Y[i] == 0:  
       			error = math.log(1-predicted)  
    	
    		sumOfErrors += error  
     		
	sumOfWeightSquares = sum( i*i for i in weights)
       
     	cost =  (float(lambda_reg)*float(sumOfWeightSquares)/2 - sumOfErrors)/len(X[0])
       # print 'cost is ',  cost 
     	return cost
 
def BiasDerivative(X,Y,weights,bias,j):
    sumErrors = 0.0    length=len(X[0])
    for i in xrange(length):
        xi = X[i]
        xij = xi[j]
        predicted = Predict(weights,bias,X[i])
        error = (float(predicted) - float(Y[i]))
        sumErrors += error
    return sumErrors/length
def Derivative(X,Y,weights,bias,j,lambda_reg):
    sumErrors = 0.0
    for i in xrange(len(X[0])):
        xi = X[i]
        xij = xi[j]
        predicted = Predict(weights,bias,X[i])
        error = (float(predicted) - float(Y[i]))*float(xij)
        sumErrors += error    deriv = (sumErrors + float(lambda_reg)*float(weights[j]))/len(X[0])
    return deriv


def Gradient_Descent(X,Y,weights,bias,learning_rate,lambda_reg):
	new_weights = []	
	count = xrange(len(weights))
	for j in count:		deriv_weight = Derivative(X,Y,weights,bias,j,lambda_reg)
        	new_weights_value = weights[j] - deriv_weight*learning_rate
        	new_weights.append(new_weights_value)

    	new_bias = bias - BiasDerivative(X,Y,weights,bias,j)*learning_rate 
   	return new_weights,new_bias


def train(X,Y,num_iterations,learning_rate,lambda_reg):     
    weights = np.random.rand(56,1)
    bias = random.random() 
    cost = []
    for x in xrange(int(num_iterations)):  
        [new_weights,new_bias] = Gradient_Descent(X,Y,weights,bias,learning_rate,lambda_reg)  
        weights = new_weights  
        bias = new_bias  
        cost.append(Cost_Function(X,Y,weights,bias,lambda_reg))
           
    #plt.plot(range(num_iterations),cost)
    
    return weights,bias  


def Start_Train(X,Y,X_test,Y_test):
    print learning_rate, num_iterations, lambda_reg
    [we,bi] = train(X,Y,num_iterations,learning_rate,lambda_reg)
    AccuracyCheck(X_test,Y_test,we,bi)
    toc=time.clock()	
    print "elapsed time" ,toc-tic    
    #plt.show()

    log_reg_soc = ScikitLearnAccuracy(X,Y,X_test,Y_test)
    print 'Scikit score: ',log_reg_soc



def SplitFeaturesLabels(train_data):
   
    X = []
    Y = []
 
    for element in train_data:	X.append(element[0:56])
        Y.append(element[57])
    return X,Y

def split_train_test(data):
    random.shuffle(data)
    train_data = data[:int((len(data)+1)*.90)]  #Remaining 90% to training set
    test_data = data[int(len(data)*.90+1):] #Splits 10% data to test set
    [X,Y] = SplitFeaturesLabels(train_data)    [X_test,Y_test] = SplitFeaturesLabels(test_data)
    Start_Train(X,Y,X_test,Y_test)    
    
def Normalize():
    # load the dataset as a df
    df = pd.read_csv("spam.data", delim_whitespace =True, header=None)
    #normalize only the features not the labels
    df.iloc[:,np.r_[0:56]] = df.iloc[:,np.r_[0:56]].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    #create new file with normalized data    split_train_test(df.values.tolist())


def AccuracyCheck(X_test,Y_test,we,bi):
	score = 0
	length = len(X_test)
	for i in xrange(length):
	 	prediction = round(Predict(we,bi,X_test[i]))
		answer = int(Y_test[i])
		if prediction == answer:
			score += 1
	
	my_score = float(score) / float(length)
	
	print 'Your score: ', my_score
	
tic=time.clock()
args = map(float, sys.argv[1:])
learning_rate = args[0]
num_iterations = args[1]
lambda_reg = args[2]
	
#normalize the dataset
Normalize()


   
