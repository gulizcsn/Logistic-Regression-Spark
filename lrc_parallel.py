####This code is an implementation of parallel version of the logistic regression classifier on Spark with Python to detect spam emails from spam.data dataset.

import timeit
import sys
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row

conf = SparkConf().setMaster("local[4]").setAppName("logistic")
sc=SparkContext(conf=conf)
sc.setLogLevel("ERROR")


from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn import datasets

NUM_FEATURES=56
scikit_score=0.0
nfold=10

def ScikitLearnAccuracy(X,Y, X_test,Y_test):
	X1 = (np.array(X).astype(float))
	Y1 = (np.array(Y).astype(float))
	X_test1 = (np.array(X_test).astype(float))
	Y_test1 = (np.array(Y_test).astype(float))
	log_reg = LogisticRegression()
	log_reg.fit(X1,Y1)
	log_reg_soc = log_reg.score(X_test1,Y_test1)
	return log_reg_soc


#sigmoid function
def Sigmoid(z):   
	G_of_Z = float(1.0 / float((1.0 + np.math.exp(-1.0 * z))))
	return G_of_Z


#calculates predictions
def Predict(weights,bias, X):
	z = 0.0
	z=np.dot(np.asarray(X,dtype=np.float64),np.asarray(weights,dtype=np.float64)) 
	z = z + bias
	return Sigmoid(z)


def Cost_map(Y):
	if int(Y[1]) == 1:  
		error = math.log(Y[0])
	elif int(Y[1]) == 0:  
		error = math.log(1.0-Y[0])
	return error


def Cost_Function(X,Y,weights,bias,lambda_reg):  
	counts = len(X[0])
	predictedY = np.zeros((counts, 2))
    
#prepare a dataframe where one column is predictions and the other column is actual labels
	for i in xrange(counts):
		predicted = Predict(weights,bias,X[i])
		predictedY[i][0] = predicted
		predictedY[i][1] = Y[i]

#paralelly compute the cost using prepared dataframe
	rddY = sc.parallelize(predictedY)
	weightsRDD = sc.parallelize(weights)
	costPart1 = rddY.map(lambda y: Cost_map(y)).reduce(lambda x,y : x+y)
	costPart2 = weightsRDD.map(lambda x: x*x).reduce(lambda x,y : x+y)

#calculate final cost that includes regularization factor
	cost =  -(1.0/counts)*costPart1 + (lambda_reg/(2.0*counts))*costPart2
	return cost
 

def BiasDerivative(X,Y,weights,bias,j):
	sumErrors = 0.0
	length=len(X[0])
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
		sumErrors += error

#derivative is sum of errors plus regularization factor
	deriv = (sumErrors + float(lambda_reg)*float(weights[j]))/len(X[0])
	return deriv


def Gradient_Descent(X,Y,weights,bias,learning_rate,lambda_reg):
#update weights and bias with a factor of derivative time learning rate
	new_weights = []
	count = xrange(len(weights))
	for j in count:
		deriv_weight = Derivative(X,Y,weights,bias,j,lambda_reg)
		new_weights_value = weights[j] - deriv_weight*learning_rate
		new_weights.append(new_weights_value)
        
	new_bias = bias - BiasDerivative(X,Y,weights,bias,j)*learning_rate 
	return new_weights,new_bias


def Train(X,Y,num_iterations,learning_rate,lambda_reg):
#initialize weights and bias    
	weights = np.random.rand(NUM_FEATURES,1)
	bias = random.random() 
	cost = []
	for x in xrange(int(num_iterations)):
#use gradient descent to iteratively update weights and bias
		[new_weights,new_bias] = Gradient_Descent(X,Y,weights,bias,learning_rate,lambda_reg)  
		weights = new_weights  
		bias = new_bias
#calculate cost function at each iteration
		cost.append(Cost_Function(X,Y,weights,bias,lambda_reg))
	return weights,bias  


def Start_Train(X,Y,X_test,Y_test):
#training
	[we,bi] = Train(X,Y,num_iterations,learning_rate,lambda_reg)
#calculate accuracy and time and scikitlearn accuracy
	result = AccuracyCheck(X_test,Y_test,we,bi)
	toc=time.clock()
	print "elapsed time: " ,toc-tic    
	log_reg_soc = ScikitLearnAccuracy(X,Y,X_test,Y_test)
	print 'Scikit score: ',log_reg_soc
	print 'Your Result: ',result
	return result

def SplitFeaturesLabels(train_data):
	X = []
	Y = []
 
	for element in train_data:
		X.append(element[0:56])
		Y.append(element[57])
	return X,Y



def Cross_Validated_Split_train_test(data):
	random.shuffle(data)
	foldSize = int(len(data) / nfold)
# arrage to store training and testing error
	trainErr = [0.0] * nfold
	testErr = [0.0] * nfold
	allIndex = range(0, len(data))
	for i in range(0, nfold):
		test_data = data[int((len(data)+1)*.10*i):int((len(data)+1)*.10*(i+1))] 
		train_data1 = data[int((len(data)*.10*(i+1))+1):]
		train_data2 = data[:int((len(data)+1)*.10*i)]
		if ((i!=0) and (i!=nfold-1)):
			train_data =np.concatenate((train_data1,train_data2),axis=0)
		elif i==0:
			train_data=train_data1
		else:
			train_data=train_data2

#print np.shape(train_data1), np.shape(train_data2)
		[X,Y] = SplitFeaturesLabels(train_data)
		[X_test,Y_test] = SplitFeaturesLabels(test_data)
		resultAcc=Start_Train(X,Y,X_test,Y_test)
		testErr[i]=resultAcc
    print "average test err =", np.mean(testErr)

    
    
def Read_and_Normalize():
#load the dataset as a df
	df = pd.read_csv("spam.data", delim_whitespace =True, header=None)
#normalize only the features not the labels
	df.iloc[:,np.r_[0:56]] = df.iloc[:,np.r_[0:56]].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
#split data for train and test
	Cross_Validated_Split_train_test(df.values.tolist())
    

def AccuracyCheck(X_test,Y_test,we,bi):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	length = len(X_test)
	for i in xrange(length):
	 	prediction = round(Predict(we,bi,X_test[i]))
		answer = int(Y_test[i])
		if ((prediction == 1) and (answer == 1)):
			tp += 1
		elif ((prediction == 0) and (answer == 0)):
			tn += 1
		elif ((prediction == 1) and (answer == 0)):
			fp += 1
		elif ((prediction == 0) and (answer == 1)):
			fn += 1
	accuracy = (float(tp) + float(tn)) / float(length)
	error = 1-accuracy
	precision = float(tp) / (float(tp)+float(fp))
	recall = float(tp) / (float(tp) + float(fn)) 
	f1 = 2*precision*recall / (precision + recall)
	print 'Your score: ', accuracy
	print 'Error: ', error
	print 'Precision: ', precision
	print 'Recall: ', recall
	print 'f1: ',f1
	return accuracy
    

#initialize timer and get inputs from console
tic=time.clock()
args = map(float, sys.argv[1:])
learning_rate = args[0]
num_iterations = args[1]
lambda_reg = args[2]
#read data and normalize it
Read_and_Normalize()


