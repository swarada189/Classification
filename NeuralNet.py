#NN implementation for 1 input layer(30 neurons), 1 hidden layer(10 neurons), 1 output layer (2 neurons)

'''
Algorithm

grad descent: 
	 for no. of iterations
	  find output (by calling feedfwd)
	  calc error (cost)
	  try to reduce it by adjusting wts(by calling backprop)

feedfwd: calc activations	
 
backprop: 
	 grad= derivative of cost fn wrt wts(theta1 & theta2)&biases
	 update wts	 

'''
import numpy as np 					# linear algebra
import pandas as pd 					# only for data processing
import seaborn as sns 					# visualization
import matplotlib.pyplot as plt				#visualization		
from sklearn.metrics import confusion_matrix		#performance evaluation
from sklearn.metrics import roc_curve, auc	
from sklearn.model_selection import StratifiedKFold	#cross-validation
from scipy import interp				#cross-validation
from itertools import cycle				#cross-validation



def activation(z):						#activation function
	#return 1/(1+np.exp(-z))
	return np.tanh(z)
	
def actGradient(z):					#gradient=>slope=>derivative of sigmoid
	#return z*(1-z)
	return 1.0 - np.tanh(z)**2

class MyNeuralNetwork:					
	def __init__(self,x,y):
		self.X = x					#input vector of size 1000 x 30 	
		self.y = y					#output vector of size 1000 x 1
		maxv=0.9					#range for random seed value
		minv=0.1
		self.theta1= (np.random.rand(self.X.shape[1],10)*(maxv- minv) ) + minv  # Weights matrix1 of size 30 x 20
		self.theta2=(np.random.rand(10,self.y.shape[1])*(maxv - minv) ) + minv   #Weights matrix2 of size 20 x 1	
		self.h=np.zeros(self.y.shape)			#initialize hypothesis output with zeros of size 1000 x 1
	

	def feedForward(self,X):					#calculates output based on inputs and weights
		
		self.z=activation(np.dot(X, self.theta1))			#1000 x 20	
		self.h=activation(np.dot(self.z, self.theta2))		#1000 x 1
		return self.h						#returns hypothesis output of size 1000 x 1


	def backPropagation(self):					#adjusts weight matrices by backpropagating error
		learning_rate=0.9
		#error terms calculation
		del_theta2= -(actGradient(self.h)*(self.h - self.y))	#1000 x 1
		
		del_theta1= actGradient(self.z)* (np.dot(del_theta2,self.theta2.T))	#1000 x 20
		
		#update weights
		self.theta2 += learning_rate*(np.dot(self.z.T , del_theta2))	#20 x 1
		self.theta1 += learning_rate*(np.dot(self.X.T,del_theta1))	#30 x 20
		

	def gradDescent(self,no_of_iter):				#for finding weights such that MS error(cost) is minimized
		for i in range(no_of_iter): 
			if i % 100 ==0: 
				print ("For iteration # " + str(i) + "\n")		#prints cost for every iteration
				print ("Cost: \n" + str(np.mean(np.square(self.feedForward(self.X) - self.y)))) #MSE= pred - actual
				print ("\n")
			self.h=self.feedForward(self.X)			#calculate output		
			self.backPropagation()				#adjusts the weights 	

	def train(self):						#trains NN with 1500 iterations
		#no_of_iter=input("Enter Number of iterations : ")
		no_of_iter=1500		
		self.gradDescent(no_of_iter)

	def test(self,x,y):						#testing NN
		output=self.feedForward(x)
		#output=output.astype(int)
		
		for i in range(output.shape[0]):
			if output[i][0]>=0.5:			#for mapping the output to -1 or 1(i.e.phishing or legitimate website)
				output[i][0]=int(1)
			else:
				output[i][0]=int(-1)
		
		print("classified as :"+ str(output[:5,:]))
		
		self.evaluateNN(y,output)
		return output



	def evaluateNN(self,actual,pred):		#returns confusion matrix, fscore, accuracy,ROC
	
		#print actual.shape			#for debugging purpose
		#print pred.shape

		#print actual.dtype			#for debugging purpose
		#print pred.dtype	

		
		print "Confusion matrix"
		conmat=confusion_matrix(actual,pred)
		print conmat

		returned = confusion_matrix(actual,pred).ravel()
		print len(returned)			
		if len(returned)==4:
			tn, fp, fn, tp = returned		
		tn=float(tn)
		fp=float(fp)
		fn=float(fn)
		tp=float(tp)
		accuracy=(tp+tn)/(tp+fn+fp+tn)
		print("\nAccuracy = "+ str(accuracy*100) + " %")
		
		recall=tp/(tp+fn)
		precision=tp/(tp+fp)	
		fscore=2*(recall * precision)/(recall + precision)
		print("Recall = "+ str(recall)+ "\nPrecision = "+ str(precision)+ "\nF-Score = "+ str(fscore))

		with open('output.txt', 'a') as f:
        		f.write("ConMatrix= %s\tAccuracy= %s\t Recall= %s\t Precision=%s\t Fscore= %s\n " %( str(conmat),str(accuracy*100),str(recall),str(precision),str(fscore) ))
		
		
if __name__ == "__main__":
	
	data=pd.read_csv("phishing_site_data.csv")	#reads data from csv file and stores in pandas dataframe
	#print data.sample(n=2)				#prints sample of data
	
	
	X=data.values[:,:30]				#stores 1 to 30 attribute columns as input vector	
	y=data.values[:,30:31]				#stores last "Result" attribute column as output vector
	#print X[0]			
	#print y[0]					#for debugging purpose

	#After appplying cross-validation

	cv = StratifiedKFold(n_splits=10)		#perform 10 folds cross-validation

	tprs = []					#true positive rate
	aucs = []					#area under curve
	mean_fpr = np.linspace(0, 1, 100)	

	i = 0
	for train, test in cv.split(X, y):		#splits data into train and test
		nn=MyNeuralNetwork(X[train],y[train])	#initialise object of NN class
		nn.train()				#trains NN	
		output=nn.test(X[test],y[test])			#tests NN
		fpr, tpr, thresholds = roc_curve(y[test], output[:, 0])	#sets parameters for ROC curve
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
	 	roc_auc = auc(fpr, tpr)
	  
	   	aucs.append(roc_auc)
	   	plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))	#plots ROC

   		i += 1

	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
		 label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
		 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		 lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
			 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC for Neural Network')
	plt.legend(loc="lower right")
	plt.show()
	#Plese find below the earlier version of NN code before applying CV
	#It splits the data as 90% trainset and 10%test set. 
	#Then trains model, classifies test set and evaluates performance	
	'''
	train_test_per = 90/100.0
	data['train'] = np.random.rand(len(data)) < train_test_per
	#print data.sample(n=5)

	train=data[data.train==1]
	train=train.drop('train',axis=1).sample(frac=1)
	print("# Training samples are :"+ str(len(train)))
	#print train.sample(n=5)

	test=data[data.train==0]
	test=test.drop('train',axis=1)
	print("# Testing samples are :"+ str(len(test)))
	#print test.sample(n=5)

	X=train.values[:,:30]			

	y=train.values[:,30:31]

	print("X= ")
	print X[0]

	print("y= ")
	print y[0]

	nn=MyNeuralNetwork(X,y)

	nn.train()
	print("Training completed!\n")
	
	X=test.values[:,:30]	
	y=test.values[:,30:31]		
	unlabeled_data=X
	output=nn.test(unlabeled_data,y)
	output=output.astype(int)
	print("Actual values: " + str(y[:5,:]))
	print("Testing completed!")

	#evaluation starts : Confusion Matrix, Accuracy, F-score, ROC curve.	
	#nn.evaluateNN(y,output)
	'''
	
	

	
