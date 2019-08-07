#------------------importing libraries--------------------------#
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#-----------------Loading data----------------------------------#
df = pd.read_csv("/home/siddhu/Downloads/new_final.csv",index_col=None)
df.set_index('ID',inplace=True)

#-----------------convering into matrix and defining input and output------#
data=np.array(df)#converting into the matrix form
m,n=np.shape(data)

y=data[:,-1].reshape(-1,1)#output set
print(np.shape(y))

def deconvert(Y):#converting 1 to g and 0 to h
	y=np.zeros((1,np.size(Y))).astype('str')
	for i in range(np.size(Y)):
		if Y[i]==1:
			y[0,i]="g"
		else:
			y[0,i]="h"
	return y

X=np.delete(data,-1,1)#input set
print((X))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)#splitting the data

#------------------baised featuring--------------#

def bais(X,y):#baising every row with one
  ones=np.array([(1)])
  for i in range(1,np.size(y)):
    ones=np.vstack((ones,np.array([(1)]))) # colomn matrix of ones
  
  X1=np.hstack((ones,X))#baised every feature with one
  X1=X1.astype('float64')#making all data of one datatype
  return X1

X1=bais(X_train,y_train)#baised X

#---sigmoid function------#

def sigmoid(z):
	return 1.0/(1.0+(np.exp(-z)))# logistic func

#-------intializing--------#

theta=np.zeros((X1.shape[1],1))#intailizing the theta
k=np.array([(0)])# intial matrix to store all J values in one array
e=1e-5# this is minimum value of log when the sigmoid value is 0
m=np.size(y_train)#no.of rows
J=0#intial value of cost functiom

#-----parameters--------#

alpha=0.0001#learning rate
ilter=range(2000)# no.of iteration


J=0
for i in ilter:
  #-----linear form-------#

  h=X1@(theta) #if datatype of all data is not same the u will get the 'matmul' error

  #-----applying logistic formula------#

  g=sigmoid(h)

  #------gradient decent------------#

  theta=theta-((alpha/m)*((np.dot(X1.T,(g-(y_train))))))

  #-----cost function---------#

  J=(np.average((-(y_train)*np.log(g+e))-((1.0-(y_train))*np.log(1-g+e))))
    
  k=np.vstack((k,np.array(J)))
  
print(J)

#------accuracy test--------#

final_theta=theta

def accuracy(X1,y):
  prediction=sigmoid(X1@(final_theta))
  predict=np.array([(0)])
  for i in prediction:
    if i<0.5:
      predict=np.vstack((predict,np.array([(0)])))
    else:
      predict=np.vstack((predict,np.array([(1)])))

  final_set=predict[1:,:]# predicted output
  print(final_set)

  f=abs(final_set-y)
  accuracy=100-(((np.sum(f))/m)*100)# accuracy
  print(accuracy,"%","accuracy")
  return final_set

f1=accuracy(X1,y_train)

#--------------J vs iteration----------------#
plt.plot(ilter,k[1:,:], label='Decision Boundary')
plt.show()

#------------final output--------------------#
final_output=deconvert(f1)# converting 1,0 to g,h
print(final_output)

#-------------test set-----------------------#

X2=bais(X_test,y_test)#prediction on testing set

f2=accuracy(X2,y_test)#test output

