# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BASKARAN V
RegisterNumber:  212222230020
IMPORT A REQUIRD PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")

def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

*/
```

## Output:
![270406580-83ae155e-6d6d-41f2-8675-db2869e10897](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/1445ea45-95eb-400d-abe2-a898fa3c3dbf)
## Dataset shape
![270406740-fc57ce2c-5ecb-436f-bce7-e112c2fb50dc](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/953c054b-9dfa-4eef-86c8-29e113c87d5b)

![270406883-43acb6fa-2e0a-424f-a783-801eadb7432f](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/74032af6-9456-40a5-bb33-43dab0a15280)
## x,y,theta value:
![270407041-e84a8833-ace3-4e57-b85d-a4353ba3e94a](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/6807c20b-4e85-4444-b18d-7781572c2051)
## Gradient descent:
![270407135-eec1fdfe-4f4a-4cea-9045-d7cf4b69fe1e](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/654f1fb0-cac9-4cb0-977a-b2a08837b2a5)
## Cost function using Gradient Descent Graph:
![270407241-c56b85e9-7611-4a38-9887-951b6f7147a5](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/d8047da1-deec-4d64-971a-4faaa4add3dd)

## Profit Prediction Graph:

![270407387-00b307cb-bd16-4f04-a419-01e8a5e4ba1c](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/edbc72c0-c499-487d-b8a0-3534601e77e7)

## Profit Prediction
![270407493-158d456b-bc07-4501-96d5-92256e99aa4e](https://github.com/BaskaranV15/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118703522/9c1aa2d2-6bed-4aee-9f98-e48c3fe7b4cb)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
