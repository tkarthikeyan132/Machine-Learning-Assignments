#!/usr/bin/env python
# coding: utf-8

# ###### Importing required libraries

# In[1]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import numpy as np


# ###### Reading from file

# In[2]:


df1 = pd.read_csv('/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 1/ass1_data/data/q3/logisticX.csv', header=None)
df2 = pd.read_csv('/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 1/ass1_data/data/q3/logisticY.csv', header=None)


# ###### Creating a dataframe

# In[3]:


frames = [df1, df2]
df = pd.concat(frames, axis=1, ignore_index=True)
df.rename(columns={0:"X1",1:"X2",2:"Y"},inplace=True)
print(df)


# ###### Describing the data

# In[4]:


df.describe()


# ###### Normalizing the data

# In[5]:


norm_df = df.copy()
norm_df["X1"] = (norm_df["X1"] - norm_df["X1"].mean())/norm_df["X1"].std()
norm_df["X2"] = (norm_df["X2"] - norm_df["X2"].mean())/norm_df["X2"].std()

norm_df


# In[6]:


norm_df.describe()


# ###### Plotting the data with different colors

# In[7]:


class_0_x = []
class_0_y = []
class_1_x = []
class_1_y = []

for i in range(len(norm_df["X1"])):
    if norm_df["Y"][i] == 0:
        class_0_x.append(norm_df["X1"][i])
        class_0_y.append(norm_df["X2"][i])
    else:
        class_1_x.append(norm_df["X1"][i])
        class_1_y.append(norm_df["X2"][i])
        
plt.scatter(class_0_x,class_0_y,marker="^",color="green")
plt.scatter(class_1_x,class_1_y,marker="*",color="blue")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend(["class 0","class 1"])
plt.savefig("xy-plot.png")
plt.show()


# ###### Newton's method

# In[8]:


# Initializing theta to zero
theta = np.zeros((3,1))
# theta = np.random.rand(3,1)
m = norm_df["X1"].count()

print("Size of training set: ", m)


# ###### Stopping criterion

# The stopping criterion is usually of the form $ \| \nabla_\theta J_\theta(x_i) \|_2 $ <= $\phi$, where $\phi$ is small and
# positive.

# In[9]:


def convergence(grad_J_theta):
    phi = 0.0000000001
    if np.linalg.norm(grad_J_theta) < phi:
        return True
    else:
        return False


# ###### Sigmoid function

# In[10]:


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ###### Gradient vector

# In[11]:


# def get_gradient_vector(Y,H_theta,X,m):
#     n = X.shape[1]
#     G = np.zeros((n,1))
    
#     for i in range(n):
#         X_i = np.array(X[:,i],ndmin=2)
#         G_i = (H_theta - Y)*X_i.T
#         G[i][0] = G_i.sum(axis=0)/m
        
#     return G


# In[12]:


def get_gradient_vector(X,Y,H_theta):
    return np.matmul(X.T, H_theta - Y)


# ###### Hessian matrix

# In[13]:


# def get_hessian_matrix(H_theta,X,m):
#     n = X.shape[1]
#     H = np.zeros((n,n))
    
#     for i in range(n):
#         for j in range(i+1):
#             X_i = np.array(X[:,i],ndmin=2)
#             X_j = np.array(X[:,j],ndmin=2)
#             H_ij = H_theta*(1 - H_theta)*X_i.T*X_j.T
#             H[i][j] = H_ij.sum(axis=0)/m
#             H[j][i] = H[i][j]
    
#     return H


# In[14]:


def get_hessian_matrix_closed_form(H_theta, X):
    M = np.diag((H_theta*(1 - H_theta)).squeeze())
    return np.matmul(np.matmul(X.T,M),X)


# ###### Vector Notation

# In[15]:


# X = np.hstack(np.array(train_df["X_1"]), np.array(train_df["X_2"]))
a = np.array(norm_df["X1"], ndmin=2)
b = np.array(norm_df["X2"], ndmin=2)
c = np.ones((m,1))
X = np.hstack((c,a.T,b.T))
print("Shape of X is ",X.shape)
# print(X)


# In[16]:


X.T.shape


# In[17]:


Y = np.array(norm_df["Y"])
Y.resize(m,1)
print("Shape of Y is ",Y.shape)
# print(Y)


# ###### Algorithm

# In[18]:


while(True):
    H_theta = np.matmul(X,theta)
    H_theta = sigmoid(H_theta)
    
    L_theta = (-1)*(Y*np.log(H_theta) + (1 - Y)*np.log(1 - H_theta))
    L_theta = L_theta.sum(axis=0)/m

    grad_L_theta = get_gradient_vector(X,Y,H_theta)
    
    H = get_hessian_matrix_closed_form(H_theta, X)
    H_inv = np.linalg.inv(H)
    
    theta = theta - np.matmul(H_inv,grad_L_theta)
    # theta = theta - (0.01)*grad_L_theta
    
    # print("Theta:",theta)
    print("Loss:",L_theta)
    
    if convergence(grad_L_theta):
        break


# ###### Final set of parameters

# In[19]:


print("Theta: ", theta)


# ###### Plotting the data with different colors along with estimated separator line

# In[20]:


fig = plt.figure()
plt.scatter(class_0_x,class_0_y,marker="^",color="green")
plt.scatter(class_1_x,class_1_y,marker="*",color="blue")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axline((0,-theta[0][0]/theta[2][0]), slope=-theta[1][0]/theta[2][0], color="red")
plt.legend(["class 0","class 1","decision boundary"])
plt.savefig("xy-plot-with-decision-boundary.png")
plt.show()


# In[21]:


print("Final set of parameters:")
print(theta)


# In[22]:


###### Number of misclassifications

# arr = sigmoid(np.matmul(X,theta))

# brr = np.zeros((100,1),dtype="int")
# for i in range(100):
#     if arr[i][0] > 0.5:
#         brr[i][0] = 1
#     else:
#         brr[i][0] = 0
        
# crr = Y - brr

# incorrect = 0
# for i in range(100):
#     if crr[i][0] != 0:
#         incorrect += 1
        
# print("Incorrect classifications: ",incorrect)
# print("Total data: ",m)

