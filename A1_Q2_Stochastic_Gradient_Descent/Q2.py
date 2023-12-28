#!/usr/bin/env python
# coding: utf-8

# ###### Importing required libraries

# In[1]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import numpy as np
from datetime import datetime


# ###### Sampling 1 million data points
# CAUTION : RUN IT ONLY ONCE

data = {'X_1':np.random.normal(3, 2, 1000000),
        'X_2':np.random.normal(-1, 2, 1000000),
       'error':np.random.normal(0, np.sqrt(2), 1000000)}

train_df = pd.DataFrame(data)
train_df["Y"] = 3 + (1*train_df["X_1"]) + (2*train_df["X_2"]) + train_df["error"]
train_df.to_csv('q2train.csv', index=False)  
# ###### importing the sampled data 

# In[2]:


train_df = pd.read_csv('q2train.csv')
train_df.describe()


# ###### Plotting the data points on a 3D plane

# In[3]:


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x_1 = train_df["X_1"]
# x_2 = train_df["X_2"]
# y = train_df["Y"]
# ax.scatter(x_1, x_2, y, marker='o')

# ax.set_xlabel('x_1')
# ax.set_ylabel('x_2')
# ax.set_zlabel('y')

# plt.show()


# ##### Stochastic Gradient Descent

# In[4]:


eta = 0.001

sliding_win_size = 10
gamma = 3

theta = np.zeros((3,1))

gamma_count = 0
m = train_df["X_1"].count()
r = 1000000

phi = 0.000001

print("Learning rate: ", eta)
print("Batch size: ", r)


# ###### Stopping criterion

# In[5]:


def convergence(J_theta_mov_avg_t_1,J_theta_mov_avg_t, gamma):
    global gamma_count
    global phi

    # print("converge criteria ",abs(J_theta_mov_avg_t_1 - J_theta_mov_avg_t))
    # print("limit: ",phi)
    if abs(J_theta_mov_avg_t_1 - J_theta_mov_avg_t) < phi:
        gamma_count += 1
    else:
        gamma_count = 0
    if gamma_count >= gamma:
        return True
    else:
        return False


# ###### Algorithm

# In[6]:


# For plotting purpose
theta_and_J_theta = []


# ###### Shuffling the rows to create uniform batches

# In[7]:


train_sdf = train_df.sample(frac = 1)
curr_block = 0


# ###### Vector Notation

# In[8]:


# X = np.hstack(np.array(train_df["X_1"]), np.array(train_df["X_2"]))
a = np.array(train_sdf["X_1"], ndmin=2)
b = np.array(train_sdf["X_2"], ndmin=2)
c = np.ones((m,1))
X = np.hstack((c,a.T,b.T))
print("Shape of X is ",X.shape)
print(X)


# In[9]:


Y = np.array(train_sdf["Y"])
Y.resize(m,1)
print("Shape of Y is ",Y.shape)
print(Y)


# In[10]:


count_iterations = 1
J_theta_mov_avg_t = 0
J_theta_mov_avg_t_1 = 0

start = datetime.now()
while(True):
    X_b = X[curr_block*r:(curr_block+1)*r]
    Y_b = Y[curr_block*r:(curr_block+1)*r]
    
    H_theta = np.matmul(X_b,theta)
    
    J_theta = np.matmul((Y_b - H_theta).T, (Y_b - H_theta)).sum()/(2*r)
    
    grad_J_theta = np.array(((Y_b - H_theta)*(-1)*X_b).sum(axis=0)/r, ndmin=2)
    grad_J_theta = grad_J_theta.T
    
    # print("theta ",theta[0][0], theta[1][0], theta[2][0])
    # print("loss ", J_theta)
    
    theta_and_J_theta.append((theta[0][0], theta[1][0], theta[2][0], J_theta))
    
    # No converge check in first "sliding window" phase
    if count_iterations <= sliding_win_size:
        J_theta_mov_avg_t += J_theta
        if count_iterations == sliding_win_size:
            J_theta_mov_avg_t /= sliding_win_size
    
    # Moving average for the first time
    if count_iterations == sliding_win_size+1:
        J_theta_mov_avg_t_1 = (J_theta_mov_avg_t*(sliding_win_size-1)+ J_theta)/(sliding_win_size)
    
    # Convergence after "sliding window + 1" iterations
    if count_iterations > sliding_win_size+1:
        J_theta_mov_avg_t = J_theta_mov_avg_t_1
        J_theta_mov_avg_t_1 = (J_theta_mov_avg_t*(sliding_win_size-1)+ J_theta)/(sliding_win_size)
        if convergence(J_theta_mov_avg_t_1,J_theta_mov_avg_t, gamma):
            break
    
    theta = theta - (eta*grad_J_theta)
    
    curr_block = (curr_block+1)%(m//r)
    
    count_iterations += 1
    if(count_iterations%3000==0):
        print(str(count_iterations)+" Iterations over ,"+"theta = ",theta.T)
    
end = datetime.now()
td = (end - start).total_seconds() * (10**3)


# ###### Loss curve

# In[11]:


zs = [theta_and_J_theta[i][3] for i in range(len(theta_and_J_theta))]
t = [i for i in range(len(theta_and_J_theta))]
fig2 = plt.figure()
plt.plot(t,zs)
plt.title("Loss over iterations")
plt.ylabel("J_theta")
plt.xlabel("iterations")
plt.show()


# ###### Final Theta values

# In[12]:


print("Theta: ",theta)


# ###### Number of iterations

# In[13]:


print("count iterations: ", count_iterations)


# ###### Time Taken

# In[14]:


print(f"The time of execution of above program is {td:.03f} ms")


# In[15]:


# Loop implementation (not the vector implementation)

# while(True):
#     J_theta = 0.0
#     grad_J_theta = np.array([0.0,0.0,0.0])
    
#     for j in range(r):
#         i = curr_block*r+j
#         x_1i, x_2i, yi = train_sdf["X_1"][i], train_sdf["X_2"][i], train_sdf["Y"][i] 
#         h_theta_xi = theta[0] + theta[1]*x_1i + theta[2]*x_2i
#         J_theta += ((yi - h_theta_xi)**2)
#         grad_J_theta += (yi - h_theta_xi)*(-1)*np.array([1.0, x_1i, x_2i])
        
#     J_theta /= (2*r)
#     grad_J_theta /= r
    
#     print("curr block - loss: ",curr_block,J_theta)
#     print("theta = ",theta)

#     #Storing theta and its corresponding J_theta for every iteration
#     theta_and_J_theta.append((theta[0], theta[1], theta[2], J_theta))
    
#     #Stopping criteria
#     if convergence(grad_J_theta):
#         break
    
#     #Updating parameters
#     theta = theta - (eta*grad_J_theta)
    
#     #Go to next block
#     curr_block = (curr_block+1)%(m//r)


# ###### Reading test data from file 

# In[16]:


test_df = pd.read_csv('/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 1/ass1_data/data/q2/q2test.csv')


# ###### Describing the dataframe

# In[17]:


test_df.describe()


# In[18]:


test_df


# ###### Test data in the vector form

# In[19]:


n = len(test_df["X_1"])
at = np.array(test_df["X_1"], ndmin=2)
bt = np.array(test_df["X_2"], ndmin=2)
ct = np.ones((n,1))
Xtest = np.hstack((ct,at.T,bt.T))
Ytest = np.array(test_df["Y"], ndmin=2)
print(Xtest)


# ###### Computing the Mean Square error with test data

# In[20]:


H_theta = Ytest.T - np.matmul(Xtest,theta)
(1/n)*np.matmul(H_theta.T,H_theta)


# ###### Final Theta values

# In[21]:


theta


# ###### Computing the Mean Square error with original hypothesis

# In[22]:


theta = np.array([[3],[1],[2]])


# In[23]:


H_theta = Ytest.T - np.matmul(Xtest,theta)
(1/n)*np.matmul(H_theta.T,H_theta)


# ###### Movement of theta over time

# In[24]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[25]:


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = [theta_and_J_theta[i][0] for i in range(0,len(theta_and_J_theta),5)]
ys = [theta_and_J_theta[i][1] for i in range(0,len(theta_and_J_theta),5)]
zs = [theta_and_J_theta[i][2] for i in range(0,len(theta_and_J_theta),5)]
ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], marker='o')
ax.scatter(xs[-1], ys[-1], zs[-1], marker='*',s=100, color="red")
ax.scatter(xs[0], ys[0], zs[0], marker='*',s=100,color="green")

ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.savefig("3d_plot_"+str(r)+".png")
plt.show()


# In[ ]:




