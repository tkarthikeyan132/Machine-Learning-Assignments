#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import random


# In[2]:


def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y


# In[3]:


def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_pred, y_true)
    print(results)


# ##### Preparing the X_train, y_train, X_test, y_test

# In[4]:


x_train_path = "/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 3/Data_b/x_train.npy"
y_train_path = "/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 3/Data_b/y_train.npy"

X_train, y_train = get_data(x_train_path, y_train_path)

x_test_path = "/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 3/Data_b/x_test.npy"
y_test_path = "/home/tkarthikeyan/IIT Delhi/COL774-Machine Learning/Assignment 3/Data_b/y_test.npy"

X_test, y_test = get_data(x_test_path, y_test_path)

#you might need one hot encoded y in part a,b,c,d,e
label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))

y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))


# In[5]:


class NeuralNetwork:
    def __init__(self, n, n_hidden_nodes, r, M):
        #Number of nodes in the architecture
        self.n = n
        self.n_hidden_nodes = n_hidden_nodes
        self.r = r
        
        #Mini batch size
        self.M = M
        
        #Weights and biases
        self.W = dict()
        self.b = dict()
        
    def initialize_weights_and_biases(self):
        n_nodes = [self.n] + self.n_hidden_nodes + [self.r]
        
        #Initialize weights
        for i in range(1,len(n_nodes)):
            self.W[str(i)] = np.random.uniform(low=-0.1, high=0.1, size=(n_nodes[i], n_nodes[i-1]))
        
        #Initialize biases
        for i in range(1,len(n_nodes)):
            self.b[str(i)] = np.zeros((n_nodes[i],1))
     
    @staticmethod
    def sigmoid(x, derivative = False):
        if derivative == False:
            return 1 / (1 + np.exp(-x))
        else:
            return NeuralNetwork.sigmoid(x, derivative = False) * (1 - NeuralNetwork.sigmoid(x, derivative = False))
    
    @staticmethod
    def relu(x, derivative = False):
        if derivative == True:
            return np.where(x > 0, 1, np.where(x < 0, 0, np.random.random_sample()))
        else:
            return np.where(x <= 0, 0, x)
    
    @staticmethod
    def softmax(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def train(self, X_train, y_train, epoch_mode = True, activation="sigmoid", EPOCHS = 200, alpha = 0.01, stopping_threshold = None, adaptive_learning=False, printafter=20):
        self.initialize_weights_and_biases()
        
        a = dict()
        z = dict()
        del_z = dict()
        del_b = dict()
        del_W = dict()
        
        if epoch_mode == True:
            for epoch in range(EPOCHS):
                for i in range(0, X_train.shape[0], self.M):
                    y_actual = y_train[i:i+self.M,:].T
                    
                    #Forward
                    a["0"] = X_train[i:i+self.M,:].T
                    
                    for j in range(1,len(self.n_hidden_nodes)+1):
                        z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
                        if activation == "relu":
                            a[str(j)] = NeuralNetwork.relu(z[str(j)])
                        else:
                            a[str(j)] = NeuralNetwork.sigmoid(z[str(j)])
                    
                    j += 1
                    z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
                    a[str(j)] = NeuralNetwork.softmax(z[str(j)])
                    
                    #Backward
                    del_z[str(j)] = a[str(j)] - y_actual
                    del_b[str(j)] = np.sum(del_z[str(j)], axis = 1).reshape(-1,1)
                    del_W[str(j)] = np.matmul(del_z[str(j)], a[str(j-1)].T)
                    
                    for k in range(j-1,0,-1):
                        if activation == "relu":
                            del_z[str(k)] = np.matmul(self.W[str(k+1)].T, del_z[str(k+1)])*(NeuralNetwork.relu(z[str(k)], derivative=True))
                        else:
                            del_z[str(k)] = np.matmul(self.W[str(k+1)].T, del_z[str(k+1)])*(NeuralNetwork.sigmoid(z[str(k)], derivative=True))
                        del_b[str(k)] = np.sum(del_z[str(k)], axis = 1).reshape(-1,1)
                        del_W[str(k)] = np.matmul(del_z[str(k)], a[str(k-1)].T)
                    
                    #Update
                    for l in range(1,len(self.n_hidden_nodes)+2):
                        if adaptive_learning == False:
                            self.W[str(l)] = self.W[str(l)] - alpha * del_W[str(l)]
                            self.b[str(l)] = self.b[str(l)] - alpha * del_b[str(l)]
                        else:
                            self.W[str(l)] = self.W[str(l)] - (alpha/np.sqrt(epoch)) * del_W[str(l)]
                            self.b[str(l)] = self.b[str(l)] - (alpha/np.sqrt(epoch)) * del_b[str(l)]
                        
                y_pred, softmax_output = NN.predict(X_train, activation=activation)
                softmax_loss = NeuralNetwork.compute_softmax_loss(softmax_output, y_train_onehot)
                if epoch%printafter==0:
                    print(f"epoch {epoch}")
                    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred))
                    if adaptive_learning:
                        print("learning rate: ",(alpha/np.sqrt(epoch)))
                    print("softmax loss: ",softmax_loss)
                    print("\n")

        else:
            epoch = 1
            window = 5
            loss_avg = 0
            while(True):
                for i in range(0, X_train.shape[0], self.M):
                    y_actual = y_train[i:i+self.M,:].T
                    
                    #Forward
                    a["0"] = X_train[i:i+self.M,:].T
                    
                    for j in range(1,len(self.n_hidden_nodes)+1):
                        z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
                        if activation == "relu":
                            a[str(j)] = NeuralNetwork.relu(z[str(j)])
                        else:
                            a[str(j)] = NeuralNetwork.sigmoid(z[str(j)])
                    
                    j += 1
                    z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
                    a[str(j)] = NeuralNetwork.softmax(z[str(j)])
                    
                    #Backward
                    del_z[str(j)] = a[str(j)] - y_actual
                    del_b[str(j)] = np.sum(del_z[str(j)], axis = 1).reshape(-1,1)
                    del_W[str(j)] = np.matmul(del_z[str(j)], a[str(j-1)].T)
                    
                    for k in range(j-1,0,-1):
                        if activation == "relu":
                            del_z[str(k)] = np.matmul(self.W[str(k+1)].T, del_z[str(k+1)])*(NeuralNetwork.relu(z[str(k)], derivative=True))
                        else:
                            del_z[str(k)] = np.matmul(self.W[str(k+1)].T, del_z[str(k+1)])*(NeuralNetwork.sigmoid(z[str(k)], derivative=True))
                        del_b[str(k)] = np.sum(del_z[str(k)], axis = 1).reshape(-1,1)
                        del_W[str(k)] = np.matmul(del_z[str(k)], a[str(k-1)].T)
                    
                    #Update
                    for l in range(1,len(self.n_hidden_nodes)+2):
                        if adaptive_learning == False:
                            self.W[str(l)] = self.W[str(l)] - alpha * del_W[str(l)]
                            self.b[str(l)] = self.b[str(l)] - alpha * del_b[str(l)]
                        else:
                            self.W[str(l)] = self.W[str(l)] - (alpha/np.sqrt(epoch)) * del_W[str(l)]
                            self.b[str(l)] = self.b[str(l)] - (alpha/np.sqrt(epoch)) * del_b[str(l)]
                        
                y_pred, softmax_output = NN.predict(X_train, activation=activation)
                softmax_loss = NeuralNetwork.compute_softmax_loss(softmax_output, y_train_onehot)
                if epoch%printafter==0:
                    print(f"epoch {epoch}")
                    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred))
                    if adaptive_learning:
                        print("learning rate: ",(alpha/np.sqrt(epoch)))
                    print("softmax loss: ",softmax_loss)
                    print("\n")
                
                if epoch <= window:
                    loss_avg += softmax_loss
                    if epoch == window:
                        loss_avg /= window
                
                #End the training
                if epoch > window:
                    new_loss_avg = ((window - 1)*loss_avg + softmax_loss)/window
                    diff_avg_loss = abs(new_loss_avg - loss_avg)
                    #print("diff avg loss:",diff_avg_loss)
                    if diff_avg_loss < stopping_threshold or epoch > EPOCHS:
                        print("Convergence criteria satisfied!")
                        print(f"epoch {epoch}")
                        print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred))
                        if adaptive_learning:
                            print("learning rate: ",(alpha/np.sqrt(epoch)))
                        print("softmax loss: ",softmax_loss)
                        print("\n")
                        break
                
                epoch += 1
                        
    def predict(self, X_test, activation="sigmoid"):
        y_pred = np.zeros((X_test.shape[0], self.r))
        softmax_output = np.zeros((X_test.shape[0], self.r))
        z = dict()
        a = dict()
        
        for i in range(X_test.shape[0]):
            a[str(0)] = X_test[i:i+1,:].T
            
            for j in range(1,len(self.n_hidden_nodes)+1):
                z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
                if activation == "relu":
                    a[str(j)] = NeuralNetwork.relu(z[str(j)])
                else:
                    a[str(j)] = NeuralNetwork.sigmoid(z[str(j)])
            
            j += 1
            z[str(j)] = np.matmul(self.W[str(j)], a[str(j-1)]) + self.b[str(j)]
            a[str(j)] = NeuralNetwork.softmax(z[str(j)])
            
            softmax_output[i] = a[str(j)].flatten()
            y_pred[i][np.argmax(a[str(j)])] = 1
            
        return y_pred, softmax_output
    
    @staticmethod
    def compute_softmax_loss(softmax_output, y_pred):
        #Softmax loss
        sm_loss = 0
        for i in range(y_pred.shape[0]):
            sm_loss = -1*np.log2(softmax_output[i][np.argmax(y_pred[i])])
            
        return sm_loss/(y_pred.shape[0])


# In[28]:


NN = NeuralNetwork(n = 1024, n_hidden_nodes = [100,50] , r = 5, M = 32)
NN.train(X_train, y_train_onehot, activation="sigmoid", EPOCHS=100, alpha=0.001, printafter=10)
y_pred, softmax_output = NN.predict(X_test, activation="sigmoid")
print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred))
get_metric(y_test_onehot, y_pred)


# In[34]:


print("f1_score: ",f1_score(y_test_onehot, y_pred, average="macro"))


# #### Experimenting with single hidden layer

# In[41]:


hidden_layers = [[1],[5],[10],[50],[100]]
number_of_hidden_units = [1,5,10,50,100]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    NN = NeuralNetwork(n = 1024, n_hidden_nodes = hidden_layer , r = 5, M = 32)
    NN.train(X_train, y_train_onehot, epoch_mode= False, activation="sigmoid", alpha = 0.01, stopping_threshold = 1.0e-06, printafter=50)
    y_pred_train, _ = NN.predict(X_train)
    y_pred_test, _ = NN.predict(X_test)
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[43]:


#Plot avg f1_scores for different number of hidden units
plt.figure(figsize=(10,5))
plt.plot(number_of_hidden_units, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(number_of_hidden_units, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs number of hidden units')
plt.xlabel('number of hidden units')
plt.ylabel('Avg f1_score')
plt.xticks(number_of_hidden_units)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# #### Experimenting with hidden layers

# In[44]:


hidden_layers = [[512], [512,256], [512,256,128], [512,256,128,64]]
network_depth = [1,2,3,4]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    NN = NeuralNetwork(n = 1024, n_hidden_nodes = hidden_layer , r = 5, M = 32)
    NN.train(X_train, y_train_onehot, epoch_mode= False, activation="sigmoid", alpha = 0.01, stopping_threshold = 1.0e-06, printafter=50)
    y_pred_train, _ = NN.predict(X_train)
    y_pred_test, _ = NN.predict(X_test)
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[45]:


#Plot avg f1_scores for different depth
plt.figure(figsize=(10,5))
plt.plot(network_depth, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(network_depth, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs network depth')
plt.xlabel('network depth')
plt.ylabel('Avg f1_score')
plt.xticks(network_depth)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# #### Adaptive learning

# In[46]:


hidden_layers = [[512], [512,256], [512,256,128], [512,256,128,64]]
network_depth = [1,2,3,4]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    NN = NeuralNetwork(n = 1024, n_hidden_nodes = hidden_layer , r = 5, M = 32)
    NN.train(X_train, y_train_onehot, epoch_mode= False, activation="sigmoid", adaptive_learning = True, alpha = 0.01, stopping_threshold = 5.0e-06, printafter=50)
    y_pred_train, _ = NN.predict(X_train)
    y_pred_test, _ = NN.predict(X_test)
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[47]:


#Plot avg f1_scores for different depth
plt.figure(figsize=(10,5))
plt.plot(network_depth, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(network_depth, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs network depth')
plt.xlabel('network depth')
plt.ylabel('Avg f1_score')
plt.xticks(network_depth)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Relu activation function

# In[48]:


hidden_layers = [[512], [512,256], [512,256,128], [512,256,128,64]]
network_depth = [1,2,3,4]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    NN = NeuralNetwork(n = 1024, n_hidden_nodes = hidden_layer , r = 5, M = 32)
    NN.train(X_train, y_train_onehot, epoch_mode= False, activation="relu", adaptive_learning = True, alpha = 0.01, stopping_threshold = 5.0e-06, printafter=50)
    y_pred_train, _ = NN.predict(X_train, activation="relu")
    y_pred_test, _ = NN.predict(X_test, activation="relu")
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[49]:


#Plot avg f1_scores for different depth
plt.figure(figsize=(10,5))
plt.plot(network_depth, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(network_depth, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs network depth')
plt.xlabel('network depth')
plt.ylabel('Avg f1_score')
plt.xticks(network_depth)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Relu activation function (with learning rate = 0.001)

# In[6]:


hidden_layers = [[512], [512,256], [512,256,128], [512,256,128,64]]
network_depth = [1,2,3,4]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    NN = NeuralNetwork(n = 1024, n_hidden_nodes = hidden_layer , r = 5, M = 32)
    NN.train(X_train, y_train_onehot, epoch_mode= False, activation="relu", adaptive_learning = True, alpha = 0.001, stopping_threshold = 5.0e-06, printafter=50)
    y_pred_train, _ = NN.predict(X_train, activation="relu")
    y_pred_test, _ = NN.predict(X_test, activation="relu")
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[7]:


#Plot avg f1_scores for different depth
plt.figure(figsize=(10,5))
plt.plot(network_depth, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(network_depth, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs network depth')
plt.xlabel('network depth')
plt.ylabel('Avg f1_score')
plt.xticks(network_depth)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Neural Networks using scikit learn

# In[19]:


hidden_layers = [[512], [512,256], [512,256,128], [512,256,128,64]]
network_depth = [1,2,3,4]
f1_score_train = []
f1_score_test = []
for hidden_layer in hidden_layers:
    print(f"Hidden layer: {hidden_layer}")
    clf = MLPClassifier(activation="relu", solver="sgd", alpha = 0, batch_size=32, hidden_layer_sizes=np.array(hidden_layer), learning_rate="invscaling", tol=5e-6, n_iter_no_change=5, verbose=True, learning_rate_init=0.01).fit(X_train, y_train_onehot)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    print("accuracy on train data: ",accuracy_score(y_train_onehot, y_pred_train))
    print("metrics for train data: ")
    get_metric(y_train_onehot, y_pred_train)
    f1_score_train.append(f1_score(y_train_onehot, y_pred_train, average="macro"))
    
    print("accuracy on test data: ",accuracy_score(y_test_onehot, y_pred_test))
    print("metrics for test data: ")
    get_metric(y_test_onehot, y_pred_test)
    f1_score_test.append(f1_score(y_test_onehot, y_pred_test, average="macro"))
    
    print("\n")


# In[20]:


#Plot avg f1_scores for different depth
plt.figure(figsize=(10,5))
plt.plot(network_depth, f1_score_train, marker='o', markersize=6, color='blue', label='Avg f1_score train data')
plt.plot(network_depth, f1_score_test, marker='o', markersize=6, color='red', label='Avg f1_score test data')

plt.title('Avg f1_score vs network depth')
plt.xlabel('network depth')
plt.ylabel('Avg f1_score')
plt.xticks(network_depth)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# In[ ]:




