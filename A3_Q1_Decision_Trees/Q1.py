#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import statistics as stat
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


# In[2]:


label_encoder = None 


# In[3]:


def get_np_array(file_name, encoding="onehot"):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','opp','host','month', 'day_match']
#     if(label_encoder is None):
    if encoding == "ordinal":
        label_encoder = OrdinalEncoder()
    else:
        label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()


# In[4]:


class DTNode:
    def __init__(self, depth, my_id=None, is_leaf=False, value=0, data=None, columns=None, split_feature = None, split_feature_value = None, threshold = None):
        #To uniquely identify a node
        self.my_id = my_id
        
        self.depth = depth 

        #add children afterwards
        self.children = []
        
        #if leaf
        self.is_leaf = is_leaf
        
        #It stores the majority y
        self.value = value 
        
        #Below attributes makes sense for decision nodes (non leaf nodes)
        self.split_feature = split_feature
        self.split_feature_value = split_feature_value #only makes sense for categorical attributes. ex we want to move to children with weather="sunny"
        self.threshold = threshold #only makes sense for continouos attributes

        self.columns = columns
        self.data = data


# In[5]:


class DTTree:
    def __init__(self, X_train, y_train, types):
        #Tree root should be DTNode
        self.root = None 
        
        self.X_train = X_train
        self.y_train = y_train
        self.types = types
        
        self.id_counter = 0
        
        self.columns = [i for i in range(X_train.shape[1])]
        self.data = np.array([i for i in range(len(X_train))])
        
    #returns a dictionary with unique values and corresponding split
    def get_data_split(self, data, idx):
        data_split = dict()
        # print(data)
        X_temp = self.X_train[data]
        
        if self.types[idx] == 'cat':
            unique_vals = np.unique(X_temp[:,idx])
            for val in unique_vals:
                indices = np.array([i for i, value in enumerate(X_temp[:,idx] == val) if value])
                data_split[val] = data[indices]
            return data_split
                
        elif self.types[idx] == 'cont':
            median = stat.median(X_temp[:,idx])
            
            indices = np.array([i for i, value in enumerate(X_temp[:,idx] <= median) if value])
            if indices.shape[0] != 0:
                data_split["less_than_equal"] = data[indices]
                
            indices = np.array([i for i, value in enumerate(X_temp[:,idx] > median) if value])
            if indices.shape[0] != 0:
                data_split["more_than"] = data[indices]
                
            return data_split
        
    @staticmethod
    def compute_entropy(y, data):
        y_temp = y[data]
        unique_values, value_counts = np.unique(y_temp, return_counts=True)
        # print("unique values:",unique_values)
        # print("value_counts:",value_counts)
        entropy = 0
        for val in value_counts:
            p = val/value_counts.sum()
            entropy += (-1)*p*np.log2(p)
        return entropy, unique_values[0]
    
    @staticmethod
    def compute_entropy_attr(X, y, data, col, col_type=None):
        entropy = 0
        X_temp = X[data]
        y_temp = y[data]
        S = len(data)
        # print("total:",S)
        # print(col, col_type)
        if col_type == 'cat':
            unique_vals = np.unique(X_temp[:,col])
            for val in unique_vals:
                S_val = np.count_nonzero(X_temp[:,col]==val)
                entropy_temp,_ = DTTree.compute_entropy(y_temp, X_temp[:,col]==val)
                # print(val, S_val, entropy_temp)
                entropy += (S_val/S)*entropy_temp
        elif col_type == 'cont':
            median = stat.median(X_temp[:,col])
            S_val1 = np.count_nonzero(X_temp[:,col] <= median)
            S_val2 = np.count_nonzero(X_temp[:,col] > median)
            entropy_temp1,_ = DTTree.compute_entropy(y_temp, X_temp[:,col] <= median)
            entropy_temp2,_ = DTTree.compute_entropy(y_temp, X_temp[:,col] > median)
            # print(S_val1, entropy_temp1)
            # print(S_val2, entropy_temp2)
            entropy = ((S_val1/S)*entropy_temp1)+((S_val2/S)*entropy_temp2)
        return entropy
        
    def get_majority_y(self, data):
        y_temp = self.y_train[data]
        y_temp = y_temp.reshape(-1)
        return stat.mode(y_temp)
        
    def build_tree(self, this_node=None, data = None, columns = None):
        # print(data)
        entropy, y_pred = DTTree.compute_entropy(self.y_train, data)
        # print("entropy: ",entropy)
        #when entropy would be zero,only one type of y would be present, y_pred makes sense else no
        if entropy == 0:
            this_node.is_leaf = True
            this_node.value = y_pred
            
            #Updating ID
            if this_node.my_id == None:
                this_node.my_id = self.id_counter
                self.id_counter += 1
            return this_node
        else:
            if(len(columns) == 0):
                this_node.is_leaf = True
                this_node.value = self.get_majority_y(data)
                
                #Updating ID
                if this_node.my_id == None:
                    this_node.my_id = self.id_counter
                    self.id_counter += 1
                return this_node
            #Splitting criteria
            entropy_cols = np.zeros(len(columns))
            IG_cols = np.zeros(len(columns))
            for i in range(len(columns)):
                entropy_cols[i] = DTTree.compute_entropy_attr(self.X_train, self.y_train, self.data, columns[i], self.types[columns[i]])
                IG_cols[i] = entropy - entropy_cols[i]
                # print(columns[i],IG_cols[i])
            #creating a node and returning
            idx = columns[np.argmax(IG_cols)]
            
            this_node.is_leaf = False
            this_node.split_feature = idx
            this_node.value = self.get_majority_y(data)
            this_node.columns = columns
            this_node.data = data
            
            data_split = DTTree.get_data_split(self, data, idx)
            
            if self.types[idx] == 'cat':
                # print(f"BUILD cat idx={idx}")
                for key,val in data_split.items():
                    new_columns = columns.copy()
                    
                    new_columns.remove(idx)
                    
                    child_node = DTNode(depth = this_node.depth + 1, my_id = self.id_counter, is_leaf=False, split_feature_value=key)
                    self.id_counter += 1
                    this_node.children.append(self.build_tree(child_node, data=val, columns=new_columns))
            elif self.types[idx] == 'cont':
                # print(f"BUILD cont idx={idx}")
                if len(data_split) > 1:
                    for key,val in data_split.items():
                        X_temp = self.X_train[data]
                        this_node.threshold = stat.median(X_temp[:,idx])
                        child_node = DTNode(depth = this_node.depth + 1, my_id = self.id_counter, is_leaf=False, split_feature_value=key)
                        self.id_counter += 1
                        this_node.children.append(self.build_tree(child_node, data=val, columns=columns))
                else:
                    new_columns = columns.copy()
                    
                    new_columns.remove(idx)
                    
                    for key,val in data_split.items():
                        X_temp = self.X_train[data]
                        this_node.threshold = stat.median(X_temp[:,idx])
                        child_node = DTNode(depth = this_node.depth + 1, my_id = self.id_counter, is_leaf=False, split_feature_value=key)
                        self.id_counter += 1
                        this_node.children.append(self.build_tree(child_node, data=val, columns=new_columns))
                    
            return this_node
            # print("Attribute to split: ", columns[idx])
            
    @staticmethod
    def print_node(root):
        if root.is_leaf:
            print(f"I am leaf and my value is {root.value} and my depth is {root.depth}")
        else:
            # print(f"I am not a leaf and i have {len(root.children)} and my value is {root.value} and my depth is {root.depth}")
            for child in root.children:
                # print(f"parent info:: split feature: {root.split_feature}, split feature value: {root.split_feature_value}, threshold: {root.threshold}")
                # print(f"split feature: {child.split_feature}, split feature value: {child.split_feature_value}, threshold: {child.threshold}")
                DTTree.print_node(child)        
    
    def print_tree(self):
        DTTree.print_node(self.root)
        
    def predict(self, X_test, max_depth):
        y_test_pred = np.zeros((len(X_test),1))
        for index in range(len(X_test)):
            x = X_test[index]
            temp = self.root
            pred = None
            while(True):
                #Terminating condition
                if temp.is_leaf == True or temp.depth >= max_depth:
                    pred = temp.value
                    break

                split_feature = temp.split_feature
                type_split_feature = self.types[split_feature]

                if type_split_feature == "cat":
                    z = x[split_feature]
                else:
                    if x[split_feature] <= temp.threshold:
                        z = "less_than_equal"
                    else:
                        z = "more_than"

                flag = 0

                for child in temp.children:
                    if child.split_feature_value == z:
                        temp = child
                        flag = 1

                #z not found in children, so return the prediction
                if flag == 0:
                    pred = temp.value
                    break

            y_test_pred[index][0] = pred
            
        return y_test_pred
            
    def execute(self):
        this_node = DTNode(depth=0, my_id=0, is_leaf=False)
        self.id_counter += 1
        self.root = self.build_tree(this_node = this_node, data = self.data, columns = self.columns)
        print("Decision tree built successfully!")


# #### Decision tree with both categorical attributes as well as continuous attributes (ordinal encoding)

# In[6]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="ordinal")
X_test, y_test = get_np_array("../Data/test.csv", encoding="ordinal")
X_val, y_val = get_np_array("../Data/val.csv", encoding="ordinal")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]


# In[7]:


dtree = DTTree(X_train,y_train, types)
dtree.execute()


# In[8]:


dtree.print_tree()


# In[9]:


max_depth_list = [5,10,15,20,25]
training_acc = []
testing_acc = []

for max_depth in max_depth_list:
    print("Max depth: ", max_depth)
    
    y_train_pred = dtree.predict(X_train, max_depth = max_depth)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("training accuracy: ",train_acc)
    training_acc.append(train_acc)

    y_test_pred = dtree.predict(X_test, max_depth = max_depth)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("testing accuracy: ",test_acc)
    testing_acc.append(test_acc)
    
    print("\n")


# In[10]:


#Plot training and testing accuracies for different max_depth
plt.figure(figsize=(10,5))
plt.plot(max_depth_list, training_acc, marker='o', markersize=6, color='blue', label='Training Accuracies')
plt.plot(max_depth_list, testing_acc, marker='o', markersize=6, color='red', label='Testing Accuracies')

plt.title('Training and Testing accuracy vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.xticks(max_depth_list)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Only win prediction on test data

# In[11]:


y_test_pred = np.ones((len(X_test),1))
print("[ONLY WIN]testing accuracy: ",accuracy_score(y_test, y_test_pred))


# ##### Only loss prediction on test data

# In[12]:


y_test_pred = np.zeros((len(X_test),1))
print("[ONLY LOSS]testing accuracy: ",accuracy_score(y_test, y_test_pred))


# #### Decision tree with only categorical attributes (one-hot encoding)

# In[7]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="onehot")
X_test, y_test = get_np_array("../Data/test.csv", encoding="onehot")
X_val, y_val = get_np_array("../Data/val.csv", encoding="onehot")


# In[8]:


types = ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cont', 'cat', 'cat', 'cat', 'cont', 'cont', 'cont']


# In[9]:


# dtree_1h = DTTree(X_train,y_train, types)
# dtree_1h.execute()


# In[45]:


import pickle

# file = open('dtree_1h.obj','wb')
# pickle.dump(dtree_1h,file)
# file.close()


# In[46]:


rfile = open('dtree_1h.obj','rb')
dtree_1h = pickle.load(rfile)
rfile.close()
print("Decision tree loaded!")


# In[12]:


dtree_1h.print_tree()


# In[13]:


# for idx in range(dtree_1h.X_train.shape[1]):
#     unique_vals = np.unique(dtree_1h.X_train[:,idx])
#     print(idx, len(unique_vals))


# In[14]:


max_depth_list = [15,25,35,45,55,65,75]
training_acc = []
testing_acc = []

for max_depth in max_depth_list:
    print("Max depth: ", max_depth)
    
    y_train_pred = dtree_1h.predict(X_train, max_depth = max_depth)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("training accuracy: ",train_acc)
    training_acc.append(train_acc)

    y_test_pred = dtree_1h.predict(X_test, max_depth = max_depth)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("testing accuracy: ",test_acc)
    testing_acc.append(test_acc)
    
    print("\n")


# In[15]:


#Plot training and testing accuracies for different max_depth
plt.figure(figsize=(10,5))
plt.plot(max_depth_list, training_acc, marker='o', markersize=6, color='blue', label='Training Accuracies')
plt.plot(max_depth_list, testing_acc, marker='o', markersize=6, color='red', label='Testing Accuracies')

plt.title('Training and Testing accuracy vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.xticks(max_depth_list)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# In[47]:


def count_nodes(root):
    if root.is_leaf == True:
        return 1
    count = 1
    for child in root.children:
        count += count_nodes(child)
    return count


# In[48]:


# count_nodes(dtree.root)


# In[49]:


count_nodes(dtree_1h.root)


# #### Post pruning in decision tree

# In[120]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="onehot")
X_test, y_test = get_np_array("../Data/test.csv", encoding="onehot")
X_val, y_val = get_np_array("../Data/val.csv", encoding="onehot")

types = ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cont', 'cat', 'cat', 'cat', 'cont', 'cont', 'cont']


# In[133]:


#Dictionary to store the children info
children = dict()

#List of non leaf nodes
non_leaf_node_ids = []

#It stores the children dictionary and fills up the list of non_leaf_node_ids subject to max_depth
def store_children_info(root, max_depth):
    global children, non_leaf_node_ids
    if root.is_leaf == True or root.depth >= max_depth:
        a=1 #dummy statement
    else:
        non_leaf_node_ids.append(root.my_id)
        for child in root.children:
            children[root.my_id] = children[root.my_id] + [child.my_id]
            store_children_info(child, max_depth)

#It returns the list of all children, grand children, .. of idx with help of children dict  
def return_all_childrens(children, idx):
    mega_lst = children[idx]
    for child in children[idx]:
        mega_lst = mega_lst + return_all_childrens(children, child)
    return mega_lst


# In[134]:


def predict_new(root, types, X_test, y_test, non_leaf_considered_leaf, max_depth):
    y_test_pred = np.zeros((len(X_test),1))
    for index in range(len(X_test)):
        x = X_test[index]
        temp = root
        pred = None
        while(True):
            #Terminating condition
            if temp.depth >= max_depth or (temp.is_leaf == True) or (temp.my_id in non_leaf_considered_leaf):
                pred = temp.value
                break

            split_feature = temp.split_feature
            type_split_feature = types[split_feature]

            if type_split_feature == "cat":
                z = x[split_feature]
            else:
                if x[split_feature] <= temp.threshold:
                    z = "less_than_equal"
                else:
                    z = "more_than"

            flag = 0

            for child in temp.children:
                if child.split_feature_value == z:
                    temp = child
                    flag = 1

            #z not found in children, so return the prediction
            if flag == 0:
                pred = temp.value
                break

        y_test_pred[index][0] = pred

    acc = accuracy_score(y_test, y_test_pred)
    return acc

def post_pruning(root, max_depth):
    no_non_leaf_nodes = []
    train_acc = []
    test_acc = []
    val_acc = []
    
    prob_pruning_when_no_improvement = 0.25
    
    global children, non_leaf_node_ids
    #Reinitializing the children dict
    children = dict()
    for i in range(count_nodes(dtree_1h.root)):
        children[i]= []
        
    #Reinitializing non leaf node ids
    non_leaf_node_ids = []

    store_children_info(dtree_1h.root, max_depth)

    v_acc = predict_new(root, types, X_val, y_val, non_leaf_considered_leaf = [], max_depth= max_depth)
    tr_acc = predict_new(root, types, X_train, y_train, non_leaf_considered_leaf = [], max_depth= max_depth)
    te_acc = predict_new(root, types, X_test, y_test, non_leaf_considered_leaf = [], max_depth= max_depth)
    
    no_non_leaf_nodes.append(len(non_leaf_node_ids))
    train_acc.append(tr_acc)
    test_acc.append(te_acc)
    val_acc.append(v_acc)
    
    
    print(f"Initial validation accuracy: {v_acc}")

    random.shuffle(non_leaf_node_ids)
    
    #Below code would store the non leaf which would be considered as leaf based on validation accuracy
    non_leaf_considered_leaf = []

    while(len(non_leaf_node_ids) > 0):
        scores = np.zeros(len(non_leaf_node_ids))
        for i in range(len(non_leaf_node_ids)):
            scores[i] = predict_new(root, types, X_val, y_val, non_leaf_considered_leaf = non_leaf_considered_leaf + [non_leaf_node_ids[i]], max_depth= max_depth) - v_acc
        k = np.argmax(scores)
        curr_node = non_leaf_node_ids[k]
        if scores[k] < 0:
            break
        elif scores[k] == 0 and np.random.sample() < prob_pruning_when_no_improvement:
            break
        
        v_acc = v_acc + scores[k]
        tr_acc = predict_new(root, types, X_train, y_train, non_leaf_considered_leaf = non_leaf_considered_leaf + [curr_node], max_depth= max_depth)
        te_acc = predict_new(root, types, X_test, y_test, non_leaf_considered_leaf = non_leaf_considered_leaf + [curr_node], max_depth= max_depth)

        no_non_leaf_nodes.append(len(non_leaf_node_ids))
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        val_acc.append(v_acc)
           
        non_leaf_node_ids.remove(curr_node)
        #update non_leaf_node_ids (in case when curr_node would be considered leaf)
        children_of_curr_node = return_all_childrens(children, curr_node)
        non_leaf_node_ids = [item for item in non_leaf_node_ids if item not in children_of_curr_node]
        non_leaf_considered_leaf = non_leaf_considered_leaf + [curr_node]
        print(f"Validation accuracy is {v_acc}. node with id:{curr_node} is now a leaf node. unvisited_non_leaf_nodes:{len(non_leaf_node_ids)}")

    #Training accuracy
    print("Final Training accuracy: ",predict_new(root, types, X_train, y_train, non_leaf_considered_leaf = non_leaf_considered_leaf, max_depth= max_depth))
    
    #Validation accuracy
    print("Final Validation accuracy: ",predict_new(root, types, X_val, y_val, non_leaf_considered_leaf = non_leaf_considered_leaf, max_depth= max_depth))
    
    #Testing accuracy
    print("Final Testing accuracy: ",predict_new(root, types, X_test, y_test, non_leaf_considered_leaf = non_leaf_considered_leaf, max_depth= max_depth))
    
    #Plot avg f1_scores for different depth
    plt.figure(figsize=(10,5))
    plt.plot(no_non_leaf_nodes, train_acc, marker='o', markersize=6, color='blue', label='train accuracy')
    plt.plot(no_non_leaf_nodes, test_acc, marker='o', markersize=6, color='red', label='test accuracy')
    plt.plot(no_non_leaf_nodes, val_acc, marker='o', markersize=6, color='green', label='validation accuracy')
    
    plt.title('accuracy vs non leaf nodes')
    plt.xlabel('non leaf nodes')
    plt.ylabel('accuracy')
    plt.gca().invert_xaxis()  # this line reverses the x-axis
    plt.xticks(no_non_leaf_nodes)
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()


# In[135]:


post_pruning(dtree_1h.root, 15)


# In[136]:


post_pruning(dtree_1h.root, 25)


# In[137]:


post_pruning(dtree_1h.root, 35)


# In[138]:


post_pruning(dtree_1h.root, 45)


# ### Decision tree using scikit learn

# In[6]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="onehot")
X_test, y_test = get_np_array("../Data/test.csv", encoding="onehot")
X_val, y_val = get_np_array("../Data/val.csv", encoding="onehot")


# ##### Keeping criterion as entropy

# In[7]:


clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, y_train)


# In[8]:


y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
print(f"Training accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred)}")


# ##### (i) Varying the max_depth = {15,25,35,45} and reporting the train and test accuracies

# In[9]:


max_depths = [15,25,35,45]
train_accuracy = []
test_accuracy = []
for max_depth in max_depths:
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"max_depth used is {max_depth}")
    print(f"Training accuracy: {train_acc}")
    print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(f"Testing accuracy: {test_acc}\n")
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)


# In[11]:


#Plot the train_acc and test_acc for various max depths
plt.figure(figsize=(10,5))
plt.plot(max_depths, train_accuracy, marker='o', markersize=6, color='blue', label='train accuracy')
plt.plot(max_depths, test_accuracy, marker='o', markersize=6, color='red', label='test accuracy')

plt.title('accuracy vs max depth')
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.xticks(max_depths)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Running the decision tree on the best value of max_depth obtained using validation accuracy

# In[12]:


clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=45)
clf = clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
print(f"max_depth used is 45")
print(f"Training accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Testing accuracy: {accuracy_score(y_test, y_test_pred)}\n")


# ##### (ii) Varying the ccp_alpha = {0.001, 0.01, 0.1, 0.2} and reporting the train and test accuracies

# In[13]:


ccp_alphas = [0.001, 0.01, 0.1, 0.2]
train_accuracy = []
test_accuracy = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
    clf = clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"ccp_alpha used is {ccp_alpha}")
    print(f"Training accuracy: {train_acc}")
    print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(f"Testing accuracy: {test_acc}\n")
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)


# In[14]:


#Plot the train_acc and test_acc for various ccp_alphas
plt.figure(figsize=(10,5))
plt.plot(ccp_alphas, train_accuracy, marker='o', markersize=6, color='blue', label='train accuracy')
plt.plot(ccp_alphas, test_accuracy, marker='o', markersize=6, color='red', label='test accuracy')

plt.title('accuracy vs ccp_alpha')
plt.xlabel('ccp_alpha')
plt.ylabel('accuracy')
plt.xticks(ccp_alphas)
plt.ylim(0,1.1)
plt.legend()
plt.show()


# ##### Running the decision tree on the best value of ccp_alpha obtained using validation accuracy

# In[15]:


clf = tree.DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001)
clf = clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)
print(f"ccp_alpha used is 0.001")
print(f"Training accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Testing accuracy: {accuracy_score(y_test, y_test_pred)}\n")


# #### Random forests

# In[17]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="onehot")
X_test, y_test = get_np_array("../Data/test.csv", encoding="onehot")
X_val, y_val = get_np_array("../Data/val.csv", encoding="onehot")


# In[24]:


best_oob_score_ = 0
best_n_estimators = None
best_max_features = None
best_min_samples_split = None

for n_estimators in [50, 150, 250, 350]:
    for max_features in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for min_samples_split in [2, 4, 6, 8, 10]:
            print(f"Parameters->(n_estimators={n_estimators}, max_features={max_features}, min_samples_split={min_samples_split})")
            clf = RandomForestClassifier(n_estimators = n_estimators, max_features=max_features, min_samples_split=min_samples_split,criterion = "entropy", oob_score=True)
            clf.fit(X_train, y_train.ravel())
            y_val_pred = clf.predict(X_val)
            print(f"validation accuracy: {accuracy_score(y_val.ravel(), y_val_pred)}")
            print("OOB(out of bag) accuracy: ",clf.oob_score_)
            print("\n")
            
            #update best
            if clf.oob_score_ > best_oob_score_:
                best_oob_score_ = clf.oob_score_
                best_n_estimators = n_estimators
                best_max_features = max_features
                best_min_samples_split = min_samples_split


# #### Running the random forest using optimal hyperparameters obtained based on OOB score

# In[25]:


print("Optimal parameters based on OOB score:")
print(f"best_oob_score_:{best_oob_score_}")
print(f"best_n_estimators:{best_n_estimators}")
print(f"best_max_features:{best_max_features}")
print(f"best_min_samples_split:{best_min_samples_split}")

clf = RandomForestClassifier(n_estimators = best_n_estimators, max_features = best_max_features, min_samples_split = best_min_samples_split,criterion = "entropy", oob_score=True)
clf.fit(X_train, y_train.ravel())
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)
print(f"Training accuracy: {accuracy_score(y_train.ravel(), y_train_pred)}")
print(f"Validation accuracy: {accuracy_score(y_val.ravel(), y_val_pred)}")
print(f"Testing accuracy: {accuracy_score(y_test.ravel(), y_test_pred)}")
print("OOB(out of bag) accuracy: ",clf.oob_score_)


# ##### Grid search to find optimal hyperparameters

# In[26]:


#making the instance
model = RandomForestClassifier()

#Hyper Parameters Set
params = {
    'n_estimators': [50, 150, 250, 350],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'min_samples_split': [2, 4, 6, 8, 10],
    'criterion' :['entropy'],
    'oob_score': [True],
    
}

grid = GridSearchCV(estimator=model, param_grid=params, verbose=2)

# Fit the grid search to the data and find best hyperparameters
grid.fit(X_train, y_train.ravel())


# ##### Running the random forest with best hyperparameters obtained using grid search

# In[27]:


# Best parameter after tuning 
print("Best parameters: ",grid.best_params_)

# Store the best model for future use 
model = grid.best_estimator_

model.fit(X_train, y_train.ravel())
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
print(f"Training accuracy: {accuracy_score(y_train.ravel(), y_train_pred)}")
print(f"Validation accuracy: {accuracy_score(y_val.ravel(), y_val_pred)}")
print(f"Testing accuracy: {accuracy_score(y_test.ravel(), y_test_pred)}")
print("OOB(out of bag) accuracy: ",model.oob_score_)


# #### Gradient Boosting Classifier

# In[37]:


X_train,y_train = get_np_array('../Data/train.csv', encoding="onehot")
X_test, y_test = get_np_array("../Data/test.csv", encoding="onehot")
X_val, y_val = get_np_array("../Data/val.csv", encoding="onehot")


# ##### Grid search to find optimal hyperparameters

# In[38]:


#making the instance
model = GradientBoostingClassifier()

#Hyper Parameters Set
params = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.1, 0.5, 1.0],
    'min_samples_split': [6, 8, 10],
    'ccp_alpha': [0.0, 0.001, 0.005]
}

grid = GridSearchCV(estimator=model, param_grid=params, verbose=2)

# Fit the grid search to the data and find best hyperparameters
grid.fit(X_train, y_train.ravel())


# ##### Running the Gradient boosting classifier with best hyperparameters obtained using grid search

# In[39]:


# Best parameter after tuning 
print("Best parameters: ",grid.best_params_)

# Store the best model for future use 
model = grid.best_estimator_

model.fit(X_train, y_train.ravel())
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
print(f"Training accuracy: {accuracy_score(y_train.ravel(), y_train_pred)}")
print(f"Validation accuracy: {accuracy_score(y_val.ravel(), y_val_pred)}")
print(f"Testing accuracy: {accuracy_score(y_test.ravel(), y_test_pred)}")


# In[ ]:




