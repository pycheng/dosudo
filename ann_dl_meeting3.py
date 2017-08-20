
# coding: utf-8

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

PATH = './db/Churn_Modelling.csv'

df = pd.read_csv(PATH)
X = df.iloc[:, 3:13].values  # get X from col 3:12
Y = df.iloc[:, 13].values    # col13 is credict


# In[3]:

#==================================================================
# pre-processing data
# problem: there's string in the data, how to process string
# solution: transfer to number by scikit learn function
# Let's try LabelEncoder
#==================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#deal with Gender column, apply labelencoder to transfer string to number
labelencoder_x_2 = LabelEncoder();

ori = df.iloc[:, 3:13].values; print(ori[5:10]);
#{Female:0, Male:1}
X[:,2] = labelencoder_x_2.fit_transform(X[:,2]);
print(X[5:10])
#print(X[:5])
#print(df[:5])
X[:,1] = labelencoder_x_2.fit_transform(X[:,1]);
print(X[5:10])


# In[4]:

#==================================================================
# problem, Why Female = 0 and Male = Female+1?
# solution: one hot encoding
#==================================================================
X_onehot = X.copy()
#one hot encoding the country
onehotencoder = OneHotEncoder(categorical_features=[1])
print(X_onehot[0:3])
X_onehot = onehotencoder.fit_transform(X_onehot)
#print(type(X_onehot)) #coo_matrix

#the one hot encoding expend before col[0]
#in this case col[0:3] one hot encoding on country 
X_onehot = X_onehot.toarray(); 

print(X_onehot[0:3])
#==================================================================
# Dummy Variable Trap
# ML assume features are independent, or somehow the weighting is accumulated
# collinearity: predict one feature by others
# E.g., if we one hot encoding Male & Female, increasing dimention but no gain
# problem: !(Spain & Germany) = France
# solution: remove France
# What is dummy variable? --> familiar features or co-relation by PCA
#==================================================================
X_onehot_DVT = X_onehot[:, 1:]
print(X_onehot_DVT[0:3])


# In[5]:

#==================================================================
# Varification
# Seperate as training, validation, testing
#==================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_onehot_DVT, Y, test_size = 0.2, random_state = 0)

print(len(X_onehot_DVT), len(Y))
print(len(x_train), len(y_train), len(x_test), len(y_test))
#==================================================================
# Standardization
# scaling your data to prevent domination feature
# http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
#==================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)
print(x_train[:5]); print(x_train_std[:5])


# In[6]:

cf = Sequential();
#units: hidden layer node 
#input_dim: input data column count
cf.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim = 11))
cf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
cf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#adam is a SGD
cf.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


# In[11]:

cf.fit(x_train_std, y_train, batch_size=10, epochs=100)


# In[13]:

# Predicting the Test set results
y_pred = cf.predict(x_test_std)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy:', accuracy_score(y_test, y_pred))


# In[17]:

#Assignment
X = df.iloc[:, 3:13].values  # get X from col 3:12
Y = df.iloc[:, 13].values    # col13 is credict
## Deal with gender 
# Label encoder: helps you convert string label => numerical label 
labelencoder_X_2 = LabelEncoder()  
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
## Deal with country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

user = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
user[:, 1] = labelencoder_X_1.transform(user[:, 1])
user[:, 2] = labelencoder_X_2.transform(user[:, 2])
user = onehotencoder.transform(user).toarray()
user = user[:, 1:]
user = sc.transform(user)
print(user)
print((cf.predict(user) > 0.5))


# In[2]:

# Part 4 - Evaluating, Improving the Tuning the ANN
# Evaluateing the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 100)

# 10-fold is recommended
# set n_jobs = 1 because it doesn't work well for parallelizing in this notebook 
accuracies = cross_val_score(estimator=classifier, X=x_train_std, y=y_train, cv=10, n_jobs = 1)
print(accuracies)


# In[ ]:

print(accuracies)
mean = accuracies.mean()
variance = accuracies.std()
print('mean:', mean, 'std', variance)


'''
print(user)
[ 0.8375      0.84249999  0.87499999  0.84625     0.87249999  0.85625
  0.83625     0.82874999  0.83749999  0.84499999]
print((cf.predict(user) > 0.5))
[ 0.8375      0.84249999  0.87499999  0.84625     0.87249999  0.85625
  0.83625     0.82874999  0.83749999  0.84499999]

print('mean:', mean, 'std', variance)
    mean: 0.847749994658 std 0.0147245539459
'''
