
# coding: utf-8

# In[1]:

from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import time


model = Sequential();
model.add(Dense(1, input_dim = 2))
model.summary()


# In[22]:

X_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [0, 1, 1, 0] #XOR
#y_train = [0, 0, 0, 1] #AND
model.compile(optimizer = 'rmsprop',
             loss = 'mean_squared_error',
             metrics=['accuracy'])
#for e in range(1000):
epoch = 10
model.fit(X_train, y_train, epochs=epoch, verbose=0)
score = model.evaluate(X_train, y_train)
print(score)    


# In[ ]:



