# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:21:15 2020
linear regression using tf
@author: lwang
"""

import tensorflow as tf
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers

    
#%%  generate data      
n = 400 #sample size

# 生成测试用数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
X_np = X.numpy()
w0 = tf.constant([[2.0],[-3.0]]) # shape=(2, 1)
b0 = tf.constant([[3.0]]) #shape=(1, 1)
print(w0)
tf.print(tf.rank(w0))
print(b0)
tf.print(tf.rank(b0))

Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)
Y_np = Y.numpy()

# plot data
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
plt.figure(figsize = (12,5))

ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b")
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g")
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()

#%% define model
model = models.Sequential()
model.add(layers.Dense(1,input_shape =(2,)))
model.summary()

#%% fit
tf.keras.backend.clear_session()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
# Launch TensorBoard from the command line: tensorboard --logdir= path_to_your_logs

model.compile(optimizer="adam",loss="mse",metrics=["mae"])
model.fit(X,Y,batch_size = 10,epochs = 100, callbacks=[tensorboard_callback])  

tf.print("w = ",model.layers[0].kernel)
tf.print("b = ",model.layers[0].bias)
w1 = model.layers[0].kernel

#%% demo result
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

w, b = model.variables

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.plot(X[:,0],w[0]*X[:,0]+b[0],"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.plot(X[:,1],w[1]*X[:,1]+b[0],"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()







