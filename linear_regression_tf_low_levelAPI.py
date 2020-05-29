# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:00:09 2020

@author: lwang
"""

import tensorflow as tf
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers

#%%
@tf.function #Compiles a function into a callable TensorFlow graph (to speed up)
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+2,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)
    
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

#%% data pipe
ds = tf.data.Dataset.from_tensor_slices((X,Y)).shuffle(buffer_size = 100).batch(10) \
     .prefetch(tf.data.experimental.AUTOTUNE)  
  
#%% define model
model = layers.Dense(units = 1) 
model.build(input_shape = (2,)) #用build方法创建variables
model.loss_func = losses.mean_squared_error
model.optimizer = optimizers.SGD(learning_rate=0.001)

#%% fit
@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    grads = tape.gradient(loss,model.variables)
    model.optimizer.apply_gradients(zip(grads,model.variables))
    return loss

# test train_step on one step
features,labels = next(ds.as_numpy_iterator())
loss = train_step(model,features,labels)
print(loss)

#%%
def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model,features,labels)
        if epoch%50==0:
            printbar()
            tf.print("epoch =",epoch,"loss = ",loss)
            tf.print("w =",model.variables[0])
            tf.print("b =",model.variables[1])
            
train_model(model,epochs = 200)


#%% demo result
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

w, b = model.variables
tf.print("w =", w)
tf.print("b =", b)
            
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
