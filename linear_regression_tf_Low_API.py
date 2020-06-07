# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:27:55 2020
low-level tf API: implement the back-prop using tf.GradientTape()
@author: lwang
"""
import time
import tensorflow as tf
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers

import matplotlib
print(matplotlib.__version__)
#%%
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

#%% make data pipeline with generator function, i.e., yield involved 
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #indices is shuffled
    for i in range(0, num_examples, batch_size):
        #print('i',i)
        indexs = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features,indexs), tf.gather(labels,indexs)
        
# test the pipeline   
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)
# for _ in range(10):
#     features, labels =next(data_iter(X,Y,batch_size))
#     print(labels)

#%% define model
w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0,dtype = tf.float32))

# 定义模型
class LinearRegression:     
    #正向传播
    def __call__(self,x): 
        return x@w + b

    # 损失函数
    def loss_func(self,y_true,y_pred):  
        return tf.reduce_mean((y_true - y_pred)**2/2)

model = LinearRegression()

#%% define train step 
# if 使用动态图调试: without @tf.function
@tf.function # will be 2x faster, but debug is not possible anymore
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)
    # 反向传播求梯度
    dloss_dw,dloss_db = tape.gradient(loss,[w,b])
    # 梯度下降法更新参数
    w.assign(w - 0.001*dloss_dw)
    b.assign(b - 0.001*dloss_db)
    
    return loss, w, b

# test one train_step
batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))
train_step(model,features,labels)

#%% define train loop
def train_model(model,epochs):
    loss_list, w_list, b_list = [], [], []
    for epoch in tf.range(1,epochs+1):
        for features, labels in data_iter(X,Y, batch_size=10):
            loss, w, b = train_step(model,features,labels)
            loss_list.append(loss.numpy())
            w_list.append(w.numpy())
            b_list.append(b.numpy()[0])

        if epoch%50==0:
            printbar()
            tf.print("epoch =",epoch,"loss = ",loss)
            tf.print("w =",w)
            tf.print("b =",b)

    return loss_list, w_list, b_list

loss_list, w_list, b_list = train_model(model,epochs = 50)

#%% saved lists --> np.array
# Option 1: np.concatenate()
start_time = time.time() 
arr = w_list[0]
for i in range(1, len(w_list)):
    arr = np.concatenate((arr, w_list[i]), axis=1)
elapsed_time = time.time() - start_time
print('elapsed_time:', elapsed_time)    
# Option 2: np.hstack()  <<< 3x faster
start_time = time.time() 
w_array = np.hstack(w_list) # Stack arrays in sequence horizontally
elapsed_time = time.time() - start_time
print('elapsed_time:', elapsed_time)
# check equality
tem_s = np.sum((arr - w_array)**2)
print(tem_s)


add_1dim = lambda x: np.reshape(x,(1, x.size))

loss_arr = np.hstack(loss_list)
x, y, z = w_array[0,:], w_array[1,:], loss_arr
# x = add_1dim(x)
# plt.figure()
# plt.plot(x.T) # plot each column of a 2D array by default
    
#%% plot training details
from mpl_toolkits.mplot3d import Axes3D

# Fig.1. w, b vs loss
fig = plt.figure(figsize = (15,5))
# ax1 = plt.subplot(121)
# plt.scatter(b_list, loss_list)
# plt.grid(True)
# plt.yscale('symlog')
# plt.xlabel("b")
# plt.ylabel("loss")
ax2 = plt.subplot(121)
ax2.plot(loss_list,"-r",linewidth = 1.0,label = "loss")
ax2.plot(b_list,"-b",linewidth = 2.0,label = "b")
ax2.plot(x,"-.b",linewidth = 2.0, label = "w1")
ax2.plot(y,"--b",linewidth = 2.0, label = "w2")
ax2.legend()
plt.yscale('symlog')    
plt.grid(True)
plt.xlabel("Step")
plt.ylabel("Value")

ax = plt.subplot(122, projection='3d')                   
ax.scatter(x, y, np.log(z))
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('loss (log)')

#%% demo result
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
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
