# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:14:59 2021

@author: csjh9
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

"""激活函數sigmoid"""
def sigmoid(z):
    result = 1/(1+math.exp(z[0]*(-1)))
    resultArr = [result]
    return resultArr

"""向前傳遞，計算出輸出結果與對應目標之間誤差"""
def feedforward(X,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21):
    z11=np.add(np.dot(np.transpose(W11),X),b11)
    a11=sigmoid(z11)

    z12=np.add(np.dot(np.transpose(W12),X),b12)
    a12=sigmoid(z12)

    z21=np.add(np.dot(np.transpose(W21),[a11[0],a12[0]]),b21)
    a21=sigmoid(z21)

    return W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21

"""訓練函數，透過訓練資料訓練權重"""
def train(trainData,learning_rate,momentunm,epochs,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21,trainingLossArr):
    for i in range(epochs):
        for j in trainData:
            temp = j
            X = np.transpose(temp[:2])
            y = [temp[2]]
            W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21 = feedforward(X,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21)
            """Loss function"""
            trainingLoss = (-1) * (y[0] * math.log(a21[0]) + (1-y[0])*math.log(1-a21[0]))
            trainingLossArr.append(trainingLoss)

            """更新權重"""
            #計算W11
            Delta_W11_previous = np.array([0,0])
            Delta_W11 = np.add(np.multiply(np.multiply(np.multiply(np.multiply((-1)*learning_rate*(W21[0]),
                np.subtract(a21,y)),a11),np.subtract([1],a11)),X),np.multiply(momentunm,Delta_W11_previous))
            #Delta_W11 = (-1)*learning_rate*(W21[0])*(a21-y)*a11*(1-a11)*X + momentunm*Delta_W11_previous
            W11 = np.add(W11,Delta_W11)
            Delta_W11_previous = Delta_W11
            #計算b11
            Delta_b11_previous = [0]
            Delta_b11 = np.add(np.multiply(np.multiply(np.multiply(np.multiply((-1)*learning_rate*(W21[0]),
                np.subtract(a21,y)),a11),np.subtract([1],a11)),1),np.multiply(momentunm,Delta_b11_previous))
            #Delta_b11 = (-1)*learning_rate*(W21[0])*(a21-y)*a11*(1-a11)+ momentunm*Delta_b11_previous
            b11 = np.add(b11,Delta_b11)
            Delta_b11_previous = Delta_b11
            #計算W12
            Delta_W12_previous = np.array([0,0])
            Delta_W12 = np.add(np.multiply(np.multiply(np.multiply(np.multiply((-1)*learning_rate*(W21[1]),
                np.subtract(a21,y)),a12),np.subtract([1],a12)),X),np.multiply(momentunm,Delta_W12_previous))
            #Delta_W12 = (-1)*learning_rate*(W21[1])*(a21-y)*a12*(1-a12)*X + momentunm*Delta_W12_previous
            W12 = np.add(W12,Delta_W12)
            Delta_W12_previous = Delta_W12
            #計算b12
            Delta_b12_previous = [0]
            Delta_b12 = np.add(np.multiply(np.multiply(np.multiply(np.multiply((-1)*learning_rate*(W21[1]),
                np.subtract(a21,y)),a12),np.subtract([1],a12)),1),np.multiply(momentunm,Delta_b12_previous))
            #Delta_b12 = (-1)*learning_rate*(W21[1])*(a21-y)*a12*(1-a12) + momentunm*Delta_b12_previous
            b12 = np.add(b12,Delta_b12)
            Delta_b12_previous = Delta_b12            
            #計算W21
            Delta_W21_previous = np.array([0,0])
            Delta_W21 = np.add(np.multiply(np.multiply(((-1)*learning_rate),np.subtract(a21,y)),[a11[0],a12[0]]),
                np.multiply(momentunm,Delta_W21_previous))
            #Delta_W21 = (-1)*learning_rate*(a21-y)*a1 + momentunm*Delta_W21_previous
            W21 = np.add(W21,Delta_W21)
            Delta_W21_previous = Delta_W21
            #計算b21
            Delta_b21_previous = [0]
            Delta_b21 = np.add(np.multiply(np.multiply(((-1)*learning_rate),np.subtract(a21,y)),1),
                np.multiply(momentunm,Delta_b21_previous))
            #Delta_b21 = (-1)*learning_rate*(a21-y) + np.multiply(Delta_b21_previous,momentunm)
            b21 = b21 + Delta_b21
            Delta_b21_previous = Delta_b21

    return W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21,trainingLossArr

"""
測試函數
testingData: XOR_training.txt
"""
def test(testData,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21):
    for i in testData:
        temp = i
        X = np.transpose(temp[:2])
        #y = [temp[2]]
        W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21 = feedforward(X,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21)
        print(i[0]," ",i[1]," ",a21)

if __name__ == '__main__':
    #讀取訓練資料和測試資料
    trainData = np.genfromtxt("XOR_training.txt", delimiter='\t')
    testData = np.genfromtxt("XOR_testing.txt", delimiter='\t')

    #設定epochs為9000
    epochs = 9000
    #設定learning_rate為0.05
    learning_rate = 0.05
    #設定momentunm為0.85
    momentunm = 0.85

    #Neuron1
    W11 = np.random.rand(2, )
    b11 = [random.random()]
    z11 = [0]
    a11 = [0]

    #Neuron2
    W12 = np.random.rand(2, )
    b12 = [random.random()]
    z12 = [0]
    a12 = [0]

    #Neuron3
    W21 = np.random.rand(2, )
    b21 = [random.random()]
    z21 = [0]
    a21 = [0]

    trainingLossArr = []
    instancesNumberArr = []
    for i in range(trainData.shape[0] * epochs):
        instancesNumberArr.append(i)    

    W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21,trainingLossArr = train(trainData,
        learning_rate,momentunm,epochs,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21,trainingLossArr)
    
    print("訓練後權重:")
    print("Neuron1(隱藏層):", W11[0], " ", W11[1])
    #print("Bias1:", b11[0])
    print("Neuron2(隱藏層):", W12[0], " ", W12[1])
    #print("Bias2:", b12[0])
    print("Neuron3:", W21[0], " ", W21[1])
    #print("Bias3:", b21[0])

    print("\n測試結果:")
    test(testData,W11,b11,z11,a11,W12,b12,z12,a12,W21,b21,z21,a21)

    plt.plot(instancesNumberArr,trainingLossArr)
    plt.show()
    input()