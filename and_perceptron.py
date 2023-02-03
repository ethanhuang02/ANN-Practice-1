# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:13:48 2021

@author: csjh9
"""
import numpy as np
import matplotlib.pyplot as plt

"""sign function"""
def sign(C):

    if C>=0:
        return 1
    else:
        return 0
    
"""訓練函數，透過訓練資料訓練權重"""    
def train(epochs, learning_rate, trainingData, W):
    #nonErrorTimes: 若此epoch不存在error，則nonErrorTimes +1
    nonErrorTimes=0

    for i in range(epochs):
        #error:目標值與預測值差距
        error=0
        
        for j in trainingData:                
                temp = j                
                X = temp[:2]
                arr = [-1]
                X = np.concatenate((arr,X))
                t = temp[2]
                C = np.dot(X,np.transpose(W))
                #預測值y
                y = sign(C)
                #W1 = W1 + learning_rate * (目標值 - 預測值) * X1
                W = W + (learning_rate*(t-y))*X            
                #若存在error，error+1
                if (t-y)!=0:
                    error=error+1
        #若此epoch不存在error，則nonErrorTimes +1
        if error==0:
            nonErrorTimes=nonErrorTimes+1
        #若有100個epochs不存在error，結束訓練
        if nonErrorTimes==100:
            break

    return W

"""
測試函數
testingData: AND_training.txt
W: 在訓練函數中訓練的W
"""
def test(testingData,W):

    instances=0
    errorInstances=0
    accuracy=0

    for i in testingData:
        instances=instances+1
        temp=i
        X = temp[:2]
        arr = [-1]
        X = np.concatenate((arr, X))
        t=temp[2]
        C=np.dot(X,np.transpose(W))
        #預測值y
        y=sign(C)
        #若預測值與目標值不同，errorInstances +1
        if(t-y)!=0:
            errorInstances=errorInstances+1
    #calculate ratio of accurate insatances
    accuracy = 1 - (errorInstances / instances)

    return accuracy

if __name__ == "__main__":
    #讀取訓練資料和測試資料
    trainingData = np.loadtxt("AND_training.txt", delimiter="	", unpack=False)
    testingData = np.loadtxt("AND_testing.txt",delimiter="	",unpack=False)

    #設定epochs為9000
    epochs = 9000
    #設定learning_rate為0.15
    learning_rate = 0.15

    Originalweights = np.random.rand(3)

    #算權重
    W = train(epochs, learning_rate, trainingData, Originalweights)

    #進行測試，算準確率
    accuracy=test(testingData,W)
    print("W0:",W[0])
    print("W1:",W[1])
    print("W2:",W[2])
    print("Total accuracy is: ",accuracy)

    #畫圖
    plt.plot([0,(W[0]/W[1])],[(W[0]/W[2]),0])
    plt.plot([0,0,1,1],[1,0,1,0],'ro')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis([0,2,0,2])
    plt.show()
    input()