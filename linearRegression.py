import pandas as pd
import numpy as np

def costFunc(X,Y,theta):
    inner = np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,Y,theta,alpha,iters,*args):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaNums = int(theta.shape[1])
    print(thetaNums)
    for i in range(iters):
        error = (X*theta.T-Y)
        for j in range(thetaNums):
            derivativeInner = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (alpha*np.sum(derivativeInner)/len(X))
            if temp[0,j]<=args[j]:
                temp[0,j]=args[j]

        theta = temp
        cost[i] = costFunc(X,Y,theta)

    return theta,cost


#learningRate学习率，Loopnum迭代次数
def liner_Regression(data_x,data_y,learningRate,Loopnum,alpha=1.0):
    Weight=np.ones(shape=(1,data_x.shape[1]))
    rr = np.zeros((data_y.shape[0],1))
    for i in range(len(rr)):
        rr[i][0] = alpha**(i/3)
    old_loss = 0.0
    #baise=np.array([[1]])
    for num in range(Loopnum):
        WXPlusB = np.dot(data_x, Weight.T) #+ baise
        loss=np.dot((data_y-WXPlusB).T,rr*(data_y-WXPlusB))/data_y.shape[0]
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
        #baise_gradient = -2*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))/data_x.shape[0]
        Weight=Weight-learningRate*w_gradient
        # for i in range(data_x.shape[1]):
        #     if Weight[0][i]<=0:
        #         Weight[0][i] = 0
        #baise=baise-learningRate*baise_gradient
        if num%100000==0:
            print(loss)#每迭代100000次输出一次loss

        if abs(loss-old_loss)<0.0000001:
            old_loss = loss
            break
        old_loss = loss
    print('Done! The loss is :%f'%loss)
    return Weight,loss

