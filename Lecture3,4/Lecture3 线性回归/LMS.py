
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle

def LMSalg(X,y):
    k = X.shape[0]
    np.random.seed()
    w_init=np.random.randn(3)
    #w_init=np.array([-3,2,5])
    w = w_init
    x1 = np.ones([k,1])
    X = np.concatenate((x1, X), axis=1)
   # for i in range(400):
   #    if float(y[i])==1.0:
   #         X[i]=-X[i]
    ###################################
    mark = 0
    sum1 = 0
    init = 0
    print("初始权值:",w)
    n=0.0001
    e=0.01
    w=w.reshape(3,1)
    while not mark:
        p=np.dot(X,w)
        p=p-y
        E=np.dot(X.T,p)
        w1=w-n*E
        dist = np.linalg.norm(w1 - w)
        w=w1
        sum1+=1
        if sum1>10000:
            break
        if dist<=0.001:
            break
    w=w.reshape(-1)
    return w

###############################################
num, dim=200,2
np.random.seed(0)
x2=np.random.randn(num, dim)
C=[[1,0],[0,1]]
W = [-2,0]
X2=np.dot(x2,C)+W
plt.scatter(X2[:,0],X2[:,1])
num, dim=200,2
np.random.seed(0)
x2_=np.random.randn(num, dim)
C=[[1,0],[0,1]]
W = [2,0]
X2_= np.dot(x2_,C)+W
plt.scatter(X2_[:,0],X2_[:,1])
X = np.concatenate((X2, X2_), axis=0)
y1 = np.linspace(1,1,200)
y1 = y1.reshape([200,1])
y2 = np.linspace(-1,-1,200)
y2 = y2.reshape([200,1])
y = np.concatenate((y1, y2), axis=0)
#X, y = shuffle(X, y)
#####################################
w=LMSalg(X,y)
print("最终权值:",w)
sum2=0
x1 = np.ones([400,1])
X = np.concatenate((x1, X), axis=1)
#for i in range(400):
#        if int(y[i])==1:
#            X[i]=-X[i]
sum3=0
for i in range(400):
    y0=np.dot(w,X[i])
    e=(float(y[i])-y0)**2
    #e=e/400
    sum3+=e
for i in range(400):
     m = np.dot(w, X[i])
     m= m*int(y[i])
     if m<=0:
            sum2+=1
print("回归性能(最小平方误差):",sum3/400)
print("分类正确率:",(400-sum2)/400)
xx = np.linspace(-0.3,0.3)
zz = -w[1]/w[2]*xx - w[0]/w[2]
plt.plot(xx, zz)
plt.show()