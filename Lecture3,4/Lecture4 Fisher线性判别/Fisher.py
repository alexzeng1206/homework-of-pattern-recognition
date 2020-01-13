
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle

###############################################
num, dim=200,2
np.random.seed(0)
x2=np.random.randn(num, dim)
C=[[1,0],[0,1]]
W = [5,0]
X2=np.dot(x2,C)+W
plt.scatter(X2[:,0],X2[:,1])
num, dim=200,2
np.random.seed(0)
x2_=np.random.randn(num, dim)
C=[[1,0],[0,1]]
W = [-5,0]
X2_= np.dot(x2_,C)+W
plt.scatter(X2_[:,0],X2_[:,1])
X = np.concatenate((X2, X2_), axis=0)
#X, y = shuffle(X, y)
#####################################
m1=np.array([0.0,0.0])
m2=np.array([0.0,0.0])
Sw1=np.array([0.,0.,0.,0.])
Sw2=np.array([0.,0.,0.,0.])
Sw1=Sw1.reshape(2,2)
Sw2=Sw2.reshape(2,2)
for i in range(200):
    m1+=X[i]/200
    m2+=X[i+200]/200
for i in range(200):
        #####################
    t1=(X[i]-m1).reshape(2,1)
    t2=(X[i]-m2).reshape(2,1)
    p=np.dot(t1,t1.T)
    Sw1+=p
    p=np.dot(t2,t2.T)
    Sw2+=p
        ################################
Sw=Sw1+Sw2
Sw_=np.linalg.pinv(Sw)
m1=m1.reshape(2,1)
m2=m2.reshape(2,1)
w=np.dot(Sw_,(m1-m2))
w=w.reshape(-1)
#print(w)
#print(X2[1])
#####################################
m1=m1.reshape(-1)
m2=m2.reshape(-1)
m1=np.dot(w,m1)
m2=np.dot(w,m2)
#print(m1)
#print(m2)
#print(yt)
#################################
yt=(m1+m2)/2
print("投影方向",w)
print("阈值：",yt)
sum2=0
for i in range(200):
    if m1>m2:
        p=1
    else:
        p=-1
    k=np.dot(w,X2[i])
   # print(k)
    if (k-yt)*p>0:
        sum2+=1
    k=np.dot(w,X2_[i])
    if (k-yt)*p<0:
        sum2+=1
#print("性能(误差):",(400-sum2)/400)
print("正确率:",(sum2)/400)
########################################
x=(X2_.sum(axis=0)+X2.sum(axis=0))/400
x0=x[0]
y0=x[1]
xx = np.linspace(-0.3,0.3)
zz = -w[0]/w[1]*(xx - x0)+y0
plt.plot(xx, zz)
plt.show()