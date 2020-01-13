
import numpy as np
import matplotlib.pyplot as plt
import pprint
import cvxopt
from cvxopt import matrix, solvers
def svm(X1,X2):
    n1 = len(X1)
    n2 = len(X2)
    P = matrix([[0.,0,0],[0,1,0],[0,1,0]])
    q = matrix([0.,0,0])
    one1 =np.ones((n1,1))
    W1 = np.hstack((one1,X1))
    one2 =np.ones((n2,1))
    W2 = -1*np.hstack((one2,X2))
    n = n1 + n2 
    G = -1*matrix(np.vstack((W1,W2)))
    h = -1*matrix(np.ones((n,1)))
    re = cvxopt.solvers.qp(P,q,G,h)
    return re['x']
###############################################
###############################################
ch0= np.array([
        [121.4,34.5],#上海
        [117.2,39.1],#天津
        [114.1,22.2],#香港
        [120.2,30.3],#杭州
        [118.1,24.5],#厦门
        [121.3,25.0] #台北
        ])
ch1 = np.array([
        [116.41667,39.91667],#北京
        [106.45000, 29.56667],#重庆
        [104.06667,30.66667],#成都
        [114.31667,30.51667]#武汉
        ])

jp0= np.array([
        [128, 26],#冲绳
        [132, 34],#广岛
        [136, 35],#桑明
        ])
jp1 = np.array([
        [135.3,34.4],   #大阪
        [132.27,34.24], #广岛
        [135.5,34.41], #奈良
        [139.46,35.42], #东京
        ])

###################################
plt.scatter(ch0[:,0],ch0[:,1])
plt.scatter(jp0[:,0],jp0[:,1])
plt.plot(123,25,color='g',marker='x')
#####################################
re = svm(ch0,jp0) 
b = re[0]
w1 = re[1]
w2 = re[2]
print('w1:',round(w1,6),' w2:',round(w2,6),' b:',round(b,6))
xx = np.linspace(123,125.5)
zz = -w1/w2*xx - b/w2
plt.plot(xx, zz)
plt.show()