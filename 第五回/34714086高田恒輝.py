import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

def drawScatter(x,y,xlabel,ylabel,data_name):     
    plt.clf()

    # グラフの軸ラベル等の設定
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # 散布図を描画
    for (i,j,k) in zip(x,y,range(1,x.shape[0]+1)):
        plt.plot(i,j,'o')
        plt.annotate(k, xy=(i, j))



    # プロットを保存（plt.show()の後だと真っ白になるので注意）
    plt.savefig(data_name + ".png")

    plt.show()


A = np.loadtxt("sample.csv", delimiter=",")
print(A)
B = np.array([np.sum(A[:,0]), np.sum(A[:,1]), np.sum(A[:,2])])
_B = np.diag(B**(-(1/2)))
C = np.array([np.sum(A[0,:]), np.sum(A[1,:]), np.sum(A[2,:]), np.sum(A[3,:]), np.sum(A[4,:])])
_C = np.diag(C**(-1))



H = _B@A.T@_C@A@_B
w, v = LA.eig(H)

X_2 = _B@v[:,2]
Y_2 = 1/(w[2]**(1/2))*_C@A@X_2
X_3 = _B@v[:,1]
Y_3 = 1/(w[1]**(1/2))*_C@A@X_3
print(X_2)
print(X_3)
print(Y_2)
print(Y_3)


drawScatter(X_2, X_3, "x2", "x3", "x")
drawScatter(Y_2, Y_3, "y2", "y3", "y")
print(_B)
print(H)
print(w)
print(X_3)
