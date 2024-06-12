import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def drawScatter(x,y,xlabel,ylabel,data_name,plot_labels = None):     
    plt.clf()

    # グラフの軸ラベル等の設定
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # plot_labelsが指定されない場合は，1～データ数の数字にする
    if plot_labels is None:
        plot_labels = range(1,x.shape[0]+1)
        
    # 散布図を描画
    for (i,j,k) in zip(x,y,plot_labels):
        plt.plot(i,j,'o')
        plt.annotate(k, xy=(i, j))



    # プロットを保存（plt.show()の後だと真っ白になるので注意）
    plt.savefig(data_name + ".png")

    plt.show()



data = np.loadtxt("lec06_data(14).csv", delimiter=",", dtype="unicode")
N = data[:,0].size
M = data[0,:].size

A = np.loadtxt("lec06_data(14).csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14))
Fx = A.sum(axis=0)
Fy = A.sum(axis=1)
B = np.diag(Fx)
C = np.diag(Fy)
Bri = np.diag(Fx**(-0.5))
Ci = np.diag(1/Fy)
H = Bri@ A.T@ Ci@ A@ Bri


w,v = LA.eig(H)
sort_index = np.argsort(w)[::-1]

sort_w = w[sort_index]
sort_v = v[:,sort_index]

x1 = Bri@v[:, sort_index[1]]
x2 = Bri@v[:, sort_index[2]]

y1 = (1/w[sort_index[1]]**(1/2))*Ci@A@x1
y2 = (1/w[sort_index[2]]**(1/2))*Ci@A@x2

drawScatter(x1, x2, "x1", "x2", "x", plot_labels=data[1:N, 15])
drawScatter(y1, y2, "y1", "y2", "y", plot_labels=data[0, 1:M-1])
