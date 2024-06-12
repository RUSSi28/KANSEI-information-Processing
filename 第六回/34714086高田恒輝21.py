import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


with open("lec06_data(21).csv", "r", encoding="shift-jis") as file:
    data = np.loadtxt(file, delimiter=',', dtype=str)

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


A = data[1:, 1:21].astype(int)

B = np.sum(A, axis=0)
B_sqrt = 1/np.sqrt(B)
C = np.sum(A, axis=1)
C_inv = 1/C

I_B = np.identity(np.shape(A)[1])
I_C = np.identity(np.shape(A)[0])
B_sqrt = B_sqrt*I_B
C_inv = C_inv*I_C

H = B_sqrt @ A.T @ C_inv @ A @ B_sqrt
print(H)
w, v = np.linalg.eig(H)

sort_index = np.argsort(w)[::-1]
sort_index = sort_index[1:]
sort_w = w[sort_index]
sort_v = v[sort_index]
print(B_sqrt)
print(sort_v)

x_score = B_sqrt @ sort_v
y_sample = (C_inv)








# data = np.loadtxt("lec06_data(21).csv", delimiter=",", dtype="unicode")
# N = data[:,0].size
# M = data[0,:].size

# df = pd.read_excel('lec06_data14_analyze.xlsx')

# A = np.array(data[1:,1:21], dtype=int)
# Fx = A.sum(axis=0)
# Fy = A.sum(axis=1)
# B = np.diag(Fx)
# C = np.diag(Fy)
# Bri = np.diag(Fx**(-0.5))
# Ci = np.diag(1/Fy)
# H = Bri@A.T@Ci@A@Bri
# print(H)



# w,v = LA.eig(H)
# sort_index = np.argsort(w)[::-1]

# sort_w = w[sort_index]
# sort_v = v[:,sort_index]

# x1 = Bri@v[:, sort_index[1]]
# x2 = Bri@v[:, sort_index[2]]

# y1 = (w[sort_index[1]]**(-0.5))*(Ci@A@x1)
# y2 = (w[sort_index[2]]**(-0.5))*(Ci@A@x2)

# drawScatter(x1, x2, "x1", "x2", "x_", plot_labels=data[1:N, 22])
# drawScatter(y1, y2, "y1", "y2", "y_", plot_labels=data[0, 1:M-1])
# print(N)
# print(data[1:N, 22])
# print(data[0, 1:M-1])