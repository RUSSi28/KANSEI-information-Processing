# -*- coding: utf-8 -*-
"""34714086高田恒輝

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DBL1DZ3Ub3r-Bfb7nTCKRGXth19__A2R
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic']

x = np.loadtxt("lec03_data.csv", delimiter=",")
d = np.size(x[0:1,:])
N = np.size(x[:,0:1])


# 列ごと平均値を求める
m = x.mean(axis=0)
s = x.std(axis=0, ddof=1)
SD = (x-m)/s
print(SD)


# データの相関係数行列を計算
R = np.cov(SD, rowvar=False, ddof=1)


# 行列Sの固有値と固有ベクトルを求める(変数の数分求められる)
w, v = LA.eig(R)
v0 = v[:,0:1]
v1 = v[:,1:2]
v2 = v[:,2:3]
v3 = v[:,3:]
w0 = w[0]
w1 = w[1]
w2 = w[2]
w3 = w[3]

# 得られたすべての主成分について、第1から順に、
# 式・寄与率・累積寄与率を出力するように作成せよ。 資料のものと同様の主成分が得られた。
print("式")
print("z1 = %f x1 + %f x2 + %f x3 + %f x4"%(v0[0], v0[1], v0[2], v0[3]))
print("z2 = %f x1 + %f x2 + %f x3 + %f x4"%(v1[0], v1[1], v1[2], v1[3]))
print("z3 = %f x1 + %f x2 + %f x3 + %f x4"%(v2[0], v2[1], v2[2], v2[3]))
print("z4 = %f x1 + %f x2 + %f x3 + %f x4"%(v3[0], v3[1], v3[2], v3[3]))

contribution_ratio = 0
for i in range(d):
  contribution_ratio += w[i]
cr0 = w0/contribution_ratio
cr1 = w1/contribution_ratio
cr2 = w2/contribution_ratio
cr3 = w3/contribution_ratio
print("寄与率")
print("第1主成分の寄与率:%f"%cr0)
print("第2主成分の寄与率:%f"%cr1)
print("第3主成分の寄与率:%f"%cr2)
print("第4主成分の寄与率:%f"%cr3)
print("累積寄与率")
k = 0
for i in range(d-2):
  k += w[i]
  print("第%d主成分の累積寄与率:%f"%(i+1, k/contribution_ratio))

xy = SD@v

x = xy[:, 0]
y = xy[:, 1]


####################################
# 散布図を描画する関数
# x,y: データ
# xlabel : x軸のラベル
# ylabel : y軸のラベル
####################################
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

    
    
####################################
# メイン関数
####################################    
if __name__ == "__main__":
      

    # 主成分得点のプロットを描画   
    data_name = "lec03_sdata"
    file_name = data_name + ".csv"
    drawScatter(x, y ,"PC1","PC2",data_name)   

    # 因子負荷量のプロットを描画
    data_name = "lec03_fdata"
    file_name = data_name + ".csv"
    f1 = v[:,0]*(w[0]**0.5)
    f2 = v[:,1]*(w[1]**0.5)
    drawScatter(f1, f2, "f1", "f2", data_name)



    
    
    


