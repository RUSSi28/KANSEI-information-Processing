import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import copy
plt.rcParams['font.family'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic']
    


####################################
# 散布図を描画する関数(プロット点のラベルを変更できる版)
# x,y: データ
# xlabel : x軸のラベル
# ylabel : y軸のラベル
# data_name : 画像保存の時のファイル名に利用
# plot_labels : プロットされる点に付けるラベル
#               （データ数文のリスト．指定しない場合は1～nの数字になる）
####################################
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

####################################
# テキストファイルを読み込む関数
# fn: ファイル名
####################################
    
def loadText(fn):     
    f = open(fn, 'r')
    
    text = f.readlines() # １行ごとリストになる
    
    f.close()
    return text
    

####################################
# SD
#
####################################
D = np.loadtxt("lec04_data_nagoya.csv", delimiter=",")
D5 = D[:,[False, True, True, True, True,True]]
for i in range(5):
   for j in range(15):
      D5[j][i] /= D[j][0]

def mySD6(D):
    d = np.size(D[0:1,:])


    # 列ごと平均値を求める
    m = D.mean(axis=0)
    s = D.std(axis=0, ddof=1)
    SD = (D-m)/s


    # データの相関係数行列を計算
    R = np.cov(SD, rowvar=False, ddof=1)
    print(R)


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
    print(w)
    print(v)

    # 得られたすべての主成分について、第1から順に、
    # 式・寄与率・累積寄与率を出力するように作成せよ。 資料のものと同様の主成分が得られた。
    print("式")
    print("z1 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5 + %f x6"%(v0[0], v0[1], v0[2], v0[3], v0[4], v0[5]))
    print("z2 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5 + %f x6"%(v1[0], v1[1], v1[2], v1[3], v1[4], v1[5]))
    print("z3 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5 + %f x6"%(v2[0], v2[1], v2[2], v2[3], v2[4], v2[5]))
    # print("z4 = %f x1 + %f x2 + %f x3 + %f x4"%(v3[0], v3[1], v3[2], v3[3]))

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
    print(xy)
    f1 = v[:,0]*(w[0]**0.5)
    f2 = v[:,1]*(w[1]**0.5)
    print(f1[2])
    drawScatter(f1, f2, "f1", "f2", data_name)
    return (xy)


def mySD5(D):
    d = np.size(D[0:1,:])


    # 列ごと平均値を求める
    m = D.mean(axis=0)
    s = D.std(axis=0, ddof=1)
    SD = (D-m)/s
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
    print(w)

    # 得られたすべての主成分について、第1から順に、
    # 式・寄与率・累積寄与率を出力するように作成せよ。 資料のものと同様の主成分が得られた。
    print("式")
    print("z1 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5"%(v0[0], v0[1], v0[2], v0[3], v0[4]))
    print("z2 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5"%(v1[0], v1[1], v1[2], v1[3], v1[4]))
    print("z3 = %f x1 + %f x2 + %f x3 + %f x4 + %f x5"%(v2[0], v2[1], v2[2], v2[3], v2[4]))
    # print("z4 = %f x1 + %f x2 + %f x3 + %f x4"%(v3[0], v3[1], v3[2], v3[3]))

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
    f1 = v[:,0]*(w[0]**0.5)
    f2 = v[:,1]*(w[1]**0.5)
    drawScatter(f1, f2, "f1", "f2", data_name)
    return (xy)




    
####################################
# メイン関数
####################################    
if __name__ == "__main__":

    # 変数名の読み込み（MACで文字化けしたら下のを使う）
    fn = "lec04_data_variable_name_sjis.txt"  # Windows用    
    #fn = "lec04_data_variable_name_utf8n.txt" # MAC用  
    variable_name_list = loadText(fn)

    # サンプル名の読み込み（MACで文字化けしたら下のを使う）
    fn = "lec04_data_sample_name_sjis.txt"  # Windows用
    #fn = "lec04_data_sample_name_utf8n.txt" # MAC用
    sample_name_list = loadText(fn)


    # データの読み込み
    
    data_name = "lec04_data_nagoya"
    file_name = data_name + ".csv"

    # # 確認のためデータの中身をprintしてみる
    # print("元データ")
    # print(d)
    
    # プロット点のラベルにsample_name_listを指定
    data_name = "lec04_fdata_nagoya6"
    d = mySD6(D)
    print(d[:,1])
    data_name = "lec04_data_nagoya6"
    drawScatter(d[:,0],d[:,1],"PC1","PC2",
                data_name,plot_labels=sample_name_list)

    # プロット点のラベルにsample_name_listを指定
    data_name = "lec04_fdata_nagoya5"
    d5 = mySD5(D5)
    data_name = "lec04_data_nagoya5"
    drawScatter(d5[:,0],d5[:,1],"PC1","PC2",
                data_name,plot_labels=sample_name_list)
    print(D5[:,0:2])



    
    
    


