# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 日本語のフォントを設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']



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
    d = np.loadtxt(file_name,delimiter=',')

    # 確認のためデータの中身をprintしてみる
    print("元データ")
    print(d)
    

    # データが読み込めたかを確認するため，変数1と変数2で２次元プロットしてみる.
    # プロット点のラベルにsample_name_listを指定
    drawScatter(d[:,0],d[:,1],variable_name_list[0],variable_name_list[1],
                data_name,plot_labels=sample_name_list)
    



    
    
    

