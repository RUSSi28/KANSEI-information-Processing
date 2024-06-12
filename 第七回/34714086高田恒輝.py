import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
plt.rcParams['font.family'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic']

X = [
    [0.11515152, 0.61818182],
    [0.1025641,  0.6025641 ],
    [0.1097561,  0.60365854],
    [0.12080537, 0.63087248],
    [0.09774436, 0.62406015],
    [0.07228916, 0.68674699],
    [0.12380952, 0.61904762],
    [0.10606061, 0.60606061],
    [0.12727273, 0.62272727],
    [0.12244898, 0.6122449 ],
    [0.10948905, 0.59854015],
    [0.15028902, 0.60693642],
    [0.15702479, 0.62396694],
    [0.14634146, 0.63414634],
    [0.12883436, 0.63803681]
]


Z_w = linkage(X, method="ward")
print(Z_w)
label_list = [
    "千種区",
    "東区",
    "北区",
    "西区",
    "中村区",
    "中区",
    "瑞穂区",
    "熱田区",
    "中川区",
    "港区",
    "南区",
    "守山区",
    "緑区",
    "名東区",
    "天白区"
]
dendrogram(Z_w, labels=label_list, orientation='right')
plt.savefig("dendrogram.png")
