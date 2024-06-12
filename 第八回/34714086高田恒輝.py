import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


def simple_regression(x, real_x):
    m = np.size(x[:,0]) #column size
    n = np.size(x[0,:]) #row size
    _x = x[:,0]
    _y = x[:,1]

    # 回帰式を求める
    x_mean = np.mean(x[:,0])
    y_mean = np.mean(x[:,1])

    S_x = _x - x_mean
    S_y = _y - y_mean
    S_xx = np.sum(S_x**2)
    S_yy = np.sum(S_x*S_y)
    print("y = %f + %f(x - %f)", y_mean, S_yy/S_xx, x_mean)

    return y_mean + S_yy*(real_x - x_mean)/S_xx


x = np.array([
    [165., 55.],
    [170., 60.],
    [152., 49.],
    [175., 65.],
    [183., 90.]
])
real_x = np.arange(0,200,1)
real_y = simple_regression(x, real_x)

plt.xlabel("height")
plt.ylabel("weight")
plt.plot(real_x, real_y, color="red")
plt.savefig("regression.png")