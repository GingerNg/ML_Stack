import math


def lse():
    """
    https://www.cnblogs.com/BlogOfMr-Leo/p/8627311.html
    :return:
    """
    import numpy as np
    from scipy import optimize

    x = np.array([0.9, 2.5, 3.3, 4.5, 5.7, 6.9])
    y = np.array([1.1, 1.6, 2.6, 3.2, 4.0, 6.0])

    def reds(p):
        # 计算以p为参数的直线和数据之间的误差
        k, b = p
        return y - (k * x + b)
        # return math.pow((y - (k * x + b)),2)

    # leastsq 使得reds()输出最小，参数的初始值是【1,0】
    r = optimize.leastsq(reds, [1, 0])
    k, b = r[0]
    print("k=", k, "\n b=", b)
    y1 = x * k + b
    a = np.array([y1[0] - y[0], y1[1] - y[1], y1[2] - y[2], y1[3] - y[3], y1[4] - y[4], y1[5] - y[5]])
    print("\n", y, "\n", y1, a)
    print("灵敏度计算", k)

if __name__ == '__main__':
    print(2^2)
    print(math.pow(2,3))
    lse()