

#区间缩放，返回值为缩放到[0, 1]区间的数据
from sklearn.datasets import load_iris
"""
IRIS数据集由Fisher在1936年整理，
包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），特征值都为正浮点数，单位为厘米。
目标值为鸢尾花的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））
"""

iris = load_iris()

print(iris.data[0])

"""
https://www.zhihu.com/question/28641663
"""

#区间缩放法，返回值为缩放到[0, 1]区间的数据
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler().fit_transform(iris.data)
print(minmax[0])

# 归一化
from sklearn.preprocessing import Normalizer
print(Normalizer().fit_transform(iris.data)[0])

# 标准化
from sklearn.preprocessing import StandardScaler
print(StandardScaler().fit_transform(iris.data)[0])