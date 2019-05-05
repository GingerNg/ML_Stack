
"""
sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内
"""
# coding:utf-8
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(["Japan", "china", "Japan", "Korea", "china"])
print('标签个数:%s' % le.classes_)
print('标签值标准化:%s' % le.transform(["Japan", "china", "Japan", "Korea", "china"]))
print('标准化标签值反转:%s' % le.inverse_transform([0, 2, 0, 1, 2]))

# 标签个数:['Japan' 'Korea' 'china']
# 标签值标准化:[0 2 0 1 2]
# 标准化标签值反转:['Japan' 'china' 'Japan' 'Korea' 'china']