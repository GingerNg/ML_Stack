"""
哑编码
example:
现在假设给你的一个病人的病情描述，一般病情的描述包含以下几个方面，将病情严重程度划分：
非常严重，严重，一般严重，轻微
现在有个病人过来了，要为他构造一个病情的特征，假设他的病情是严重情况，我们可以给他的哑编码是
0 1 0 0
"""
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[1, 1, 4], [2, 2, 1], [1, 3, 2], [2, 1, 3]])
print(enc.n_values_)
print(enc.feature_indices_)
print(enc.transform([[1, 1, 4], [2, 2, 1], [1, 3, 2], [2, 1, 3]]))
print(enc.transform([[1, 2, 2]]).toarray())