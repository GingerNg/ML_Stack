

"""
Bunch和字典结构类似，也是由键值对组成，和字典区别：其键值可以被实例对象当作属性使用。
"""
from sklearn.datasets import base

buch = base.Bunch(A=1,B=2,c=3)

print(buch.A)