"""

模型可视化
训练可视化
https://blog.csdn.net/m0_37477175/article/details/79131456

"""
import matplotlib.pyplot as plt
def show_loss(history):
    print(history.history.keys())
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'loss'], loc='upper left')
    fig.savefig('performance.png')