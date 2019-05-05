



tensorflow
Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。



tf.flags
用于帮助我们添加命令行的可选参数
FLAGS = tf.flags.FLAGS #FLAGS保存命令行参数的数据
FLAGS._parse_flags() #将其解析成字典存储到FLAGS.__flags中

###tf.Graph()
tf.Graph() 默认执行
graph定义了计算方式，是一些加减乘除等运算的组合，类似于一个函数。它本身不会进行任何计算，也不保存任何中间计算结果。
session用来运行一个graph，或者运行graph的一部分。它类似于一个执行者，给graph灌入输入数据，得到输出，并保存中间的计算结果。
同时它也给graph分配计算资源（如内存、显卡等）。
op -- 操作
graph就是由一系列op构成的




###tf.train
 tf.train.global_step()用来获取当前sess的global_step值


###tf.summary
在训练过程中，主要用到了tf.summary()的各类方法，能够保存训练过程以及参数分布图并在tensorboard显示。
histograms:变量显示
tf.summary.histogram

scalars:标量显示
tf.summary.scalar


text:展示文本输入类型为Tensor
tf.summary.text

tf.summary.merge(inputs, collections=None, name=None)
合并summaries

该op创建了一个summary协议缓冲区，它包含了输入的summaries的所有value的union


tf.argmax(input, axis=None, name=None, dimension=None)
此函数是对矩阵按行或列计算最大值


###tf.nn
tf.nn.embedding_lookup（tensor, id）函数的用法主要是选取一个张量里面索引(id)对应的元素

tf.nn.conv2d
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

tf.placeholder(dtype, shape=None, name=None)
此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

Tensorflow API解读
https://www.cnblogs.com/lainey/p/7927973.html


tf.session 配置
log_device_placement=True : 是否打印设备分配日志
allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片

控制使用哪块GPU
/ CUDA_VISIBLE_DEVICES=0  python your.py#使用GPU0
~/ CUDA_VISIBLE_DEVICES=0,1 python your.py#使用GPU0,1
#注意单词不要打错
#或者在 程序开头
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1


tf.get_variable
tf.Variable

tf.name_scope
tf.variable_scope
scope 作用域

当tf.get_variable用于变量创建时，和tf.Variable的功能基本等价。
tf.get_varialbe和tf.Variable最大的区别在于：
tf.Variable的变量名是一个可选项，通过name=’v’的形式给出
tf.get_variable必须指定变量名

tf.get_variable与tf.variable_scope：
当reuse为False或者None时（这也是默认值），同一个tf.variable_scope下面的变量名不能相同；
当reuse为True时，tf.variable_scope只能获取已经创建过的变量

TensorFlow中通过变量名获取变量的机制主要是通过tf.get_variable和tf.variable_scope实现的

tf.variable_scope，tf.name_scope函数也提供了命名空间管理的功能。这两个函数在大部分情况下是等价的
唯一的区别是在使用tf.get_variable函数时：tf.get_variable函数不受tf.name_scope的影响。
http://www.cnblogs.com/MY0213/p/9270864.html



tf.random_normal    从正态分布中输出随机值。
tf.truncated_normal  从截断的正态分布中输出随机值;生成的值服从具有指定平均值和标准偏差的正态分布
tf.constant
tf.random_uniform

tf.global_variables_initializer()  初始化模型的参数
global_variables_initializer 返回一个用来初始化 计算图中 所有global variable的 assign op


tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
strides 卷积时在图像每一维的步长，这是一个一维的向量，长度4


tf.sequence_mask
tf.boolean_mask


tf.nn.sparse_softmax_cross_entropy_with_logits

[TensorFlow高效读取数据的方法](https://blog.csdn.net/u012759136/article/details/52232266)