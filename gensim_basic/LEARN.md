###gensim
gensim是一个python的自然语言处理库，能够将文档根据TF-IDF, LDA, LSI 等模型转化成向量模式，以便进行进一步的处理。
此外，gensim还实现了word2vec功能，能够将单词转化为词向量。

###models
潜在主题(topic)的数目，也等于转成lsi模型以后每个文档对应的向量长度。转化以后的向量在各项的值，即为该文档在该潜在主题的权重。
因此lsi和lda的结果也可以看做该文档的文档向量，用于后续的分类，聚类等算法。



[gensim使用方法以及例子](https://blog.csdn.net/u014595019/article/details/52218249)
[](https://blog.csdn.net/u014595019/article/details/52433754)