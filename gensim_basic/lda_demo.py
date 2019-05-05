"""
主题模型是对文字隐含主题进行建模的方法。
它克服了传统信息检索中文档相似度计算方法的缺点，并且能够在海量互联网数据中自动寻找出文字间的语义主题。

主题就是词汇表上词语的条件概率分布

语义挖掘


主题模型的作用：
1.文档相似度-- 语义相似度
2.解决多义词的问题
3.排除噪声
4.无监督
5.语言无关

pLSA主要使用的是EM（期望最大化）算法；
LDA采用的是Gibbs sampling方法。


VSM模型
前提条件：文档之间重复的词语越多越可能相似
"""
from gensim import corpora, models, similarities
import gensim

if __name__ == '__main__':
    texts = [
        ['无偿','居间','介绍','买卖','毒品','行为','定性'],
        ['吸毒','男','动态','持有','毒品','行为','认定'],
    ]

    new_text = ['毒贩','出狱','再次','毒品','途中','被抓','行为','认定']
    id2word = corpora.Dictionary(texts)  # 生成词典
    print(id2word)
    print(sorted(list(id2word.items()), key=lambda x: x[0]))
    corpus = [id2word.doc2bow(text) for text in texts]         # 文档转化为向量形式

    # tfidfModel = models.TfidfModel(corpus)

    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=3)
    lda.print_topic(2, topn=3)

    new_doc_bow = id2word.doc2bow(new_text)   # one-hot表示  

    print(new_doc_bow)       # one-hot表示

    new_doc_lda = lda[new_doc_bow]

    print(new_doc_lda)    #  lda表示
