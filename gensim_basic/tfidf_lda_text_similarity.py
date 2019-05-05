# ­*­ coding: utf­8 ­*­
"""
@author: ginger
@file: tfidf_lda_text_similarity.py
@time: 2018/8/23 22:11
https://blog.csdn.net/qq_34941023/article/details/51786765

文本相似性计算

TFIDF模型
LDA模型
"""

from gensim import corpora,models,similarities,utils
import jieba
import jieba.posseg as pseg
# jieba.load_userdict( "user_dic.txt" ) #载入自定义的词典，主要有一些计算机词汇和汽车型号的词汇
#定义原始语料集合

output = "../data/tfidf/"

def etl(x):
    return x

def get_train_set():
    train_set=[]
    f=open("../data/xinsanban.txt")
    lines=f.readlines()
    for line in lines:
        content = (line.lower()).split("\t")[2] + (line.lower()).split("\t")[1]
        #切词，etl函数用于去掉无用的符号，cut_all表示非全切分
        # word_list = filter(lambda x: len(x)>0,map(etl,jieba.cut(content,cut_all=False)))
        # word_list = filter(lambda x: len(x) > 0, jieba.cut(content, cut_all=False))
        # word_list = jieba.cut(content, cut_all=False)

        texts = [word for word in jieba.cut(content, cut_all=True)]

        train_set.append(texts)
    f.close()
    return train_set

def create_dictionary(train_set):
    #生成字典
    dictionary = corpora.Dictionary(train_set)
    """
    Removing words creates gaps in the dictionary, but calling dictionary.compactify() re-assigns ids to fill in the gaps.
    But that means our vectorized_corpus from above doesn't use the same id's as the dictionary any more, and if we pass them into a model,
    we'll get an IndexError.
    """
    # #去除极低频的杂质词
    # dictionary.filter_extremes(no_below=2,no_above=0.5,keep_n=None)
    #
    # dictionary.compactify()
    #将词典保存下来，方便后续使用
    dictionary.save(output + "all.dic")
    return dictionary

def create_corpus(train_set):
    dictionary = corpora.Dictionary(train_set)
    # 将语料导入词典后，每个词实际上就已经被编号成1,2,3....这种编号了，这是向量化的第一步，然后把词典保存下来。
    # 然后生成数字语料
    corpus = [dictionary.doc2bow(text) for text in train_set]  # 实现词袋模型
    return corpus


def create_tfidf_model(corpus):
    #使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)
    #存储tfidfModel
    tfidfModel.save(output + "allTFIDF.mdl")
    return tfidfModel

def create_tfidf_vector(tfidfModel,corpus):
    #把全部语料向量化成TFIDF模式，这个tfidfModel可以传入二维数组
    tfidfVectors = tfidfModel[corpus]
    #建立索引并保存
    indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
    indexTfidf.save(output + "allTFIDF.idx")
    return tfidfVectors,indexTfidf


def lda(tfidfVectors,dictionary):
    #通过TFIDF向量生成LDA模型，id2word表示编号的对应词典，num_topics表示主题数，我们这里设定的50，主题太多时间受不了。
    lda = models.LdaModel(corpus=tfidfVectors,
                          id2word=dictionary,
                          num_topics=10,
                          alpha=0.01,
                          eta=0.01,
                          minimum_probability=0.001,
                          update_every=1,
                          chunksize=10,
                          passes=1
                          )
    #把模型保存下来
    lda.save(output + "allLDA50Topic.mdl")
    #把所有TFIDF向量变成LDA的向量
    corpus_lda = lda[tfidfVectors]
    #建立索引，把LDA数据保存下来
    indexLDA = similarities.MatrixSimilarity(corpus_lda)
    indexLDA.save(output + "allLDA50Topic.idx")

def predict(query):
    # 载入字典
    dictionary = corpora.Dictionary.load(output + "all.dic")
    # 载入TFIDF模型和索引
    tfidfModel = models.TfidfModel.load(output + "allTFIDF.mdl")
    indexTfidf = similarities.MatrixSimilarity.load(output + "allTFIDF.idx")
    # 载入LDA模型和索引
    ldaModel = models.LdaModel.load(output + "allLDA50Topic.mdl")
    indexLDA = similarities.MatrixSimilarity.load(output + "allLDA50Topic.idx")

    texts = [word for word in jieba.cut(query, cut_all=True)]
    query_bow = dictionary.doc2bow(texts)
    # query就是测试数据，先切词
    # query_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(etl, jieba.cut(query, cut_all=False))))
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]
    # 然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]
    # TFIDF相似性
    simstfidf = indexTfidf[tfidfvect]
    # LDA相似性
    simlda = indexLDA[ldavec]   # 训练集中与该文本的相似度
    return simlda,simstfidf



if __name__ == "__main__":
    # train_set = get_train_set()
    # dictionary = create_dictionary(train_set)
    # corpus = create_corpus(train_set)
    # tfidfModel = create_tfidf_model(corpus)
    # tfidfVectors, indexTfidf = create_tfidf_vector(tfidfModel, corpus)
    #
    # lda(tfidfVectors, dictionary)
    query = """
    据新三板+AppAiLab统计，前一交易日共有7家公司发布公告，公司增发方案已经获得董事会及股东大会批准。数据显示，本次7家公司共计募集资金1.46亿元。“新三板+”AppAiLab显示以下为各家公司增发方案详情：数据来源：“新三板+”AppAiLab坦程物联、力石科技等7家公司公告增发1.46亿元
    """
    result, simstfidf= predict(query)

    sort_sims = sorted(enumerate(result), key=lambda item: -item[1])
    print (sort_sims)
    # for sim in sort_sims[:10]:
    #     print("ID : " + docinfos[sim[0]]["id"] + "\t" + docinfos[sim[0]]["title"] + "\tsimilary:::" + str(sim[1]))


    # print(len(simstfidf))
    # result = result.tolist()
    # max_index = result.index(max(result))
    # print(max_index)
    # print(result[max_index])
