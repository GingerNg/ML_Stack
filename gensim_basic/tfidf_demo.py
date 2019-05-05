from gensim import corpora, models, similarities
import gensim
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    texts = [
        ['无偿', '居间', '介绍', '买卖', '毒品', '行为', '定性'],
        ['吸毒', '男', '动态', '持有', '毒品', '行为', '认定'],
    ]

    new_text = ['毒贩', '出狱', '再次', '毒品', '途中', '被抓', '行为', '认定']
    id2word = corpora.Dictionary(texts)  # 生成词典
    corpus = [id2word.doc2bow(text) for text in texts]

    new_doc_bow = id2word.doc2bow(new_text)  # one-hot表示
    print(new_doc_bow)

    tfidf_model = models.TfidfModel(corpus=corpus,
                                    dictionary=id2word)
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    lsi_model = models.LsiModel(corpus = corpus_tfidf,
                                id2word = id2word,
                                num_topics=3)

    corpus_lsi = [lsi_model[doc] for doc in corpus]

    new_text_tfidf = tfidf_model[new_doc_bow]
    new_text_lsi = lsi_model[new_text_tfidf]
    print(new_text_tfidf)
    print(new_text_lsi)

    # gensim 2 crs_matrix
    rows = []
    cols = []
    data = []
    line_count = 0
    for line in corpus_lsi:  # lsi_corpus_total 是之前由gensim生成的lsi向量
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_sparse_matrix = csr_matrix((data, (rows, cols)))  # 稀疏向量
    lsi_matrix = lsi_sparse_matrix.toarray()  # 密集向量
    print(lsi_matrix)
