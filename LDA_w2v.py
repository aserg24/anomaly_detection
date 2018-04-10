from preprocessing import *
from gensim import corpora, models
from multiprocessing import Pool
import time


def getPoints(pnts_a_b):
    print(pnts_a_b[1], '-', pnts_a_b[2])
    points = []
    for pnt in pnts_a_b[0]:
        text = tokenize(normalize(remove_stop_words(' '.join(regexp_tokenize(pnt.text)))))
        points.append(text)
    return [points, pnts_a_b[1], pnts_a_b[2]]



if __name__ == '__main__':
    pnts = db_session.query(Point).order_by(Point.text_id, Point.order_id)
    #points = getPoints(pnts, 45000, 50000)

    args = [[pnts[x:x+100], x, x+100] for x in range(0, 540900, 100)] #540900

    t1 = time.time()
    p = Pool(10)
    points_a_b = p.map(getPoints, args)
    t2 = time.time()
    print(t2-t1)

    points_a_b.sort(key=lambda i: i[1])
    points = []
    for point in points_a_b:
        points.extend(point[0])


    print('lda begins')

    t1 = time.time()

    dictionary = corpora.Dictionary(points)
    corpus = [dictionary.doc2bow(text) for text in points]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    ldamodel.update(corpus)
    ldamodel.save('lda.model')

    t2 = time.time()
    print(t2 - t1)

    print('lda is ready, w2v begins')

    t1 = time.time()

    w2vmodel = models.Word2Vec(size=15, min_count=1)
    w2vmodel.build_vocab(points)
    w2vmodel.train(points, total_examples=w2vmodel.corpus_count, epochs=w2vmodel.iter)
    w2vmodel.save('w2v.model')

    t2 = time.time()
    print(t2 - t1)


