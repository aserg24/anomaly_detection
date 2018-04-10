from preprocessing import *
from gensim import models
from sklearn import neighbors
import time
import numpy as np



if __name__ == '__main__':
    ldamodel = models.LdaModel.load('lda.model')
    w2vmodel = models.Word2Vec.load('w2v.model')

    '''
    print('start')
    t1 = time.time()
    docs_id = []
    for instance in db_session.query(Point).order_by(Point.text_id, Point.order_id):
        if instance.text_id not in docs_id:
            docs_id.append(instance.text_id)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    with open('points.pickle', 'rb') as f:
        points = pickle.load(f)
    print('points is open')
    t2 = time.time()
    print(t2 - t1)

    print('creation of vector_lda_w2v')
    t1 = time.time()
    y = []
    n = 0
    len = 0
    vector_lda_w2v = []
    for doc_id in tqdm(docs_id):  # по id документа
        doc_len = db_session.query(Point).filter(Point.text_id == doc_id).count()
        l = db_session.query(Point).filter(Point.text_id == doc_id).count()
        for instance in db_session.query(Point).filter(Point.text_id == doc_id)[1:l-1]:
            y.append(int(instance.is_correct))



        # lda
        dictionary = corpora.Dictionary([points[n]])   # для нулевого параграфа в документе
        corpus = dictionary.doc2bow(points[n])
        vector_lda_w2v.append([instance[1] for instance in ldamodel[corpus]])
        len += 1

        dictionary = corpora.Dictionary([points[n+1]])    # для первого параграфа в документе
        corpus = dictionary.doc2bow(points[n+1])
        vector_lda_w2v.append([instance[1] for instance in ldamodel[corpus]])
        len += 1

        for i in range(2, doc_len - 2):  # по документу с id = doc_id (по его параграфам со 2 по 3 с конца)
            dictionary = corpora.Dictionary([points[n+i]])
            corpus = dictionary.doc2bow(points[n+i])
            vec = [instance[1] for instance in ldamodel[corpus]]
            vector_lda_w2v[len-2].extend(vec)
            vector_lda_w2v.append(vec)
            len += 1

        dictionary = corpora.Dictionary([points[n + i]])  # для 2 с конца параграфа в документе
        corpus = dictionary.doc2bow(points[n + i])
        vector_lda_w2v[len-2].extend([instance[1] for instance in ldamodel[corpus]])

        dictionary = corpora.Dictionary([points[n + i + 1]])  # для 1 с конца параграфа в документе
        corpus = dictionary.doc2bow(points[n + i + 1])
        vector_lda_w2v[len-1].extend([instance[1] for instance in ldamodel[corpus]])

        #print('after lda', vector_lda_w2v)

        # word2vec
        cur = 0

        #print('doc_id', doc_id)
        #print('points[n]', points[n])
        vec = sum([w2vmodel[j] for j in points[n]]) # для нулевого параграфа в документе
        #print('vec', vec)
        #print('len - (doc_len - 2)', len - (doc_len - 2))
        #print('vector_lda_w2v[len - (doc_len - 2)]', vector_lda_w2v[len - (doc_len - 2)])
        vector_lda_w2v[len - (doc_len - 2)].extend(vec)
        cur += 1

        vec = sum([w2vmodel[j] for j in points[n+1]])  # для первого параграфа в документе
        vector_lda_w2v[len - (doc_len - 2) + 1].extend(vec)
        cur += 1

        for i in range(2, doc_len - 2):  # по документу с id = doc_id (по его параграфам со 2 по 3 с конца)
            vec = sum([w2vmodel[j] for j in points[n + i]])
            vector_lda_w2v[len - (doc_len - 2) + cur - 2].extend(vec)
            vector_lda_w2v[len - (doc_len - 2) + cur].extend(vec)
            cur += 1

        vec = sum([w2vmodel[j] for j in points[n]])  # для 2 с конца параграфа в документе
        vector_lda_w2v[len - (doc_len - 2) + cur - 2].extend(vec)

        vec = sum([w2vmodel[j] for j in points[n]])  # для 1 с конца параграфа в документе
        vector_lda_w2v[len - (doc_len - 2) + cur - 1].extend(vec)

        n += doc_len

    print('vector_lda_w2v created')
    t2 = time.time()
    print(t2 - t1)

    print('save vector_lda_w2v...')
    t1 = time.time()
    with open('vector_lda_w2v.pickle', 'wb') as f:
        pickle.dump(vector_lda_w2v, f)
    t2 = time.time()
    print(t2 - t1)
    print('save y...')
    t1 = time.time()
    with open('is_correct.pickle', 'wb') as f:
        pickle.dump(y, f)
    t2 = time.time()
    print(t2 - t1)
    
'''
    with open('vector_lda_w2v.pickle', 'rb') as f:
        vector_lda_w2v = pickle.load(f)

    with open('is_correct.pickle', 'rb') as f:
        y = pickle.load(f)


    print('classification begins')

    num = 0
    j = 0
    while j < len(vector_lda_w2v):
        if len(vector_lda_w2v[j]) < 50:
            del vector_lda_w2v[j]
            del y[j]
        else:
            j += 1

    x_train = np.array(vector_lda_w2v[len(vector_lda_w2v) * 3 // 10:])
    x_test = np.array(vector_lda_w2v[:len(vector_lda_w2v) * 3 // 10])
    y_train = np.array(y[len(y) * 3 // 10:])
    y_test = np.array(y[:len(y) * 3 // 10])
    #print(type(vector_lda_w2v), type(vector_lda_w2v[0]))
    #x = np.array(vector_lda_w2v)
    #print(type(x[0]))
    #y = np.array(y)
    t1 = time.time()
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    print('classification fitted')
    t2 = time.time()
    print(t2 - t1)

    n = 0
    nums = [0, 0]
    y = clf.predict(x_test)
    for i, j in enumerate(y):
        nums[j] += 1
        if j == y_test[i]:
            n += 1
    print('results', n/len(y_test), nums)

    print('save clf...')
    t1 = time.time()
    with open('clf_knn.pickle', 'wb') as f:
        pickle.dump(clf, f)
    t2 = time.time()
    print(t2 - t1)
