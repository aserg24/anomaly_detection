from structure import *
from gensim import corpora, models

if __name__ == '__main__':
    doc = Document()
    doc.load('txts_pickled/contract1')
    points = []

    # на примере одного из документов представляем весь договор
    # в виде списка из последовательных пунктов этого договора:

    #  title
    for sentence in doc.title.sentences:
        snt = sentence.regexp_tokenize()
        snt = Sentence(' '.join(snt)).remove_stop_words().normalize().tokenize()
        points.append(snt)

    #  preamble
    for sentence in doc.preamble.sentences:
        snt = sentence.regexp_tokenize()
        snt = Sentence(' '.join(snt)).remove_stop_words().normalize().tokenize()
        points.append(snt)

    #  numered text
    chapters = doc.get_titles_of_chapters()
    for chapter in chapters:
        doc[chapter].get_sentences()
        for paragraph in doc[chapter].queue:
            point = paragraph.sentences
            point = [pnt.sentence for pnt in point]
            point = Sentence(' '.join(point)).regexp_tokenize()
            point = Sentence(' '.join(point)).remove_stop_words().normalize().tokenize()
            points.append(point)

    #  attachment
    for sentence in doc.attachment.sentences:
        snt = sentence.regexp_tokenize()
        snt = Sentence(' '.join(snt)).remove_stop_words().normalize().tokenize()
        if snt:
            points.append(snt)

    # построение LDA модели с 4 темами
    dictionary = corpora.Dictionary(points)
    print(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in points] # list(lists(tuples(doc_id, count)))

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=4, num_words=3))

