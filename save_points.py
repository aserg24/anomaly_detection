from preprocessing import *
import time
from multiprocessing import Pool


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

    with open('points.pickle', 'wb') as f:
        pickle.dump(points, f)
