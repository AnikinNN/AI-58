import time
from queue import Queue, Empty
from threading import Thread


def threaded_feeder(q: Queue, _id):
    maximum = 10
    count = 0
    while True:
        if count < maximum:
            q.put(count + _id * 100)
            count += 1
        print(f'threaded_feeder_{_id}')
        time.sleep(1)


qu = Queue(maxsize=4)

for i in range(1):
    thr = Thread(target=threaded_feeder, args=(qu, i))
    thr.start()

for i in range(100):
    print(qu.get(block=True))
