import threading
import cv2
from datetime import datetime
from queue import Queue
import time

class VideoGet(threading.Thread):
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, name="VideoReader", queueSize=128):
        threading.Thread.__init__(self)
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.name = name
        self.daemon = True
        self.Q = Queue(maxsize=queueSize)

    def run(self):
        while not self.stopped:
            t1 = datetime.now()

            # if the queue is not full
            if not self.Q.full():
                # if self.grabbed = False, we have reached the end of the stream / video file.
                # No more frames read.
                if not self.grabbed:
                    self.stop()
                else:
                    (self.grabbed, self.frame) = self.stream.read()

                # add the frame to the queue
                self.Q.put(self.frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

            t2 = datetime.now()
            # print("Video get in: ", str(t2-t1))

        self.stream.release()

    def read(self):
        # return next frame in the queue
        # print("Queue size: ", self.Q.qsize())
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # tell that thread should be stopped
        self.stopped = True
        # self.join()