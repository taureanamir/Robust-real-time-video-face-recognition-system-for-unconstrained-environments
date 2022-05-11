from timeit import time

import numpy as np
import queue
import cv2
import tensorflow as tf
import threading
from PIL import Image
import logging
from collections import Counter

from tracking_DS_yV3.deep_sort import preprocessing
from tracking_DS_yV3.deep_sort.detection import Detection
from utils import Utils as utils
from Saver import Saver
from ConfigManager import ConfigManager


cfg = ConfigManager.Instance()

# Definition of the parameters
# Number of frames after which to run face detection
frame_interval = cfg.iFaceDetectionInterval
fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0
faceConfThreshold = cfg.fFaceConfThreshold
q_tracks = queue.Queue(20)
nms_max_overlap = 1.0

# constant for name = Unknown
UNKNOWN = 'Unknown'


class Processor(threading.Thread):
    """
    Class that continuously processes a frame using a dedicated thread.
    """

    def __init__(self, encoder, tracker, face_recognition, yolo, camera_id, camera_name, nodeJS_POST_URL, frame=None, name="Processor"):
        threading.Thread.__init__(self)
        self.frame = frame
        self.sess = tf.Session()
        self.stopped = False
        self.name = name
        self.daemon = True
        self.encoder = encoder
        self.tracker = tracker
        self.face_recognition = face_recognition
        self.yolo = yolo
        self.start_time = time.time()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.nodeJS_POST_URL = nodeJS_POST_URL
        # num of frames to consider before confirming the identity
        self.numOfFrames = cfg.iNumFramesToConfirmIdentity

    def confirm_identity(self, track_name):
        # make a list of distinct names and their counts in the descending order of count
        distinct_names_list = sorted(
            Counter(track_name).items(), reverse=True, key=lambda x: x[1])
        # select the name with the highest count
        potential_identity = distinct_names_list[0]

        if potential_identity[1] >= (self.numOfFrames * 0.6):
            confirmed_identity = potential_identity[0]
        else:
            confirmed_identity = UNKNOWN

        return confirmed_identity

    def run(self):
        # count = 0
        while not self.stopped:
            # utils.overlay_framerate(self.frame, frame_rate)
            image = Image.fromarray(self.frame[..., ::-1])  # bgr to rgb
            boxs = self.yolo.detect_image(image, thread_safe=True)
            features = self.encoder(self.frame, boxs)

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature)
                          for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            # Make directory to save the recognized face image
            directory = utils.makeDir(self.camera_name)
            path = directory

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

            # loop over detections to get the bbox

                bbox = track.to_tlbr()

                if cfg.bDispHumanBbox:
                    utils.draw_human_bbox(self.frame, int(bbox[0]), int(
                        bbox[1]), int(bbox[2]), int(bbox[3]), str(track.track_id))

                croppedTrackBbox = self.frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]

                # Make directory to save the track bbox
                # track_directory = utils.makeDir(self.camera_id, str(track.track_id))
                # track_path = track_directory
                #
                # saveCroppedTrackBbox = os.path.join(track_path, str(track.track_id) + '-' + str(count) + '-' + datetime.now().strftime("%Y%m%d%H%M%S") + "-croppedTrackBbox.png")
                #
                # cv2.imwrite(saveCroppedTrackBbox, croppedTrackBbox)

                faces = self.face_recognition.identify(
                    croppedTrackBbox, thread_safe=True)

                if faces is None:
                    break
                else:
                    if cfg.bDispFaceBbox:
                        utils.add_overlays(croppedTrackBbox, faces)

                    if cfg.bDispFaceLandmark:
                        utils.draw_facial_landmarks(croppedTrackBbox, faces)

                    # push name to the track object when meets the following criteria
                    # faces generally consists of only one face, but to handle the case where there might be multiple faces
                    for nface in faces:
                        if float(nface.confidence) >= faceConfThreshold and nface.is_frontal and nface.name != UNKNOWN:
                            # set_prev_track = True
                            track.set_name(nface.name)
                        elif nface.name == UNKNOWN:
                            track.set_name(UNKNOWN)

                    # after the recognition result of the first numOfFrames frames, calculate to confirm the identity
                    # CONFIRM identity
                    if len(track.name) == self.numOfFrames:
                        identity = self.confirm_identity(track.name)

                        if cfg.bDebugMode:
                            logging.info(track.name)
                            logging.info(identity)

                        if identity != UNKNOWN:
                            # send command to open the door once the identity is confirmed.
                            if track.track_id not in q_tracks.queue:
                                logging.info("Saver thread called . . .")
                                saver = Saver(
                                    track, identity, nface, path, self.camera_id, self.nodeJS_POST_URL)
                                saver.start()
                                logging.info("Saving info complete . . .")

                                if q_tracks.full():
                                    q_tracks.get()
                                    q_tracks.put(track.track_id)
                                else:
                                    q_tracks.put(track.track_id)

            # count += 1
            # print("Video processed in: ", str(t4 - t3))

            if cfg.bDispVideoOutput:
                cv2.imshow("Video", self.frame)
            # cv2.waitKey(0)

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

        cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
        # self.join()
