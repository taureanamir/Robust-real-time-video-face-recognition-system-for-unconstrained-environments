import threading
import cv2
from datetime import datetime
import os
import logging

from ConfigManager import ConfigManager

cfg = ConfigManager.Instance()
server_cert = cfg.sServerCert
HA_POST_URL = cfg.sHAPostUrl


class Saver(threading.Thread):
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    # def __init__(self, track, identity, nface, path, camera_id, camera_name, hasLock, nodeJS_POST_URL, name="Saver"):
    def __init__(self, track, identity, nface, path, camera_id, nodeJS_POST_URL, name="Saver"):
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True
        self.track = track
        self.identity = identity
        self.nface = nface
        self.path = path
        self.camera_id = camera_id
        self.nodeJS_POST_URL = nodeJS_POST_URL
        # self.hasLock = hasLock
        # self.camera_name = camera_name
        # if self.hasLock:
        #     if self.camera_name == 'Entry':
        #         self.nodeJS_POST_URL = cfg.sNodeJSPostUrl1
        #         # self.nodeJS_POST_URL = 'http://192.168.5.198:4000'
        #     elif self.camera_name == 'Exit':
        #         self.nodeJS_POST_URL = cfg.sNodeJSPostUrl2
        #         # self.nodeJS_POST_URL = 'http://192.168.5.140:4000'

    def run(self):
        curtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S +07")

        # save detected face image to disk
        save_detected_face_img = os.path.join(self.path, str(self.track.track_id) + '-' + self.identity + '-' + self.nface.confidence +
                                              '-' + 'detect-conf-' + str(self.nface.detection_conf) + '-' + datetime.now().strftime("%Y%m%d%H%M%S") + ".png")

        cv2.imwrite(save_detected_face_img, self.nface.image)
        # logging.info(self.nface.image)

        # if self.hasLock:
        #     if cfg.bDebugMode:
        #         logging.info("Message sent to Node.js app at: " +
        #                      datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

        #     msg2NodeApp = "curl -d '{\"track\":\"" \
        #         + str(self.track.track_id) \
        #         + "\", \"class\":\"" \
        #         + self.identity \
        #         + "\", \"confidence\":\"" \
        #         + self.nface.confidence + \
        #         "\"}' -H\"Content-Type: application/json\" -X POST " + self.nodeJS_POST_URL

        #     # send message to node application
        #     if cfg.bDebugMode:
        #         logging.info(msg2NodeApp)

        #     os.system(msg2NodeApp)

        if cfg.bDebugMode:
            logging.info("Message sent to Node.js app at: " +
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

        msg2NodeApp = "curl -d '{\"track\":\"" \
            + str(self.track.track_id) \
            + "\", \"class\":\"" \
            + self.identity \
            + "\", \"confidence\":\"" \
            + self.nface.confidence + \
            "\"}' -H\"Content-Type: application/json\" -X POST " + self.nodeJS_POST_URL

        # send message to node application
        if cfg.bDebugMode:
            logging.info(msg2NodeApp)

        os.system(msg2NodeApp)

        if cfg.bDebugMode:
            logging.info("Message sent to HA web app at: " +
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

        # for debugging purpose we can add
        # " --verbose " \
        # after the first line of curl
        msg2HA = "curl -H \"Accept: application/json\" --insecure --cacert " + server_cert + \
                 " -F \"face_access[name]=" + self.identity + "\"" \
                 " -F \"face_access[time]=" + curtime + "\"" \
                 " -F \"face_access[confidence]=" + self.nface.confidence + "\"" \
                 " -F \"face_access[camera_id]=" + self.camera_id + "\"" \
                 " -F \"face_access[detected_face]=@\"" + save_detected_face_img + "\";type=image/jpeg\" " \
                 + HA_POST_URL

        if cfg.bDebugMode:
            logging.info(msg2HA)

        # send message to human analytics
        os.system(msg2HA)
        logging.info("Recognition info sent to HA Web !!!")
