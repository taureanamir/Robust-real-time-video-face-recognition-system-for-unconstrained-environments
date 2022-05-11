import os
import cv2
# import logging
import logging.config
from datetime import datetime

from ConfigManager import ConfigManager

cfg = ConfigManager.Instance()
_dir = os.path.normpath(cfg.APPDIR + cfg.slogDir)

class Utils:

    @staticmethod
    def add_overlays(frame, faces):
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(frame,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              (0, 255, 0), 2)
                if face.name is not None:
                    cv2.putText(frame, face.name + " - " + face.confidence, (face_bb[0], face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                thickness=2, lineType=2)

    @staticmethod
    def overlay_framerate(frame, frame_rate):
        cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)

    @staticmethod
    def logger(cam):
        logFilePath = os.path.join(os.path.normpath(_dir), cam)

        if not os.path.exists(logFilePath):
            os.makedirs(logFilePath)
        logFile = os.path.join(logFilePath, cfg.sLogFile)
        logFileMode = 'a'
        logLevel = logging.INFO
        logFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logDateFmt = '%d-%b-%y %H:%M:%S'
        logging.basicConfig(filename=logFile, filemode=logFileMode, format=logFormat, datefmt=logDateFmt,
                            level=logLevel)

    @staticmethod
    def makeDir(cam):
        
        directory = os.path.join(_dir, cam, cfg.sCapturedFaceDir, datetime.today().strftime('%Y-%m-%d'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    @staticmethod
    def draw_facial_landmarks(frame, faces):
        """
        facial_landmarks contains a numpy n-D array of 10 elements.
           1. Mouth Left Corner -> [0],[9]
           2. Mouth right corner -> [1],[8]
           3. Nose -> [2],[7]
           4. Left eye -> [3],[6]
           5. Right eye -> [4],[5]
        """

        for face in faces:
            cv2.circle(frame, (face.face_landmarks[0], face.face_landmarks[9]), 5, (0, 0, 255), -2)
            cv2.circle(frame, (face.face_landmarks[1], face.face_landmarks[8]), 5, (0, 0, 255), -2)
            cv2.circle(frame, (face.face_landmarks[2], face.face_landmarks[7]), 5, (0, 0, 255), -2)
            cv2.circle(frame, (face.face_landmarks[3], face.face_landmarks[6]), 5, (0, 0, 255), -2)
            cv2.circle(frame, (face.face_landmarks[4], face.face_landmarks[5]), 5, (0, 0, 255), -2)
            cv2.putText(frame, str(face.is_frontal), (face.face_landmarks[3] + 50, face.face_landmarks[6] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2, lineType=2)

    @staticmethod
    def draw_human_bbox(frame, topLeftX, topLeftY, bottomRightX, bottomRightY, trackID):
        cv2.rectangle(frame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (255, 255, 255), 2)
        cv2.putText(frame, trackID, (topLeftX, topLeftY), 0, 5e-3 * 200, (0, 255, 0), 2)