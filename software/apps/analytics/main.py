import logging
import argparse
import sys
import os

from face_recognition import face
from tracking_DS_yV3.yolo import YOLO
from tracking_DS_yV3.deep_sort import nn_matching
from tracking_DS_yV3.deep_sort.tracker import Tracker
from tracking_DS_yV3.tools import generate_detections as gdet
from utils import Utils as utils
from VideoGet import VideoGet
from Processor import Processor
from imutils.video import FPS
from ConfigManager import ConfigManager


def main(args):

    # Dictionary to store camera config variables

    cameraConfigDict = {
        '1': [cfg.cam1_codec, cfg.cam1_sCamNum, cfg.cam1_iImgHeight, cfg.cam1_iImgWidth, cfg.cam1_sName, cfg.cam1_sVideoPath, cfg.cam1_sNodeJSPostUrl],
        '2': [cfg.cam2_codec, cfg.cam2_sCamNum, cfg.cam2_iImgHeight, cfg.cam2_iImgWidth, cfg.cam2_sName, cfg.cam2_sVideoPath, cfg.cam2_sNodeJSPostUrl],
        '3': [cfg.cam3_codec, cfg.cam3_sCamNum, cfg.cam3_iImgHeight, cfg.cam3_iImgWidth, cfg.cam3_sName, cfg.cam3_sVideoPath, cfg.cam3_sNodeJSPostUrl],
        '4': [cfg.cam4_codec, cfg.cam4_sCamNum, cfg.cam4_iImgHeight, cfg.cam4_iImgWidth, cfg.cam4_sName, cfg.cam4_sVideoPath, cfg.cam4_sNodeJSPostUrl]
    }

    print("---------------- -------------- cameraConfigDict: ", cameraConfigDict)

    cameraConfig = cameraConfigDict[args.camera_number]

    camera_id = cameraConfig[1]
    camera_name = cameraConfig[4]
    camera_stream = cameraConfig[5]
    nodeJS_POST_URL = cameraConfig[6]

    utils.logger(camera_name)

    logging.info(
        "----------------------------------------------------------------------------------------------------")
    logging.info("Program started !!!")
    logging.info("Using config from file: " + cfg.config_file_path)
    logging.info("Configuration output saved to: " + cfg.saveConfig())
    logging.info("Running analytics for camera id: {}, name: {} ".format(
        camera_id, camera_name))

    # Params for deep SORT
    max_cosine_distance = 0.3
    nn_budget = None

    # Load deep learning models (face recognition and deep SORT tracking)
    model_filename = os.path.join(
        os.path.normpath(cfg.APPDIR + cfg.sDeepSortModel))
    face_recognition = face.Recognition()

    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Video stream to use
    # source = camera_stream
    # USB or builtin cam
    source = 0

    # start a thread to read the video from source specified above.
    video_getter = VideoGet(source)
    video_getter.start()

    # start a thread to process the acquired video frame.
    # processor = Processor(encoder, tracker, face_recognition, yolo, camera_id, video_getter.frame)
    processor = Processor(encoder, tracker, face_recognition,
                          yolo, camera_id, camera_name, nodeJS_POST_URL, video_getter.frame)
    processor.start()
    fps = FPS().start()

    while True:
        # stop the app if no more frames read from the video source
        if video_getter.stopped or processor.stopped:
            processor.stop()
            video_getter.stop()
            break

        # get the video frame
        frame = video_getter.read()
        processor.frame = frame
        fps.update()

    # wait until stream resources are released
    video_getter.join()
    processor.join()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Run face recognition')
    parser.add_argument('--camera_number', required=False, default="1",
                        help="Enter the camera id that you want to run the analytics for.")

    return parser.parse_args(argv)


if __name__ == '__main__':
    cfg = ConfigManager.Instance()
    yolo = YOLO()  # creating object of class YOLO
    main(parse_arguments(sys.argv[1:]))
