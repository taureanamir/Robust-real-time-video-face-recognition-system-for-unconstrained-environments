# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import timeit
import logging

import face_recognition.align.detect_face
from face_recognition import facenet
from ConfigManager import ConfigManager


cfg = ConfigManager.Instance()
gpu_memory_fraction = cfg.fGpuMemoryFraction
facenet_model_checkpoint = os.path.normpath(cfg.APPDIR + cfg.sFacenetModelCheckpoint)
classifier_model = os.path.normpath(cfg.APPDIR + cfg.sFaceClassifierModel)
faceConfThreshold = cfg.fFaceConfThreshold
frontal_face_threshold = cfg.fFrontalFaceThreshold


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.confidence = None
        self.face_landmarks = None
        self.is_frontal = None
        self.detection_conf = None

class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    # this method is not called anywhere
    def add_identity(self, image, person_name):
        # faces, face_landmarks = self.detect.find_faces(image)
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image, thread_safe = False):
        # faces, face_landmarks = self.detect.find_faces(image)
        faces = self.detect.find_faces(image)


        for i, face in enumerate(faces):
            if cfg.bDebugMode:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face, thread_safe)
            face.name = self.identifier.identify(face)
            face.confidence = self.identifier.get_confidence(face)
            face.face_landmarks = self.identifier.get_face_landmarks(face)
            face.is_frontal = self.identifier.check_frontal(face)

        return faces


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            if best_class_probabilities > faceConfThreshold:
                return self.class_names[best_class_indices[0]]
            else:
                return 'Unknown'

    def get_confidence(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            if best_class_probabilities > faceConfThreshold:
                return str(round(best_class_probabilities[0],2))
            else:
                return str("0")

    def get_face_landmarks(self, face):
        return face.face_landmarks


    def check_frontal(self, face):
        """
        facial_landmarks contains a numpy n-D array of 10 elements.
            1. Mouth Left Corner -> [0],[9]
            2. Mouth right corner -> [1],[8]
            3. Nose -> [2],[7]
            4. Left eye -> [3],[6]
            5. Right eye -> [4],[5]
        """

        # we consider the difference in X-axis only
        rightEyeToMidNose = face.face_landmarks[4] - face.face_landmarks[2]
        leftEyeToMidNose = face.face_landmarks[2] - face.face_landmarks[3]
        xDiffOnRightAndLeftEye = face.face_landmarks[4] - face.face_landmarks[3]

        if (abs(leftEyeToMidNose - rightEyeToMidNose) / xDiffOnRightAndLeftEye) <= frontal_face_threshold:
            is_frontal = True
        else:
            is_frontal = False

        return is_frontal


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face, thread_safe):
        # Get input and output tensors
        if thread_safe:
            with self.sess.graph.as_default():
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # prewhiten_face = facenet.prewhiten(face.image)
                # cv2.imshow("prewhiten_face ", facenet.prewhiten(face.image))

                # using fixed image standardization
                img_std = (np.float32(face.image) - 127.5) / 128.0
                # cv2.imshow("img_std ", img_std)

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: [img_std], phase_train_placeholder: False}
                # feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        else:
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # prewhiten_face = facenet.prewhiten(face.image)
            # cv2.imshow("prewhiten_face ", face.image)

            # using fixed image standardization
            img_std = (np.float32(face.image) - 127.5) / 128.0
            # cv2.imshow("img_std ", img_std)

            # Run forward pass to calculate embeddings
            # feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
            feed_dict = {images_placeholder: [img_std], phase_train_placeholder: False}

        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = cfg.iFaceMinsize  # minimum size of face
    threshold = cfg.enumFaceDetectThreshold  # three steps's threshold
    factor = cfg.fScaleFactor  # scale factor

    def __init__(self, face_crop_size=cfg.iFaceCropSize, face_crop_margin=cfg.iFaceCropMargin):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess_face = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess_face.as_default():
                return face_recognition.align.detect_face.create_mtcnn(sess_face, None)

    def find_faces(self, image):
        faces = []
        """
        facial_landmarks contains the 5 facial landmark points (left mouth corner, right mouth corner, nose, left eye
        right eye. facial_landmarks contains a numpy n-D array of 10 elements.
                            1. Mouth Left Corner -> [0],[9]
                            2. Mouth right corner -> [1],[8]
                            3. Nose -> [2],[7]
                            4. Left eye -> [3],[6]
                            5. Right eye -> [4],[5]
        """
        bounding_boxes, facial_landmarks = face_recognition.align.detect_face.detect_face(image, self.minsize,
                                                                           self.pnet, self.rnet, self.onet,
                                                                           self.threshold, self.factor)

        """
        In case where multiple faces are detected we need to flatten the n-D array to extract the facial landmark points
        corresponding to the bounding box. Hence we transpose the matrix.
        """
        facial_landmarks = facial_landmarks.T

        if bounding_boxes is not None:
            face = Face()

        for bb in bounding_boxes:
            face.container_image = image
            img_size = np.asarray(image.shape)[0:2]
            # confidence value of the detected face
            face.detection_conf = round(bb[4], 2)

            face.bounding_box = np.zeros(4, dtype=np.int32)

            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])

            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

        for landmarkPt in facial_landmarks:
            face.face_landmarks = np.zeros(10, dtype=np.int32)
            for i in range(0,10):
                face.face_landmarks[i] = np.round(landmarkPt[i])

            faces.append(face)

        return faces
