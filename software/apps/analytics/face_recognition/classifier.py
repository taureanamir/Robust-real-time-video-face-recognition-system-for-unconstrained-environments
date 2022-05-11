"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def main(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:

            np.random.seed(seed=args.seed)

            TP = 0
            FP = 0
            FN = 0
            TN = 0

            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                    args.nrof_train_images_per_class)
                if (args.mode == 'TRAIN'):
                    dataset = train_set
                elif (args.mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            num_of_images = len(paths)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % num_of_images)
            # print("paths: ", paths)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # print("-------------------------------------------")
            # print(embeddings)
            # print("-------------------------------------------")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                if args.use_fixed_image_standardization:
                    images = facenet.load_data(paths_batch, False, False, args.image_size, do_prewhiten=False)
                    # Do fixed image standardization
                    images = (images - 127.5) / 128.0
                else:
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                # print("Images: ", images)
                # print("paths_batch: ", paths_batch)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                # print("embeddings of ID: ", labels)
                # print(emb_array)
                # print("*************************************************")
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                # print("predictions: ", predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                # print("best_class_indices: ", best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                # print("best_class_probabilities: ", best_class_probabilities)

                threshold = 0.80

                for i in range(num_of_images):
                    # _, ground_truth = os.path.split(os.path.dirname(paths[i]))
                    print(os.path.basename(os.path.normpath(paths[i])))

                for i in range(num_of_images):
                    _, ground_truth = os.path.split(os.path.dirname(paths[i]))


                    if best_class_probabilities[i] > threshold:
                        if class_names[best_class_indices[i]] == ground_truth:
                            TP += 1
                            print("TP | Ground Truth : %s  |  Prediction : %s   |   Confidence : %.3f" % (
                            ground_truth, class_names[best_class_indices[i]], best_class_probabilities[i]))
                            # print('%4d  %s: %.3f ' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                        elif class_names[best_class_indices[i]] != ground_truth:
                            FP += 1
                            print("FP | Ground Truth : %s  |  Prediction : %s   |   Confidence : %.3f" % (
                            ground_truth, class_names[best_class_indices[i]], best_class_probabilities[i]))
                            # print('%4d  %s: %.3f ' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    else:
                        if class_names[best_class_indices[i]] == ground_truth:
                            FN += 1
                            print("FN | Ground Truth : %s  |  Prediction : %s   |   Confidence : %.3f | Unknown" % (
                            ground_truth, class_names[best_class_indices[i]], best_class_probabilities[i]))
                            # print('%4d  %s: %.3f %s' % (
                            #     i, class_names[best_class_indices[i]], best_class_probabilities[i], '--> Unknown'))
                        else:
                            TN += 1
                            print("TN | Ground Truth : %s  |  Prediction : %s   |   Confidence : %.3f | Unknown" % (
                            ground_truth, class_names[best_class_indices[i]], best_class_probabilities[i]))
                            # print('%4d  %s: %.3f %s' % (
                            #     i, class_names[best_class_indices[i]], best_class_probabilities[i], '--> Unknown'))


                accuracy = np.mean(np.equal(best_class_indices, labels))
                # print('Accuracy: %.3f' % accuracy)
                print("TP : %d , FP : %d , FN : %d, TN : %d" % (TP, FP, FN, TN))
                print("# of total examples: ", len(best_class_indices))

                accuracy_from_TP_TN = (TP + TN)/num_of_images
                precision = TP/ (TP+FP)
                recall = TP/ (TP+FN)
                tpr = recall
                fpr = FP / (FP+TN)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                print("Precision : %.3f | Recall : %.3f | Accuracy : %.3f " %(precision, recall, accuracy_from_TP_TN))
                print("True positive rate : %.3f | False positive rate : %.3f " %(tpr, fpr))
                print("F1 score: %.3f" %(f1_score))


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
