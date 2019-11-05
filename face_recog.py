# face_recog.py

import face_recognition
import cv2
import camera
import os
import numpy as np
from cyclegan import model as cycleganmodel
import argparse
import tensorflow as tf
from cyclegan.module import *
from cyclegan.utils import *
from collections import namedtuple

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./cyclegan/checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./cyclegan/sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./cyclegan/test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()

args.which_direction = 'AtoB'
args.phase = 'test'
args.dataset_dir = 'chicken2simple'

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

sess = tf.Session(config=tfconfig)
# sess = tf.InteractiveSession()


# model = cycleganmodel.cyclegan(sess, args)

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        ############################
        # self.detection_graph = tf.Graph()
        # with self.detection_graph.as_default():
        #     od_graph_def = tf.GraphDef()
        #     x = tf.constant(1)
        #     # with tf.gfile.GFile(self.graph_file, 'rb') as fid:
        #     #     serialized_graph = fid.read()
        #     #     od_graph_def.ParseFromString(serialized_graph)
        #     #     tf.import_graph_def(od_graph_def, name='')
        #
        # self.sess = tf.Session(graph=self.detection_graph)
        # self.sess.run(x)
        # with tf.Session(config=tfconfig) as sess:
        #     self.model = cycleganmodel.cyclegan(sess, args)
        #     self.model.realtime_test_build()


        self.model = cycleganmodel.cyclegan(sess, args)
        self.model.realtime_test_build()

        # if args.use_resnet:
        #     self.generator = generator_resnet
        # else:
        #     self.generator = generator_unet
        #
        # OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
        #                               gf_dim df_dim output_c_dim is_training')
        # self.options = OPTIONS._make((args.batch_size, args.fine_size,
        #                               args.ngf, args.ndf, args.output_nc,
        #                               args.phase == 'train'))
        #
        # self.image_size = args.fine_size
        # self.input_c_dim = args.input_nc
        # self.output_c_dim = args.output_nc
        #
        # self.test_A = tf.placeholder(tf.float32,
        #                              [None, self.image_size, self.image_size,
        #                               self.input_c_dim], name='test_A')
        # self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")


    def __del__(self):
        del self.camera

    def get_frame(self):
        # Grab a single frame of video
        frame = self.camera.get_frame()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        # for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #
        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = self.sess.run(out_var, feed_dict={in_var: frame})
        # self.model.realtime_test_build()
        frame = self.model.realtime_test(frame)
        ret, jpg = cv2.imencode('.jpg', frame)

        # fake_img = self.sess.run((self.testA, self.test_B), feed_dict={(self.testA, self.test_B): sample_image})


        # sess = tf.InteractiveSession(config=tfconfig)
        # model = cycleganmodel.cyclegan(sess, args)

        # with tf.Session(config=tfconfig) as sess:
        #     model = cycleganmodel.cyclegan(sess, args)
        #     model.train(args) if args.phase == 'train' \
        #         else model.test(args)
        # with tf.Session(config=tfconfig) as sess:
        #     model = cycleganmodel.cyclegan(sess, args)
        #     model.test(args)
        return jpg.tobytes()

