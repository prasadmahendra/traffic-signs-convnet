import matplotlib
import os
import sys
import math
import pickle
import logging
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class ImageData:
    def __init__(self, pickle_features_and_labels_file=None, debug=False):
        self.features_and_labels_file = pickle_features_and_labels_file
        self.features = {}
        self.labels = []
        self.debug = debug

        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if self.features_and_labels_file:
            with open(self.features_and_labels_file, mode='rb') as f:
                loaded = pickle.load(f)
                self.features = loaded['features']
                self.labels = loaded['labels']

            if self.debug:
                self.logger.info("debug: %s", self.debug)
                self.__take_subset_of_data()

            self.n_labels = len(self.labels)
        else:
            self.n_labels = 0

        self.n_labels_friendly_names = {}
        with open('data/signnames.csv', 'rt', encoding='utf8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.n_labels_friendly_names[row['ClassId']] = row['SignName']

        self.n_classes = len(self.n_labels_friendly_names)
        self.is_pre_processed = False
        self.hot_encoded_labels = False
        self.new_data_generated = False

    def __take_subset_of_data(self):
        # 25 images per label
        debug_features = []
        debug_labels = []
        seen_labels = {}
        for n in range(len(self.features)):
            if not seen_labels.get(self.labels[n]):
                seen_labels[self.labels[n]] = 1
            else:
                seen_labels[self.labels[n]] += 1

            if seen_labels[self.labels[n]] < 25:
                debug_features.append(self.features[n])
                debug_labels.append(self.labels[n])

        self.logger.info("features: %s", len(self.features))
        self.logger.info("labels: %s", len(self.labels))

        self.features = debug_features
        self.labels = debug_labels

        self.logger.info("debug features: %s", len(self.features))
        self.logger.info("debug labels: %s", len(self.labels))

    def type(self):
        raise NotImplementedError

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_features_len(self):
        return len(self.features)

    def get_is_pre_processed(self):
        return self.is_pre_processed

    def to_grayscale(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError

    def query_yes_no(self, question, default="yes"):
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

    def get_image_height(self):
        # todo assert on invalid feautres datastructure
        a_feature = self.features[0]
        return a_feature.shape[0]

    def get_image_width(self):
        # todo assert on invalid feautres datastructure
        a_feature = self.features[0]
        return a_feature.shape[1]

    def get_color_channels(self):
        # todo assert on invalid feautres datastructure
        a_feature = self.features[0]
        height, width, chan = a_feature.shape
        return chan

    def get_n_classes(self):
        return self.n_classes

    def one_hot_encode_labels(self):
        if not self.hot_encoded_labels:
            """Turn labels into numbers and apply One-Hot Encoding"""
            encoder = LabelBinarizer()
            encoder.fit(self.labels)
            self.labels = encoder.transform(self.labels)

            # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
            self.labels = self.labels.astype(np.float32)
            self.hot_encoded_labels = True

    def dump_info(self, prompt_for_images=False):
        print("**** %s dataset ****" % (self.type()))

        # todo: assert self.features len > 0
        # todo: assert self.labels len > 0 and == features len
        a_feature = self.features[0]

        print("features: ")
        print("\tcount: %s" % (len(self.features)))
        print("\teach image: %s" % (str(a_feature.shape)))

        unique_train_labels, unique_train_label_indices = np.unique(self.labels, return_index=True)

        print("labels: ")
        print("\tcount: %s" % (len(self.labels)))
        print("\tunique labels (classes): %s" % (len(unique_train_labels)))

        print("Label ID\tName")
        for label_id in sorted(self.n_labels_friendly_names.keys()):
            print("%s\t%s" % (label_id, self.n_labels_friendly_names.get(label_id)))

        display = True
        if prompt_for_images:
            display = self.query_yes_no("Display representative sample images?", "no")

        if display:
            grids = gridspec.GridSpec(math.ceil(len(unique_train_labels) / 10), 10, wspace=0.0, hspace=0.0)
            fig = plt.figure(figsize=(10, math.ceil(len(unique_train_labels) / 10)))
            fig.suptitle('All %s classes representative sample' % (self.type()), fontsize=20)
            uq_labels = tqdm(range(len(unique_train_labels)), desc='Loading a sample image per label', unit='training classes')

            for n in uq_labels:
                grid = grids[n]
                unqiue_label_idx = unique_train_label_indices[n]
                traffic_sign_class_img = self.features[unqiue_label_idx]
                ax = plt.Subplot(fig, grid)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(traffic_sign_class_img, cmap=plt.cm.gray)
                fig.add_subplot(ax)

            plt.show()

    def to_grayscale(self, image):
        """"Convert to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def to_yuvspace(self, image):
        """"Convert to YUV colorspace"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


    def normalize(self, image):
        return cv2.normalize(image, image, alpha=0.1, beta=0.9, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    def mean_subtraction(self, image):
        image = image.astype(dtype='float64')
        image -= np.mean(image, dtype='float64', axis=0)
        return image


    def transform_image(self, img, ang_range, shear_range, trans_range):
        # Rotation

        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        img = cv2.warpAffine(img, Trans_M, (cols, rows))
        img = cv2.warpAffine(img, shear_M, (cols, rows))

        return img

    def pre_process_image(self, image):
        image = self.to_yuvspace(image)
        image = self.mean_subtraction(image)
        image = self.normalize(image)
        return image

    def load_image_from_file(self, image_file):
        if os.path.isfile(image_file):
            img = Image.open(image_file)
            img = img.resize((32, 32), Image.ANTIALIAS)
            img = np.asarray(img)
            img = self.pre_process_image(img)
            height, width, channels = img.shape
            self.logger.info("Image loaded: %s" % (str(img.shape)))
            return img, height, width, channels
        else:
            raise FileNotFoundError(image_file)

class TrainData(ImageData):
    def __init__(self, pickle_features_and_labels_file, debug=False):
        super(TrainData, self).__init__(pickle_features_and_labels_file, debug)

        # vars used in batch shuffle/generate get_next_batch() func
        self.train_valid_split = False
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.train_features = None
        self.valid_features = None
        self.train_labels = None
        self.valid_labels = None

    def type(self):
        return "training"

    def get_train_features(self):
        self.init_train_valid_set_if()
        return self.train_features

    def get_train_labels(self):
        self.init_train_valid_set_if()
        return self.train_labels

    def get_valid_features(self):
        self.init_train_valid_set_if()
        return self.valid_features

    def get_valid_labels(self):
        self.init_train_valid_set_if()
        return self.valid_labels

    def init_train_valid_set_if(self):
        if not self.train_valid_split:
            if self.debug:
                self.logger.warning("WARNING: debug mode. train and validation data not split. Risks over fitting!")
                self.train_features, self.valid_features, self.train_labels, self.valid_labels = self.features, self.features, self.labels, self.labels
            else:
                self.train_features, self.valid_features, self.train_labels, self.valid_labels = train_test_split(self.features, self.labels, test_size=0.05, random_state=832289)
            self.train_valid_split = True
            logging.info("train_features: %s" % (len(self.train_features)))
            logging.info("train_labels: %s" % (len(self.train_labels)))
            logging.info("valid_features: %s" % (len(self.valid_features)))
            logging.info("valid_labels: %s" % (len(self.valid_labels)))

    def reset_batch_to_start(self):
        self.index_in_epoch = 0

    def get_train_batches_count(self, batch_size):
        return math.ceil(len(self.get_train_features()) / batch_size)

    def has_next_batch(self):
        ret = self.index_in_epoch < len(self.get_train_features())
        return ret

    def get_next_batch(self, batch_size, shuffle=False):
        assert batch_size <= len(self.get_train_features())
        self.init_train_valid_set_if()

        start = self.index_in_epoch
        if self.index_in_epoch + batch_size >= len(self.get_train_features()):
            self.index_in_epoch = len(self.get_train_features())
        else:
            self.index_in_epoch += batch_size

        end = self.index_in_epoch
        return self.train_features[start:end], self.train_labels[start:end], self.valid_features, self.valid_labels

    def get_epochs_completed(self):
        return self.epochs_completed

    def pre_process(self, gen_variants=False):
        if not self.is_pre_processed:
            self.logger.info("Preprocess %s data ..." % (self.type()))
            if gen_variants:
                self.generate_new_data()

            count = 0
            features_new = np.ndarray(shape=(len(self.features), 32, 32, 3), dtype=float)
            self.logger.info("convert images to YUV color space ...")
            self.logger.info("mean subtract (zero center) all features ...")
            self.logger.info("min-max rescale/normalize image pixel values ...")
            for image in tqdm(self.features, desc="Pre-process %s images" % (self.type()), unit='image'):
                image = self.pre_process_image(image)
                features_new[count] = image
                count += 1

            self.features = features_new
            self.one_hot_encode_labels()

            self.logger.info("Preprocessed %s features: %s" % (self.type(), len(self.features)))
            self.logger.info("1-hot encoded %s labels: %s" % (self.type(), len(self.labels)))
            self.is_pre_processed = True

    def generate_new_data(self):
        if not self.new_data_generated:
            self.logger.info("generating jittered images ...")
            new_features, new_labels = self.__generate_new_data()

            self.features = np.concatenate((self.features, new_features))
            self.labels = np.concatenate((self.labels, new_labels))
            self.n_labels = len(self.labels)
            self.new_data_generated = True
        else:
            self.logger.info("generating jittered images already present!")

    def __generate_new_data(self):
        new_features = []
        new_labels = []
        count = 0
        new_count = 0
        for image in tqdm(self.features, desc='Generate image variants', unit='image'):
            for n in range(5):
                new_features.append(self.transform_image(image,20,10,5))
                new_labels.append(self.labels[count])
                new_count += 1
            count += 1

        self.logger.info("generated jittered images: %s" % (len(new_features)))
        self.logger.info("generate jittered image labels: %s" % (len(new_labels)))
        return np.asarray(new_features), np.asarray(new_labels)

    def run_self_tests(self):
        # batching tests ..
        batch_size = 1000
        ret_count = 0
        while self.has_next_batch():
            batch_x, batch_y, valid_features, valid_labels = self.get_next_batch(batch_size)
            ret_count += len(batch_x)
        assert(ret_count == len(self.train_features))

        self.logger.info("Display fake data ...")
        new_features, new_labels = self.__generate_new_data()
        self.__display_image_data(new_features, new_labels)

        self.logger.info("Testing hot 1 encoding ...")
        # test one_hot_encode_labels
        old_labels = self.labels
        self.one_hot_encode_labels()

        for n in tqdm(range(len(self.labels)), desc='Testing hot-1-encodings', unit='label'):
            label = old_labels[n]
            hot_1_count = 0
            hot_0_count = 0
            for n2 in range(len(self.labels[n])):
                if n2 == label:
                    assert(self.labels[n][n2] == 1)
                    hot_1_count += 1
                else:
                    assert (self.labels[n][n2] == 0)
                    hot_0_count += 1

            assert(hot_0_count == self.n_classes - 1)
            assert(hot_1_count == 1)

        self.logger.info("Testing grayscale conversion ...")
        sample_image = self.to_grayscale(self.features[0])
        flatten = np.reshape(sample_image, (self.get_image_height() * self.get_image_width()))

        self.logger.info("Testing normalization (min/max encoding) ...")
        sample_image = self.normalize(self.features[0])
        flatten = np.reshape(sample_image, (self.get_image_height() * self.get_image_width() * 3,))
        if __name__ == '__main__':
            for pixel_data in flatten:
                if not(pixel_data >= 0.1 and pixel_data <= 0.9001):
                    logging.warning("mix/max (0.1 - 0.9) encoded pixel value is: ", pixel_data)
                assert(pixel_data >= 0.1 and pixel_data <= 0.9001)

    def __display_image_data(self, features, labels):
        unique_labels, unique_label_indices = np.unique(labels, return_index=True)

        print("__display_fake_data")
        print("labels: ")
        print("\tcount: %s" % (len(unique_labels)))
        print("\tunique labels (classes): %s" % (len(unique_labels)))

        display = self.query_yes_no("Display images?", "no")

        if display:
            grids = gridspec.GridSpec(math.ceil(len(unique_labels) / 10), 10, wspace=0.0, hspace=0.0)
            fig = plt.figure(figsize=(10, math.ceil(len(unique_labels) / 10)))
            fig.suptitle('All %s classes representative sample' % (self.type()), fontsize=20)
            uq_labels = tqdm(range(len(unique_labels)), desc='Loading a sample image per label', unit='training classes')

            for n in uq_labels:
                grid = grids[n]
                unqiue_label_idx = unique_label_indices[n]
                traffic_sign_class_img = features[unqiue_label_idx]
                ax = plt.Subplot(fig, grid)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(traffic_sign_class_img, cmap=plt.cm.gray)
                fig.add_subplot(ax)

            plt.show()

    def __transform_test(self):
        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0.01, hspace=0.02)
        # set the spacing between axes.
        plt.figure(figsize=(12,12))
        for i in range(100):
            image = self.features[i]  # mpimg.imread('../data/stopsign.png')
            plt.imshow(image);
            plt.axis('off');

            ax1 = plt.subplot(gs1[i])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            img = self.transform_image(image,20,10,5)
            plt.subplot(10,10,i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    def __test_display_img(self, image, cmap="gray"):
        plt.imshow(image, cmap=cmap)
        plt.show()

class TestData(TrainData):
    def type(self):
        return "test"

    def pre_process(self, gen_variants=False):
        super(TestData, self).pre_process(gen_variants=False)       # don't generate fake data in Test set.


