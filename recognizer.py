#!/usr/bin/python

import argparse
import logging
import os
from trafficsigns.data import TestData
from trafficsigns.data import TrainData
from trafficsigns.convnet import ConvNet
from trafficsigns.convnet import PredictConvNet

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Traffic Signs Recoginizer')

parser.add_argument('-testdatafile', default="../data/test.p", help='Test data pickle file path')
parser.add_argument('-traindatafile', default="../data/train.p", help='Train data pickle file path')
parser.add_argument('-epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument('-learning_rate', default=0.01, type=float, help='Learning rate')
parser.add_argument('-optimizer', default="AdagradOptimizer", help='Optimizer (GradientDescentOptimizer, AdagradOptimizer (default), MomentumOptimizer, AdamOptimizer)')
parser.add_argument('-activation', default="relu", help='Activation funciton (relu (default), tanh, sigmoid)')
parser.add_argument('-batch_size', default=100, type=int, help='Batch size')
parser.add_argument('-dropout', default=1.0, type=float, help='Dropout')
parser.add_argument('-genfakedata', action="store_true", help='Generate additional data')
parser.add_argument('-input_image', default="../data/my-sample-images/t1.png", help="Input image to use for classification (required when cmd=predict")
parser.add_argument('-debug', action="store_true", help='Run in debug mode (smaller data sets for testing purposes)')
parser.add_argument('-cmd', default="traindnn", help='Recognizer commands', choices=['dumpdata', 'preprocess', 'traindnn', 'unittests', 'predict'])
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)

def run(args):
    logger.info("Running cmd: %s" % (args.cmd))
    debug = args.debug

    if args.cmd == "dumpdata":
        train_data = TrainData(os.path.abspath(args.traindatafile))
        train_data.dump_info(prompt_for_images=True)

        test_data = TestData(os.path.abspath(args.testdatafile))
        test_data.dump_info(prompt_for_images=True)
    elif args.cmd == "preprocess":
        train_data = TrainData(os.path.abspath(args.traindatafile), debug=debug)
        train_data.pre_process(gen_variants=args.genfakedata)
    elif args.cmd == "unittests":
        train_data = TrainData(os.path.abspath(args.traindatafile))
        train_data.run_self_tests()
    elif args.cmd == "traindnn":
        train_data = TrainData(os.path.abspath(args.traindatafile), debug=debug)
        train_data.pre_process(gen_variants=args.genfakedata)

        test_data = TestData(os.path.abspath(args.testdatafile), debug=debug)
        test_data.pre_process()

        nn = ConvNet(train_data, test_data)
        nn.train(training_epochs=args.epochs,
                 learning_rate=args.learning_rate,
                 activation=args.activation,
                 optimizer=args.optimizer,
                 batch_size=args.batch_size,
                 dropout=args.dropout)
    elif args.cmd == "predict":
        nn = PredictConvNet(args.input_image)
        nn.predict()

# python recognizer.py -cmd traindnn -traindatafile data/train.p -testdatafile data/test.p -epoch 250 -dropout=1.0 -batch_size 32 -activation relu -optimizer AdagradOptimizer -learning_rate 0.01 > dnn-output.txt 2>&1
run(args)
