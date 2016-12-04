# traffic-signs-convnet

### Cmdline usage:

```sh
usage: recognizer.py [-h] [-testdatafile TESTDATAFILE]
                     [-traindatafile TRAINDATAFILE] [-epochs EPOCHS]
                     [-topk TOPK] [-learning_rate LEARNING_RATE]
                     [-optimizer OPTIMIZER] [-activation ACTIVATION]
                     [-batch_size BATCH_SIZE] [-dropout DROPOUT]
                     [-genfakedata] [-input_image INPUT_IMAGE] [-chkpt CHKPT]
                     [-debug]
                     [-cmd {dumpdata,preprocess,traindnn,unittests,predict}]
                     [-v]

Traffic Signs Recognizer

optional arguments:
  -h, --help            show this help message and exit
  -testdatafile TESTDATAFILE
                        Test data pickle file path
  -traindatafile TRAINDATAFILE
                        Train data pickle file path
  -epochs EPOCHS        Number of training epochs
  -topk TOPK            Return top k predictions (when cmd == predict)
  -learning_rate LEARNING_RATE
                        Learning rate
  -optimizer OPTIMIZER  Optimizer (GradientDescentOptimizer, AdagradOptimizer
                        (default), MomentumOptimizer, AdamOptimizer)
  -activation ACTIVATION
                        Activation function (relu (default), tanh, sigmoid)
  -batch_size BATCH_SIZE
                        Batch size
  -dropout DROPOUT      Dropout
  -genfakedata          Generate additional data
  -input_image INPUT_IMAGE
                        Input image to use for classification (required when
                        cmd=predict
  -chkpt CHKPT          Saved checkpoint file)
  -debug                Run in debug mode (smaller data sets for testing
                        purposes)
  -cmd {dumpdata,preprocess,traindnn,unittests,predict}
                        Recognizer commands
  -v, --verbose         Verbose output
```


### Examples

Train:
```sh
$ python recognizer.py -cmd traindnn -traindatafile data/train.p -testdatafile data/test.p -epoch 150 -dropout=1.0 -batch_size 32 -activation relu -optimizer AdagradOptimizer -learning_rate 0.01
```

Train w/ generated data:
```sh
$ python recognizer.py -cmd traindnn -traindatafile data/train.p -testdatafile data/test.p -epoch 150 -dropout=1.0 -batch_size 64 -activation relu -optimizer AdagradOptimizer -learning_rate 0.01 -genfakedata
```

Predicting
```sh
$ python recognizer.py -cmd predict -chkpt saved_data/dnn-traffic-signs-trained-model.dat -input_image data/my-sample-images/t2.png -topk 5
```