import tensorflow as tf
from tqdm import tqdm
import logging

class ConvNet:
    def __init__(self, train_data=None, test_data=None):
        if train_data==None and test_data==None:
            return

        assert(train_data.get_is_pre_processed())
        assert(test_data.get_is_pre_processed())

        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.train_data = train_data
        self.test_data = test_data
        self.image_width = train_data.get_image_width()
        self.image_height = train_data.get_image_height()
        self.color_channels = train_data.get_color_channels()
        self.n_classes = train_data.get_n_classes()

        self.logger.info("images: %sx%s chan: %s n_classes: %s" % (self.image_width, self.image_height, self.color_channels, self.n_classes))

        self.layer_width = {
            'layer_1': 32,
            'layer_2': 64,
            'layer_3': 128,
            'fully_connected_1': 1024,
            'fully_connected_2': 1024
        }

        self.weights = {
            # filter_size_width, filter_size_height, color_channels, k_output

            # 1x1x32 layer
            'layer_1': tf.Variable(tf.truncated_normal([5, 5, self.color_channels, self.layer_width['layer_1']])),

            # 5x5x64 layer
            'layer_2': tf.Variable(tf.truncated_normal([5, 5, self.layer_width['layer_1'], self.layer_width['layer_2']])),

            # 5x5x128 layer
            'layer_3': tf.Variable(tf.truncated_normal([5, 5, self.layer_width['layer_2'], self.layer_width['layer_3']])),

            'fully_connected_1': tf.Variable(tf.truncated_normal([2048, self.layer_width['fully_connected_1']])),
            'fully_connected_2': tf.Variable(tf.truncated_normal([1024, self.layer_width['fully_connected_2']])),
            'out': tf.Variable(tf.truncated_normal([self.layer_width['fully_connected_2'], self.n_classes]))
        }

        self.biases = {
            'layer_1': tf.Variable(tf.zeros(self.layer_width['layer_1'])),
            'layer_2': tf.Variable(tf.zeros(self.layer_width['layer_2'])),
            'layer_3': tf.Variable(tf.zeros(self.layer_width['layer_3'])),
            'fully_connected_1': tf.Variable(tf.zeros(self.layer_width['fully_connected_1'])),
            'fully_connected_2': tf.Variable(tf.zeros(self.layer_width['fully_connected_2'])),
            'out': tf.Variable(tf.zeros(self.n_classes))
        }

    def conv2d(self, x, W, b, keep_prob, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, k=3, strides=2):
        return tf.nn.max_pool(
            x,
            ksize=[1, k, k, 1],
            strides=[1, strides, strides, 1],       # [batch, input_height, input_width, input_channels]
            padding='SAME')

    # Create model
    def conv_net(self, activation, x, weights, biases, keep_prob):
        layer = 1

        conv1 = self.conv2d(x, weights['layer_1'], biases['layer_1'], keep_prob)
        self.logger.info("\tconv2d Layer %s: %s", layer, conv1.get_shape())
        conv1 = self.maxpool2d(conv1, strides=1)
        self.logger.info("\tmaxpool2d Layer %s: %s", layer, conv1.get_shape())
        conv1 = tf.nn.dropout(conv1, keep_prob)

        layer += 1

        # Layer 2
        conv2 = self.conv2d(conv1, weights['layer_2'], biases['layer_2'], keep_prob)
        self.logger.info("\tconv2d Layer %s: %s", layer, conv2.get_shape())
        conv2 = self.maxpool2d(conv2)
        self.logger.info("\tmaxpool2d Layer %s: %s", layer, conv2.get_shape())
        conv2 = tf.nn.dropout(conv2, keep_prob)

        layer += 1

        # Layer 3
        conv3 = self.conv2d(conv2, weights['layer_3'], biases['layer_3'], keep_prob)
        self.logger.info("\tconv2d Layer %s: %s", layer, conv3.get_shape())
        conv3 = self.maxpool2d(conv3, strides=4)
        self.logger.info("\tmaxpool2d Layer %s: %s", layer, conv3.get_shape())
        conv3 = tf.nn.dropout(conv3, keep_prob)

        layer = 1

        # Fully connected layer

        conv3_flat = tf.reshape(conv3, [-1, weights['fully_connected_1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(conv3_flat, weights['fully_connected_1']), biases['fully_connected_1'])
        self.logger.info("\tfc1 Layer %s: %s", layer, fc1.get_shape())
        fc1 = self.get_activation_func(activation, fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob)

        layer += 1

        fc2 = tf.add(tf.matmul(fc1, weights['fully_connected_2']), biases['fully_connected_2'])
        self.logger.info("\tfc2 Layer %s: %s", layer, fc2.get_shape())
        fc2 = self.get_activation_func(activation, fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob)

        layer += 1

        # Output Layer - class prediction
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        self.logger.info("\tout Layer %s: %s", layer, out.get_shape())

        return out

    def get_activation_func(self, activation, fc):
        if activation == "tanh":
            self.logger.info("\tactivation funciton=tanh")
            return tf.nn.tanh(fc)
        elif activation == "sigmoid":
            self.logger.info("\tactivation funciton=sigmoid")
            return tf.nn.sigmoid(fc)
        else:
            self.logger.info("\tactivation funciton=relu")
            return tf.nn.relu(fc)

    def get_optimizer_tensor(self, cost, optimizer, learning_rate, current_iteration):
        if optimizer.lower() == "adagradoptimizer":
            self.logger.info("optimizer=AdagradOptimizer")        
            return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimizer.lower() == "momentumoptimizer":
            self.logger.info("optimizer=MomentumOptimizer")       
            return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=1.0).minimize(cost)
        elif optimizer.lower() == "adamoptimizer":
            self.logger.info("optimizer=AdamOptimizer")       
            return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)
        elif optimizer.lower() == "gradientdescentoptimizer":
            self.logger.info("optimizer=GradientDescentOptimizer")       
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    def train(self, training_epochs, learning_rate, activation, optimizer, batch_size, dropout):
        self.logger.info("training_epochs: %s, learning_rate: %s, batch_size: %s, dropout: %s" % (training_epochs, learning_rate, batch_size, dropout))

        # tf input
        self.logger.info("create input placeholders ...")
        x = tf.placeholder("float", [None, self.image_width, self.image_height, self.color_channels])
        y = tf.placeholder("float", [None, self.n_classes])
        keep_prob = tf.placeholder(tf.float32)

        self.logger.info("create model ...")
        # create our model
        logits = self.conv_net(activation,  x, self.weights, self.biases, keep_prob)

        self.logger.info("create cost (or loss) and optimizer ...")
        # cost (or loss) and optimizer

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = self.get_optimizer_tensor(cost, optimizer=optimizer, learning_rate=learning_rate, current_iteration=0)

        self.logger.info("create prediction and accuracy tensors ...")
        # prediction and accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.logger.info("init ...")
        # Initializing the variables
        #init = tf.initialize_all_variables()

        self.logger.info("launch graph ...")
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            #sess.run(init)
            tf.global_variables_initializer().run()

            # Training cycle
            for epoch in range(training_epochs):
                self.train_data.reset_batch_to_start()
                batch_x, batch_y, valid_features, valid_labels = None, None, None, None
                self.logger.info('Epoch {:>2}/{}'.format(epoch + 1, training_epochs))

                processed_count = 0
                for batch_i in tqdm(range(self.train_data.total_batches(batch_size)), desc='Epoch {:>2}/{}'.format(epoch + 1, training_epochs), unit='batches'):
                    if self.train_data.has_next_batch():
                        batch_x, batch_y, valid_features, valid_labels = self.train_data.get_next_batch(batch_size)
                        # Run optimization op (backprop) and cost op (to get loss value)
                        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                        processed_count += len(batch_x)

                self.logger.info("Processed %s of %s" % (processed_count, len(self.train_data.get_train_features())))
                # Display logs per epoch step
                self.logger.info("Calc loss...")
                loss, training_acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                validation_acc = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})

                self.logger.info("Epoch %04d cost=%s, train_accurarcy=%s, validation_acc=%s" % ((epoch+1), "{:.9f}".format(loss), "{:.5f}".format(training_acc), "{:.5f}".format(validation_acc)))

            self.logger.info("Optimization Finished! Saving model ...")
            saver = tf.train.Saver()
            saver.save(sess, "dnn-traffic-signs-trained-model.dat")

            if self.test_data:
                # Test model
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                self.logger.info("Accuracy: %s" % (accuracy.eval({x: self.test_data.get_features(), y: self.test_data.get_labels(), keep_prob: 1.0})))

    def restore(self):
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph('dnn-traffic-signs-trained-model.dat.meta')
        new_saver.restore(sess, 'dnn-traffic-signs-trained-model.dat')
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.get_shape())
