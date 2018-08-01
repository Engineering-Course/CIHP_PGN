# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf

class PGNModel(Network):
    def setup(self, is_training, n_classes, keep_prob):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
        '''
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu', 
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu', 
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

        (self.feed('res4a_relu', 
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu', 
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu', 
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu', 
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu', 
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu', 
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu', 
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu', 
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu', 
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu', 
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu', 
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu', 
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu', 
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu', 
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu', 
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu', 
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu', 
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu', 
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu', 
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu', 
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu', 
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu', 
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu'))

######################################parsing networks################################################################
        (self.feed('res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu'))
        
        (self.feed('res3b3_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch1'))
        (self.feed('res4b22_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch2'))
        (self.feed('res5c_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch3'))

        (self.feed('parsing_branch1', 'parsing_branch2', 'parsing_branch3')
             .concat(axis=3, name='parsing_branch_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_branch'))

        (self.feed('parsing_branch')
             .pyramid_pooling(512, 60, name='parsing_pp1'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 30, name='parsing_pp2'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 20, name='parsing_pp3'))
        (self.feed('parsing_branch')
             .pyramid_pooling(512, 10, name='parsing_pp4'))

        (self.feed('parsing_branch', 'parsing_pp1', 'parsing_pp2', 'parsing_pp3', 'parsing_pp4')
             .concat(axis=3, name='parsing_pp_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_pp_conv')
             .dropout(keep_prob, name='parsing_pp_dropout')
             .conv(3, 3, n_classes, 1, 1, biased=True, relu=False, name='parsing_fc'))

######################################edge networks################################################################

        (self.feed('res3b3_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch1'))
        (self.feed('res4b22_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch2'))
        (self.feed('res5c_relu')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch3'))

        (self.feed('edge_branch1', 'edge_branch2', 'edge_branch3')
             .concat(axis=3, name='edge_branch_concat')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_branch'))

        (self.feed('edge_branch')
            .pyramid_pooling(512, 60, name='edge_pp1'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 30, name='edge_pp2'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 20, name='edge_pp3'))
        (self.feed('edge_branch')
            .pyramid_pooling(512, 10, name='edge_pp4'))

        (self.feed('edge_branch', 'edge_pp1', 'edge_pp2', 'edge_pp3', 'edge_pp4')
             .concat(axis=3, name='edge_pp_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_pp_conv')
             .dropout(keep_prob, name='edge_pp_dropout')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='edge_fc'))

##################intermidium supervision######################################

        (self.feed('edge_branch3')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res5'))
        (self.feed('edge_branch3')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res5'))
        (self.feed('fc1_edge_c0_res5', 'fc1_edge_c1_res5', 'fc1_edge_c2_res5', 'fc1_edge_c3_res5', 'fc1_edge_c4_res5')
             .add(name='fc1_edge_res5'))

        (self.feed('edge_branch2')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res4'))
        (self.feed('edge_branch2')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res4'))
        (self.feed('fc1_edge_c0_res4', 'fc1_edge_c1_res4', 'fc1_edge_c2_res4', 'fc1_edge_c3_res4', 'fc1_edge_c4_res4')
             .add(name='fc1_edge_res4'))

        (self.feed('edge_branch1')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='fc1_edge_c0_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 2, padding='SAME', relu=False, name='fc1_edge_c1_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 4, padding='SAME', relu=False, name='fc1_edge_c2_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 8, padding='SAME', relu=False, name='fc1_edge_c3_res3'))
        (self.feed('edge_branch1')
             .atrous_conv(3, 3, 1, 16, padding='SAME', relu=False, name='fc1_edge_c4_res3'))
        (self.feed('fc1_edge_c0_res3', 'fc1_edge_c1_res3', 'fc1_edge_c2_res3', 'fc1_edge_c3_res3', 'fc1_edge_c4_res3')
             .add(name='fc1_edge_res3'))

##################################### refine networks ############################################################
        (self.feed('parsing_pp_conv')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='parsing_fea'))
        (self.feed('parsing_fc')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='parsing_remap'))
        (self.feed('edge_pp_conv')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='edge_fea'))
        (self.feed('edge_fc')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='edge_remap'))

        (self.feed('parsing_fea', 'parsing_remap', 'edge_fea', 'edge_remap')
             .concat(axis=3, name='parsing_rf_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_rf'))

        (self.feed('parsing_rf')
             .pyramid_pooling(512, 60, name='parsing_rf_pp1'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 30, name='parsing_rf_pp2'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 20, name='parsing_rf_pp3'))
        (self.feed('parsing_rf')
             .pyramid_pooling(512, 10, name='parsing_rf_pp4'))

        (self.feed('parsing_rf', 'parsing_rf_pp1', 'parsing_rf_pp2', 'parsing_rf_pp3', 'parsing_rf_pp4')
             .concat(axis=3, name='parsing_rf_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='parsing_rf_conv')
             .dropout(keep_prob, name='parsing_rf_dropout')
             .conv(3, 3, n_classes, 1, 1, biased=True, relu=False, name='parsing_rf_fc'))

        (self.feed('edge_fea', 'edge_remap', 'parsing_fea', 'parsing_remap')
             .concat(axis=3, name='edge_rf_concat')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_rf'))

        (self.feed('edge_rf')
            .pyramid_pooling(512, 60, name='edge_rf_pp1'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 30, name='edge_rf_pp2'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 20, name='edge_rf_pp3'))
        (self.feed('edge_rf')
            .pyramid_pooling(512, 10, name='edge_rf_pp4'))

        (self.feed('edge_rf', 'edge_rf_pp1', 'edge_rf_pp2', 'edge_rf_pp3', 'edge_rf_pp4')
             .concat(axis=3, name='edge_rf_out')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='edge_rf_conv')
             .dropout(keep_prob, name='edge_rf_dropout')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='edge_rf_fc'))
