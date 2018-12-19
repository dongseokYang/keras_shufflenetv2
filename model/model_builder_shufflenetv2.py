import model.model_config as model_config
import tensorflow as tf

class ShuffleNetV2Model(tf.keras.Model):
    def __init__(self):
        super(ShuffleNetV2Model, self).__init__()

        self.input_and_out_channel_sv2_1 = 64
        self.half_channel = self.input_and_out_channel_sv2_1 // 2

        self.Conv2d_32 = tf.keras.layers.Conv2D(self.input_and_out_channel_sv2_1, (3, 3), activation='relu', padding='same')

        self.Conv2d_1_1 = tf.keras.layers.Conv2D(self.half_channel, (1, 1), padding='same')

        self.SeparableConv2D = tf.keras.layers.SeparableConv2D(self.half_channel, (3,3), padding='same')

        ##common
        self.MaxPool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.Dense_512 = tf.keras.layers.Dense(units=512, activation='relu')
        self.Dense_output = tf.keras.layers.Dense(model_config._NUM_CLASSES, activation='softmax')

        self.Batchnorm = tf.keras.layers.BatchNormalization()

        self.Flatten = tf.keras.layers.Flatten()
        #self.Dropout = tf.keras.layers.Dropout(0.5, name="dropout")

        self.Dropout = tf.keras.layers.Dropout(0.5)

    def call(self, input):
   
        with tf.name_scope("pre_step"):
            result = self.Conv2d_32(input)
            result = self.MaxPool2D(result)

        with tf.name_scope("shuffle_unit1"):
            top, bottom = tf.split(result, num_or_size_splits=2, axis=3)

            top = self.conv_bn_relu(top)
            top = self.depthwise_conv_bn(top)
            top = self.conv_bn_relu(top)

            result = tf.concat([top, bottom], axis=3)

            result = self.shuffle_unit(result)

        with tf.name_scope("post_step"):
            
            result = self.Flatten(result)
            result = self.Dense_512(result)
            result = self.Dropout(result)
            output = self.Dense_output(result)

        return output


    def conv_bn_relu(self, input):

        result = self.Conv2d_1_1(input)
        result = self.Batchnorm(result)

        return result

    def depthwise_conv_bn(self, input):

        result = self.SeparableConv2D(input)
        result = self.Batchnorm(result)

        return result

    def shuffle_unit(self, input):

        n, h, w, c = input.get_shape().as_list()

        result = tf.reshape(input, shape=tf.convert_to_tensor([tf.shape(input)[0], h, w, model_config._GROUPS, c // model_config._GROUPS]))
        result = tf.transpose(result, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        result = tf.reshape(result, shape=tf.convert_to_tensor([tf.shape(result)[0], h, w, c]))

        return result
