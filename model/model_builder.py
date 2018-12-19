import model.model_config as model_config
import tensorflow as tf

class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.Conv2d_32 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    self.Conv2d_64 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')

    self.MaxPool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    self.Dense_512 = tf.keras.layers.Dense(units=512, activation='relu')
    self.Dense_output = tf.keras.layers.Dense(model_config._NUM_CLASSES, activation='softmax')

    self.Flatten = tf.keras.layers.Flatten()
    self.Dropout = tf.keras.layers.Dropout(0.5)

  def call(self, input):

    result = self.Conv2d_32(input)
    result = self.MaxPool2D(result)

    result = self.Conv2d_64(result)


    result = self.Flatten(result)
    result = self.Dense_512(result)
    result = self.Dropout(result)
    output = self.Dense_output(result)

    return output

