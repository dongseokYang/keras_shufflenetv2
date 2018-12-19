import tensorflow as tf  # only work from tensorflow==1.9.0-rc1 and after
from model.model_builder_shufflenetv2 import ShuffleNetV2Model
from model.model_builder import MNISTModel
import model.model_config as model_config
import path_manager as pm



def training_pipeline_ShuffleNetV2(training_set, testing_set):

    model = ShuffleNetV2Model()  # your keras model here

    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    tb_hist = tf.keras.callbacks.TensorBoard(log_dir=pm.Tensorboard_path + '/graph_shufflenetv2', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(
        training_set.make_one_shot_iterator(),
        steps_per_epoch=len(x_train) // model_config._BATCH_SIZE,
        epochs=model_config._EPOCHS,
        validation_data=testing_set.make_one_shot_iterator(),
        validation_steps=len(x_test) // model_config._BATCH_SIZE,
        verbose=1, callbacks=[tb_hist])

    # display model config
    model.summary()

    model.save_weights(pm.Model_path + "/model_ShuffleNetV2.h5")
    print("Saved model to disk")

    with open(pm.Summary_path + '/report_ShuffleNetV2.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def training_pipeline_CNN(training_set, testing_set):

    model = MNISTModel()  # your keras model here

    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])

    tb_hist = tf.keras.callbacks.TensorBoard(log_dir=pm.Tensorboard_path + '/graph_cnn', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(
        training_set.make_one_shot_iterator(),
        steps_per_epoch=len(x_train) // model_config._BATCH_SIZE,
        epochs=model_config._EPOCHS,
        validation_data=testing_set.make_one_shot_iterator(),
        validation_steps=len(x_test) // model_config._BATCH_SIZE,
        verbose=1, callbacks=[tb_hist])

    # display model config
    model.summary()

    model.save_weights(pm.Model_path + "/model_CNN.h5")
    print("Saved model to disk")

    with open(pm.Summary_path + '/report_CNN.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def tfdata_generator(images, labels, is_training, batch_size=128):
    '''Construct a data generator using tf.Dataset'''

    def preprocess_fn(image, label):
        '''A transformation function to preprocess raw data
        into trainable input. '''
        x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
        y = tf.one_hot(tf.cast(label, tf.uint8), model_config._NUM_CLASSES)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    training_set = tfdata_generator(x_train, y_train, is_training=True, batch_size=model_config._BATCH_SIZE)
    testing_set = tfdata_generator(x_test, y_test, is_training=False, batch_size=model_config._BATCH_SIZE)


    training_pipeline_ShuffleNetV2(training_set, testing_set)
    training_pipeline_CNN(training_set, testing_set)