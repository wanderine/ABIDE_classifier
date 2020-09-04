# import keras.backend as K
import os
import time
import datetime

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#import nibabel as nib

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.losses import binary_crossentropy as BC

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # Set seed from random number generator, for better comparisons
# from numpy.random import seed
# seed(123)


def plot_results(history):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()


#------------
# Load data
#------------

##

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        # 'file_name': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    image = tf.io.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [128, 128, 128, 1])
    # file_name = parsed_features['file_name']
    label = tf.reshape(parsed_features['label'], [1])
    label = parsed_features['label']

    return image, label

##

def create_dataset(filepath, subset, batch_size):

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    if subset == 'train' or subset == 'valid':
        dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    if subset == 'train':
        dataset = dataset.shuffle(buffer_size=2048)

    # Set the batchsize
    # dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.batch(batch_size=batch_size)

    # Bring your picture back in shape
    # image = tf.reshape(image, [-1, 192, 192, 192, 1])

    # Create a one hot array for your labels
    # label = tf.one_hot(label, NUM_CLASSES)
    # label = tf.cast(label, tf.float32)
    # label = tf.reshape(label, [-1, 1])
    if subset == 'train':
        dataset = dataset.prefetch(64)

    return dataset


data_folder = '/raid/andek67/ABIDE_classifier/data/ABIDE_128cubes_47000_tf'

tfrecord_files_train = ['%s/%s_%04i.tfrecord' % (data_folder, 'train', i) for i in range(1471)]
tfrecord_files_valid = ['%s/%s_%04i.tfrecord' % (data_folder, 'valid', i) for i in range(49)]
tfrecord_files_test = ['%s/%s_%04i.tfrecord' % (data_folder, 'test', i) for i in range(49)]
# tfrecord_files = ['%s_%03i.tfrecord' % ('test/range', i) for i in range(3)]

n_GPUs = 2

n_im_train = 47060
n_im_valid = 1540
n_im_test = 1560
batch_size = 64 * n_GPUs

nFilts = [4]
dropoutRates = [0.5, 0.6]
denseNodes = [25, 50, 100, 200, 500]
doubleFirsts = [False]
convLayers = 4

learning_rate=0.00001

# Loop over settings
for nFilt in nFilts:

    for dropoutRate in dropoutRates:

        for dense in denseNodes:

            for doubleFirst in doubleFirsts:

                if doubleFirst:
                    filename = 'cnn_real_47000_5conv3dlayers_' + str(nFilt) + 'filters_' + '1denselayer_' + str(dense) + 'nodes_' + str(dropoutRate) + 'dropoutrate_learningrate0.00001_doublefirst.txt' 
                else:
                    filename = 'cnn_real_47000_5conv3dlayers_' + str(nFilt) + 'filters_' + '1denselayer_' + str(dense) + 'nodes_' + str(dropoutRate) + 'dropoutrate_learningrate0.00001.txt' 

                if os.path.isfile(filename):
                    print("This combination is already done, skipping")
                    continue

                strategy = tf.distribute.MirroredStrategy()

                with strategy.scope():
                    
                    input = Input(shape=[128,128,128,1])

                    x = Conv3D(nFilt, kernel_size=(3,3,3), activation='relu', padding='same')(input)
                    x = BatchNormalization()(x)

                    if doubleFirst:
                        x = Conv3D(nFilt, kernel_size=(3,3,3), activation='relu', padding='same')(x)
                        x = BatchNormalization()(x)

                    x = MaxPooling3D(pool_size=(2,2,2))(x)

                    for i in range(convLayers):
                        x = Conv3D(nFilt*(2**(i+1)), kernel_size=(3,3,3), activation='relu', padding='same')(x)
                        x = BatchNormalization()(x)
                        x = MaxPooling3D(pool_size=(2,2,2))(x)

                    x = Flatten()(x)

                    for i in range(1):
                        x = Dense(dense, activation='relu')(x)
                        x = BatchNormalization()(x)
                        x = Dropout(dropoutRate)(x)

                    output = Dense(1, activation='sigmoid')(x)

                    model = Model(inputs=input, outputs=output)
                    model.compile(loss=BC, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

                model.summary()

                dataset_train = create_dataset(tfrecord_files_train, 'train', batch_size)
                dataset_valid = create_dataset(tfrecord_files_valid, 'valid', batch_size)
                dataset_test = create_dataset(tfrecord_files_test, 'test', batch_size)

                ## Train
                start_time = time.time()
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, mode='max')
                
                try:

                    history = model.fit(dataset_train, epochs=100, steps_per_epoch=n_im_train//batch_size, validation_data=dataset_valid, validation_steps=n_im_valid//batch_size, callbacks=[callback])

                    elapsed_time = time.time() - start_time
                    elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
                    print('Training time: ', elapsed_time_string)

                    score = model.evaluate(dataset_test, steps=n_im_test//batch_size)
                    print('Test loss: %.4f' % score[0])
                    print('Test accuracy: %.4f' % score[1])

                    # Save results to text file
                    accuracy = np.zeros((1,1))
                    accuracy[0] = score[1]
                    np.savetxt(filename,accuracy*100,fmt='%.4f')

                except:

                    print('Training failed')
                    accuracy = np.zeros((1,1))
                    accuracy[0] = 10000
                    np.savetxt(filename,accuracy,fmt='%.4f')

                finally:

                    del x
                    del model
                    del strategy
                    del dataset_train
                    del dataset_valid
                    del dataset_test



