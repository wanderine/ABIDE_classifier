
import glob
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import load_single_volume

## Read image paths

dataset_path = '/raid/andek67/ABIDE_classifier/data/ABIDE_128cubes_augmented65'
out_path = '/raid/andek67/ABIDE_classifier/data/ABIDE_128cubes_47000_tf'

if not os.path.isdir(dataset_path):
    sys.exit(' Dataset ' + subfolder + ' does not exist')

# volume paths
controls_train_path = sorted(glob.glob(os.path.join(dataset_path, 'training', 'CONTROL*.nii.gz')))
asds_train_path = sorted(glob.glob(os.path.join(dataset_path, 'training', 'ASD*.nii.gz')))
controls_valid_path = sorted(glob.glob(os.path.join(dataset_path, 'validation', 'CONTROL*.nii.gz')))
asds_valid_path = sorted(glob.glob(os.path.join(dataset_path, 'validation', 'ASD*.nii.gz')))
controls_test_path = sorted(glob.glob(os.path.join(dataset_path, 'test', 'CONTROL*.nii.gz')))
asds_test_path = sorted(glob.glob(os.path.join(dataset_path, 'test', 'ASD*.nii.gz')))

Xtrain = controls_train_path + asds_train_path
Xvalid = controls_valid_path + asds_valid_path
Xtest = controls_test_path + asds_test_path

nA_train = len(controls_train_path)
nB_train = len(asds_train_path)
nImages_train = nA_train + nB_train

nA_valid = len(controls_valid_path)
nB_valid = len(asds_valid_path)
nImages_valid = nA_valid + nB_valid

nA_test = len(controls_test_path)
nB_test = len(asds_test_path)
nImages_test = nA_test + nB_test

print('Training: %i, %i, total: %i' % (nA_train, nB_train, nImages_train))
print('Validation: %i, %i, total: %i' % (nA_valid, nB_valid, nImages_valid))
print('Test: %i, %i, total: %i' % (nA_test, nB_test, nImages_test))

## Create ground truth labels

Ytrain = np.concatenate( (np.zeros(nA_train), np.ones(nB_train)), axis=None )
Yvalid = np.concatenate( (np.zeros(nA_valid), np.ones(nB_valid)), axis=None )
Ytest = np.concatenate( (np.zeros(nA_test), np.ones(nB_test)), axis=None )

## TF features

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## Permute data

perm = np.random.permutation(np.arange(nImages_train))

Xtrain = [Xtrain[i] for i in perm]
Ytrain = Ytrain[perm]

## Determine number of TFRecord files

maxImagesPerFile = 32

folderOut = out_path
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

def compute_n_files(nImages):
    nImagesPerFile = []
    temp = nImages
    while temp >= 0:
        if temp > maxImagesPerFile:
            nImagesPerFile.append(maxImagesPerFile)
            temp -= maxImagesPerFile
        else:
            nImagesPerFile.append(temp)
            break

    return nImagesPerFile

nImagesPerFileTrain = compute_n_files(nImages_train)
nFilesTrain = len(nImagesPerFileTrain)

nImagesPerFileValid = compute_n_files(nImages_valid)
nFilesValid = len(nImagesPerFileValid)

nImagesPerFileTest = compute_n_files(nImages_test)
nFilesTest = len(nImagesPerFileTest)

print('nFiles: %i, %i, %i' % (nFilesTrain, nFilesValid, nFilesTest))

## Write TFRecord files

#sys.exit()

def write_tfrecord_files(X, Y, nImagesPerFile, fileOutRoot):

    imIndex = 0

    for fileIndex, nImagesFile in enumerate(nImagesPerFile):

        fileName = '%s_%04i.tfrecord' % (fileOutRoot, fileIndex)
        print(fileName)
        print('Done %i files' % imIndex)

        writer = tf.io.TFRecordWriter(fileName)

        for volumeFile, label in zip(X[imIndex:imIndex+nImagesFile], Y[imIndex:imIndex+nImagesFile]):

            # Load volumes
            image = load_single_volume(volumeFile)
            #image = image / 60 - 1
            image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image)) )
            image = image / 127.5 - 1
            image = image[:,:,:,np.newaxis]
            #image = np.clip(image, -1, 1)

            # Define features
            feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                'label': _int64_feature(int(label))}

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(tf_example.SerializeToString())

        imIndex += nImagesFile

fileOutRoot = os.path.join(folderOut, 'train')
write_tfrecord_files(Xtrain, Ytrain, nImagesPerFileTrain, fileOutRoot)

fileOutRoot = os.path.join(folderOut, 'valid')
write_tfrecord_files(Xvalid, Yvalid, nImagesPerFileValid, fileOutRoot)

fileOutRoot = os.path.join(folderOut, 'test')
write_tfrecord_files(Xtest, Ytest, nImagesPerFileTest, fileOutRoot)

