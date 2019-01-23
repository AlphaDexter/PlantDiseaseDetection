import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,adam

data_path = '/home/satyam/Desktop/Major/Plant_Disease_Detection/data'
data_dir_list = os.listdir(data_path)


def loadimage():

    img_rows = 256
    img_cols = 256
    num_channel = 3
    num_epoch = 20
    num_classes = 8
    img_data_list = []
    labels_list = []
    labels_name = {'c_0':0,'c_1':1,'c_2':2,'c_3':3, 'c_4':4,'c_5':5,'c_6':6,'c_7':7}

    for dataset in data_dir_list:
        img_list = os.listdir(data_path+'/'+ dataset)
        print ('Loading the images of dataset-'+'{}\n'.format(dataset))
        label = labels_name[dataset]
        #print(type(labels_name))
        for img in img_list:
            input_img = cv2.imread(data_path + '/'+ dataset + '/' + img)
            input_img_resize = cv2.resize(input_img, (256, 256))
            img_data_list.append(input_img_resize)
            labels_list.append(label)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    # print (img_data.shape)

    labels = np.array(labels_list)
    # print(np.unique(labels, return_counts=True))
    Y = np_utils.to_categorical(labels, num_classes)
    x, y = shuffle(img_data,Y, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


    USE_SKLEARN_PREPROCESSING = False

    if USE_SKLEARN_PREPROCESSING:

        from sklearn import preprocessing

        def image_to_feature_vector(image, size=(256, 256)):
            return cv2.resize(image, size).flatten()

        img_data_list = []
        for dataset in data_dir_list:
            img_list = os.listdir(data_path + '/' + dataset)
            print ('Loaded the images of dataset-' + '{}\n'.format(dataset))
            for img in img_list:
                input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
                input_img_flatten = image_to_feature_vector(input_img, (256, 256))
                img_data_list.append(input_img_flatten)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')
        print (img_data.shape)
        img_data_scaled = preprocessing.scale(img_data)
        print (img_data_scaled.shape)
        print (np.mean(img_data_scaled))
        print (np.std(img_data_scaled))
        print (img_data_scaled.mean(axis=0))
        print (img_data_scaled.std(axis=0))

        img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
        print (img_data_scaled.shape)
    if USE_SKLEARN_PREPROCESSING:
        img_data = img_data_scaled

    input_shape = img_data[0].shape

    return X_train, X_test, y_train, y_test, input_shape