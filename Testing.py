from keras.models import load_model
import cv2
import numpy as np


def testImage(path):
    plant_list=['Apple Scab','Apple Black Rot','Cedar Apple Rust','Apple Healthy','Blueberry healthy','Cherry Powdery Mildew','Cherry Healthy','Corn Gray Leaf Spot']

    # path1 = 'C:/Users/Ambuj Mittal/PycharmProjects/Plant_Disease_Detection/media/0d3c0790-7833-470b-ac6e-94d0a3bf3e7c___FREC_Scab%202959.JPG'
    # print(path1)

    model = load_model('Plant_Disease_model.h5')

    test_image = cv2.imread(path)
    test_image = cv2.resize(test_image, (256, 256))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    # print (test_image.shape)
    test_image = np.expand_dims(test_image, axis=0)
    # print(test_image.shape)
    # print(model.predict(test_image))
    img_class = int(model.predict_classes(test_image))
    print('Class of Image is:', img_class)

    print('\n\n\n\n')
    print('Disease of the image identified -', plant_list[img_class])
    print('\n\n\n\n')
    return img_class, plant_list[img_class]
# model.summary()