from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import random
from Database import data
from keras.models import load_model
import cv2
import numpy as np
from Testing import testImage
import Segmentation

import os


def index(request):
    return HttpResponse("<h1>This is imageupload homepage</h1>")


def home(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']

        fs = FileSystemStorage()
        name = fs.save('active' + str(random.randint(0, 10000000)) + '.JPG', uploaded_file)
        url = fs.url(name)

        # print(url)
        path = str(url)
        # print(path)
        path = path[1:]

        # print(path)
        os.system("python Segmentation.py " + path)

        segmented_image_path = 'C:/Users/Ambuj Mittal/PycharmProjects/Plant_Disease_Detection/' + path.split('.')[0] + '_marked.' + path.split('.')[1]

        # print(segmented_image_path)

        img_class, img_name = testImage('C:/Users/Ambuj Mittal/PycharmProjects/Plant_Disease_Detection/'+path)

        displaydata = {'determiner': 1, 'content': data[img_class] }

    else:
        displaydata = {'determiner': 0}
        # displaydata = {'determiner': 1, 'content': data[5]}

    return render(request, 'HomePage.html', displaydata)
