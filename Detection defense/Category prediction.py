from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
#from keras.applications.vgg19 import (VGG19, preprocess_input, decode_predictions)
#from keras.applications.resnet50 import (ResNet50, preprocess_input, decode_predictions)
#from keras.applications.mobilenet import (MobileNet, preprocess_input, decode_predictions)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
import os
from keras.models import load_model

from art.attacks import DeepFool
from art.attacks import FastGradientMethod
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.elastic_net import ElasticNet
from art.classifiers import KerasClassifier
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod


from scipy import ndimage
import imageio

x_val = np.load('VGG16_val_338.npy',mmap_mode = 'c')
y_val = np.load('VGG16_y_val_215.npy',mmap_mode = 'r')
min_, max_ = 0,255


# In[5]:


model = VGG16(weights='imagenet')
#model = MobileNet(weights='imagenet')
#model.summary()

from PIL import Image
img111 = Image.fromarray(np.squeeze(x_val[0]).astype(np.uint8), 'RGB')
img111.save('oooo1ooook.png')

# img4predict = np.expand_dims(img, axis=0)
# img4predict = img4predict.astype('float32')
# predictions = model.predict(img4predict)
# predicted_class = np.argmax(predictions)
# print (predicted_class)
# print (decode_predictions(predictions))

# img1 = np.expand_dims(x_val[2][:,:,::-1], axis=0)
# img1 = img1.astype('float32')

# predictions = model.predict(img)
# predicted_class = np.argmax(predictions)

# predictions1 = model.predict(img1)
# predicted_class1 = np.argmax(predictions1)



# print (y_val[2])
# print (predicted_class)
# print (predicted_class1)
# print ('**************')




# 中值滤波测试
# from scipy import ndimage
# import imageio

# # image = imageio.imread('./backup/test/lu.png')
# # images = np.expand_dims(image, axis=0)
# # image_median = ndimage.filters.median_filter(images, size=(1,3,3,1), mode='reflect')

# # img = Image.fromarray(np.squeeze(image_median).astype(np.uint8), 'RGB')
# # img.save('luuuuu.png')
# # img.show()

# image = imageio.imread('./backup/test/panda_blur_3_3.png')
# image1 = imageio.imread('./backup/test/luuuuu.png')
# print (image)
# print (image1)



# non-local mean测试

# image = imageio.imread('./backup/test/lu.png')
# image_non_local = cv2.fastNlMeansDenoisingColored(image,None,3,3,3,11)

# img = Image.fromarray(np.squeeze(image_non_local).astype(np.uint8), 'RGB')
# img.save('luuuuu_nonLocal.png')
# img.show()

# def non_local_means_color_py(imgs, search_window, block_size, photo_render):
#     import cv2
#     ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoisingColored, [None,photo_render,photo_render,block_size,search_window])
#     return ret_imgs

# fastNlMeansDenoisingColored( InputArray src, OutputArray dst, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)

# image = imageio.imread('./backup/test/lu.png')
# image1 = imageio.imread('./backup/test/luuuuu_nonLocal.png')
# print (image)
# print (image1)

K.clear_session()