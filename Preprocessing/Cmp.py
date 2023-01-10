
# coding: utf-8

# In[1]:


from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from art.defences.feature_squeezing import FeatureSqueezing
# from keras.applications.vgg19 import (VGG19, preprocess_input, decode_predictions)
# from keras.applications.resnet50 import (ResNet50, preprocess_input, decode_predictions)
from keras.preprocessing import image
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

# In[2]:


def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # avg
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # create heat map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


# In[3]:


def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model


# In[4]:


x_val = np.load('VGG16_val_215.npy',mmap_mode = 'c')
y_val = np.load('VGG16_y_val_215.npy',mmap_mode = 'r')
min_, max_ = 0,255


# In[5]:


model = VGG16(weights='imagenet')
model.summary()


# In[6]:

classifier = KerasClassifier(model=model, clip_values=(min_, max_), use_logits=False)

# for depth in range(1, 9):
#     successIndex = []
#     preproc = FeatureSqueezing(clip_values=(0, 255), bit_depth=depth)
#     x_squeezed, _ = preproc(x_val)
#     predictions = classifier.predict(x_squeezed)

#     cmp = (np.argmax(predictions, axis=1))
#     for i in range(len(predictions)):
#         if cmp[i] == y_val[i]:
#             successIndex.append(i)
#     print ('Current depth is:%d  Detection accuracy is:%.4f' %(depth,len(successIndex)/len(x_squeezed)))


# image_median = ndimage.filters.median_filter(x_val, size=(1,2,2,1), mode='reflect')
# predictions = classifier.predict(image_median)

# successIndex = []
# cmp = (np.argmax(predictions, axis=1))
# for i in range(len(x_val)):
#     if cmp[i] == y_val[i]:
#         successIndex.append(i)
# print ('Detection accuracy is:%.4f' %(len(successIndex)/len(image_median)))  



# success_nonLocal = []
# for i in range(len(x_val)):
#     image_non_local = cv2.fastNlMeansDenoisingColored((x_val[i].astype('uint8')),None,2,2,3,13)
#     image_nonLocal4predict = np.expand_dims((image_non_local.astype('float32')), axis=0)
#     prediction = classifier.predict(image_nonLocal4predict)

#     if np.argmax(prediction) == y_val[i]:
#         success_nonLocal.append(i)
# print (len(success_nonLocal))
# print ('Detection accuracy is:%.4f' %(len(success_nonLocal)/len(x_val)))



success_all = []
for i in range(len(x_val)):
    image_non_local = cv2.fastNlMeansDenoisingColored((x_val[i].astype('uint8')),None,4,4,3,11)
    image_nonLocal4predict = np.expand_dims((image_non_local.astype('float32')), axis=0)
    image_median = ndimage.filters.median_filter(image_nonLocal4predict, size=(1,2,2,1), mode='reflect')
    preproc = FeatureSqueezing(clip_values=(0, 255), bit_depth=5)
    x_squeezed, _ = preproc(image_median)
    prediction = classifier.predict(x_squeezed)

    if np.argmax(prediction) == y_val[i]:
        success_all.append(i)
print (len(success_all))
print ('Detection accuracy is:%.4f' %(len(success_all)/len(x_val)))





# attack = FastGradientMethod(classifier=classifier, eps=0.8, batch_size=16)
# imgAdv = attack.generate(x=x_val)
# imgAdvPredictions = classifier.predict(imgAdv)

# cmp = (np.argmax(imgAdvPredictions, axis=1))
# attackSuccessIndex = []
# for i in range(len(imgAdvPredictions)):
#     if cmp[i] != y_val[i]:
#         attackSuccessIndex.append(i)       
# print ('Attack success rate is:%0.4f'%(len(attackSuccessIndex)/len(imgAdv)))

# imgGen = np.zeros((len(attackSuccessIndex), 224, 224, 3), dtype=np.float32)
# imgGen_Label = np.zeros((len(attackSuccessIndex),), dtype=np.int32)
# for i in range(len(attackSuccessIndex)):
#     imgGen[i] = imgAdv[attackSuccessIndex[i]]
#     imgGen_Label[i] = y_val[attackSuccessIndex[i]]
# print(len(imgGen))
# print(len(imgGen_Label))


# for depth in range(1, 9):
#     successIndex = []
#     preproc = FeatureSqueezing(clip_values=(0, 255), bit_depth=depth)
#     x_squeezed, _ = preproc(imgGen)
#     predictions = classifier.predict(x_squeezed)

#     cmp = (np.argmax(predictions, axis=1))
#     for i in range(len(predictions)):
#         if cmp[i] == imgGen_Label[i]:
#             successIndex.append(i)
#     print ('Current depth is:%d  Detection accuracy is:%.4f' %(depth,len(successIndex)/len(x_squeezed)))
    #print ('Detection accuracy is:%.4f' %(len(successIndex)/len(x_squeezed)))




# image_median = ndimage.filters.median_filter(imgGen, size=(1,3,3,1), mode='reflect')
# predictions = classifier.predict(image_median)

# successIndex = []
# cmp = (np.argmax(predictions, axis=1))
# for i in range(len(predictions)):
#     if cmp[i] == imgGen_Label[i]:
#         successIndex.append(i)
# print ('Detection accuracy is:%.4f' %(len(successIndex)/len(image_median)))    



# non-local-mean
# success_nonLocal = []
# for i in range(len(imgGen)):
#     image_non_local = cv2.fastNlMeansDenoisingColored((imgGen[i].astype('uint8')),None,2,2,3,13)
#     image_nonLocal4predict = np.expand_dims((image_non_local.astype('float32')), axis=0)
#     prediction = classifier.predict(image_nonLocal4predict)

#     if np.argmax(prediction) == imgGen_Label[i]:
#         success_nonLocal.append(i)
# print (len(success_nonLocal))
# print ('Detection accuracy is:%.4f' %(len(success_nonLocal)/len(imgGen)))  

# def non_local_means_color_py(imgs, search_window, block_size, photo_render=h):
#     import cv2
#     ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoisingColored, [None,photo_render,photo_render,block_size,search_window])
#     return ret_imgs

# fastNlMeansDenoisingColored( InputArray src, OutputArray dst, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)



# success_all = []
# for i in range(len(imgGen)):
#     image_non_local = cv2.fastNlMeansDenoisingColored((imgGen[i].astype('uint8')),None,4,4,3,11)
#     image_nonLocal4predict = np.expand_dims((image_non_local.astype('float32')), axis=0)
#     image_median = ndimage.filters.median_filter(image_nonLocal4predict, size=(1,2,2,1), mode='reflect')
#     preproc = FeatureSqueezing(clip_values=(0, 255), bit_depth=5)
#     x_squeezed, _ = preproc(image_median)
#     prediction = classifier.predict(x_squeezed)

#     if np.argmax(prediction) == imgGen_Label[i]:
#         success_all.append(i)
# print (len(success_all))
# print ('Detection accuracy is:%.4f' %(len(success_all)/len(imgGen)))


# attack = FastGradientMethod(classifier=classifier, eps=0.8, batch_size=16)
# #attack = DeepFool(classifier, max_iter=10, epsilon=0.02, nb_grads=1000, batch_size=4)
# #attack  =  ElasticNet(classifier, confidence=0.0, targeted=False, learning_rate=1e-2, binary_search_steps=9,
# #                 max_iter=100, beta=1e-3, initial_const=1e-3, batch_size=1, decision_rule='EN')
# imgAdv = attack.generate(x=x_val)  
# imgAdvPredictions = classifier.predict(imgAdv)

# print ('Done')
# # In[7]:


# cmp = (np.argmax(imgAdvPredictions, axis=1))
# attackSuccessIndex = []
# for i in range(len(imgAdvPredictions)):
#     if cmp[i] != y_val[i]:
#         attackSuccessIndex.append(i)
        
# print (attackSuccessIndex)
# print (len(attackSuccessIndex))


# # In[13]:




# #for i in range(len(attackSuccessIndex)):
# need_test = [0.000000000001,0.1,0.5]
# for j in range(len(need_test)):
#     successNum = 0
#     successDetectionNum = 0

#     for i in range(100):
#         #img = x_val[attackSuccessIndex[i]]
#         img = x_val[i]

#         img1 = np.expand_dims(img, axis=0)
#         img1 = img1.astype('float32')

#         img4predict = np.copy(img1)

#         predictions = model.predict(img4predict)
#         predicted_class = np.argmax(predictions)
        

#         cam_image, heat_map = grad_cam(model, img4predict, predicted_class, "block5_conv3")
#         combineImg = cam_image * need_test[j] + img

#         # cv2.imwrite("123.JPEG", combineImg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#         # combineImg = cv2.imread("123.JPEG")
        
#         #*******
#         img2 = np.expand_dims(combineImg, axis=0)
#         # img2 = np.expand_dims(combineImg[:,:,::-1], axis=0)
#         img2 = img2.astype('float32')
        
#         #*******
#         # img2 = preprocess_input(img2)

#         #*******
#         combine4predict = np.copy(img2)
#         # combine4predict = np.copy(img2[:,:,:,::-1])


#         predictions_Combine = model.predict(combine4predict)
#         predicted_class_Combine = np.argmax(predictions_Combine)


#         if predicted_class == predicted_class_Combine:
#             successNum += 1
        
#     #print ('Detection Accuracy is: %.2f'  %(100 * successNum/len(attackSuccessIndex)))
#     print ('Detection Accuracy is: %.4f'  %(successNum/100))

#     # for i in range(len(attackSuccessIndex)):
#     for i in range(100):
#         img = imgAdv[attackSuccessIndex[i]]

#         img1 = np.expand_dims(img, axis=0)
#         img1 = img1.astype('float32')

#         img4predict = np.copy(img1)


#         predictions = model.predict(img4predict)
#         predicted_class = np.argmax(predictions)
        

#         cam_image, heat_map = grad_cam(model, img4predict, predicted_class, "block5_conv3")
#         combineImg = cam_image * need_test[j] + img
        
#         # cv2.imwrite("123.JPEG", combineImg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#         # combineImg = cv2.imread("123.JPEG")

#         #*******
#         img2 = np.expand_dims(combineImg, axis=0)
#         # img2 = np.expand_dims(combineImg[:,:,::-1], axis=0)
#         img2 = img2.astype('float32')
        
#         #*******
#         # img2 = preprocess_input(img2)

#         #*******
#         combine4predict = np.copy(img2)
#         # combine4predict = np.copy(img2[:,:,:,::-1])

#         predictions_Combine = model.predict(combine4predict)
#         predicted_class_Combine = np.argmax(predictions_Combine)

#         if predicted_class != predicted_class_Combine:
#             successDetectionNum += 1
        
#     #print ('Adversarial Samples Detection Accuracy is: %.2f'  %(100 * successDetectionNum/len(attackSuccessIndex)))
#     print ('Adversarial Samples Detection Accuracy is: %.4f'  %(successDetectionNum/100))

# print (j)
# print (i)
# print (successNum)
# print (successDetectionNum)

# In[ ]:

K.clear_session()

