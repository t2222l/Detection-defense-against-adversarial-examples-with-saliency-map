import numpy as np
import cv2
import os
from art.attacks.evasion import DeepFool
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from art.classifiers import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
import cv2

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def preprocess_image(img):
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    #
    #preprocessed_img = img.copy()[:, :, ::-1]
    # for i in range(3):
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.transpose(img, (2, 0, 1))
    return preprocessed_img

x_val = np.load('Datasets/Imagenet_x.npy', mmap_mode ='r')

min_, max_ = 0, 1

model = models.densenet121(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), .1, momentum=0.9, weight_decay=1e-4)
classifier = PyTorchClassifier(model=model, clip_values=(min_, max_),loss=criterion, optimizer=optimizer,
                               input_shape=(3, 224, 224), nb_classes=1000, channel_index=1)
attack = FastGradientMethod(classifier, eps=0.1, batch_size=16)
# attack = DeepFool(classifier=classifier, max_iter=10, epsilon=1e-6, nb_grads=1000, batch_size=1)
#attack  =  ElasticNet(classifier, confidence=0.0, targeted=False, learning_rate=1e-2, binary_search_steps=9,
#                 max_iter=100, beta=1e-3, initial_const=1e-3, batch_size=1, decision_rule='EN')
#attack = CarliniL2Method(classifier)
#imgAdv = attack.generate(x=preprocess_input(x_val))  #对抗样本的集合

t = np.float32(x_val) / 255
img = np.ones((len(t), 3, 224, 224), dtype=np.float32)
for i in range(len(t)):
    img[i] = preprocess_image(t[i])

print(img.shape)
y_val = np.argmax(classifier.predict(img), axis=1)
imgAdv = attack.generate(img)
imgAdvPredictions = classifier.predict(imgAdv)

t = np.uint8(imgAdv*255)
adi = np.transpose(t, [0, 2, 3, 1])

np.save('Datasets/Imagenet_dense_y.npy', y_val)
np.save('Datasets/FGSM_0.1_dense_x.npy', adi)
np.save('Datasets/FGSM_0.1_dense_y.npy', np.argmax(imgAdvPredictions, axis=1))


preds = np.argmax(classifier.predict(img), axis=1)
acc = np.sum(preds == y_val) / y_val.shape[0]
print("\nTest accuracy was set to : %.2f%%" % (acc * 100))

preds = np.argmax(classifier.predict(imgAdv), axis=1)
acc = np.sum(preds == y_val) / imgAdvPredictions.shape[0]
print("\nTest accuracy on adversarial sample : %.2f%%" % (acc * 100))