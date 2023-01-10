"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.attacks import DeepFool
from art.attacks import FastGradientMethod
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset

from keras import backend as K
from keras.models import load_model


# # Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('[%(levelname)s] %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# Step 1:  Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape

# Step 2: Create the model

model = load_model('net_in_net.h5')
#model = load_model ('../models/cifar10/net_in_net.h5')
model.summary()

# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(min_, max_), use_logits=False)

# Step 4: Train the ART classifier

#classifier.fit(x_train, y_train, batch_size=64, nb_epochs=200)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.2)
# attack = CarliniL2Method(classifier=classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
                 # max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))

K.clear_session()