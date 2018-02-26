from __future__ import print_function
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

import anogan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]
X_test_original = X_test.copy()

X_train = X_train[(y_train==1)] # | (y_train==0)]
X_test = X_test[(y_test==1)] # | (y_test==0)]

print (X_train.shape)
print (X_test.shape)

X_train = np.append(X_train, X_test, axis=0)
print (X_train.shape)

# Model_d, Model_g = anogan.train(64, X_train)
# Model_d, Model_g = anogan.load_model()

generated_img = anogan.generate(64)
img = anogan.combine_images(generated_img)
img = (img+1)/2
# cv2.namedWindow('generated', 0)
# cv2.resizeWindow('generated', 256, 256)

plt.figure(num=4, figsize=(4, 4))
plt.title('trained generator')
plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# exit()

# ## compute anomaly score - sample from test set
# X_test = X_test.astype(np.float32) - 127.5 / 127.
# X_test = X_test.reshape(-1, 28, 28, 1)
# test_img = X_test_original[y_test==1][30]

# model = anogan.anomaly_detector()
# ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))

# test_img = (test_img+1)/2
# similar_img = (similar_img+1)/2

# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# plt.show()
# print("anomaly score : " + str(ano_score))
# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
# plt.imshow(residual, cmap='jet', alpha=.5)
# plt.show()

# ## compute anomaly score - sample from strange image

# # test_img = plt.imread('assets/test_img.png')
# # test_img = test_img[:,:,0]
# test_img = X_test_original[y_test==0][30]

# model = anogan.anomaly_detector()
# ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))

# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# plt.show()
# print("anomaly score : " + str(ano_score))
# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
# plt.imshow(residual, cmap='jet', alpha=.5)
# plt.show()

## compute anomaly score - sample from strange image

# test_img = plt.imread('assets/test_img.png')
# test_img = test_img[:,:,0]
test_img = X_test_original[y_test==7][3]

model = anogan.anomaly_detector()
ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))

plt.figure(1, figsize=(2, 2))
plt.title('query image')
plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# plt.show()
print("anomaly score : " + str(ano_score))
plt.figure(2, figsize=(2, 2))
plt.title('generated image')
plt.imshow(similar_img.reshape(28,28), cmap=plt.cm.gray)
residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
plt.figure(3, figsize=(2, 2))
plt.title('anomaly detection')
plt.imshow(residual, cmap='jet', alpha=.5)
plt.show()

# from sklearn.manifold import TSNE

# ## t-SNE embedding 

# # generating anomaly image for test (radom noise image)

# random_image = np.random.uniform(0,1, (100, 28,28, 1))
# print("a sample from generated anomaly images(random noise image)")
# plt.figure(figsize=(2, 2))
# plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
# plt.show()

# # intermidieate output of discriminator
# model = anogan.feature_extractor()
# feature_map_of_random = model.predict(random_image, verbose=1)
# feature_map_of_minist = model.predict(X_test[:300], verbose=1)

# # t-SNE for visulization
# output = np.concatenate((feature_map_of_random, feature_map_of_minist))
# output = output.reshape(output.shape[0], -1)
# anomaly_flag = np.array([1]*100+ [0]*300)

# X_embedded = TSNE(n_components=2).fit_transform(output)
# plt.title("t-SNE embedding on the feature representation")
# plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
# plt.scatter(X_embedded[100:,0], X_embedded[100:,1], label='minist(normal)')
# plt.legend()
# plt.show()