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

### 0. prepare data
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

### 1. train generator & discriminator
# Model_d, Model_g = anogan.train(64, X_train)
# Model_d, Model_g = anogan.load_model()

### 2. test generator
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

### 3. other class anomaly detection

### compute anomaly score - sample from test set

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

### compute anomaly score - sample from strange image

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

### compute anomaly score - sample from strange image

# test_img = plt.imread('assets/test_img.png')
# test_img = test_img[:,:,0]
test_img = X_test_original[y_test==7][3]

model = anogan.anomaly_detector()
ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=1000)

print (test_img.dtype, similar_img.dtype)

# anomaly area, 255 normalization
np_residual = test_img.reshape(28,28) - similar_img.reshape(28,28)
np_residual = (np_residual - np_residual.min())/(np_residual.max()-np_residual.min())

np_residual = (255*np_residual).astype(np.uint8)
original_x = (test_img.reshape(28,28)*127.5+127.5).astype(np.uint8)
similar_x = (similar_img.reshape(28,28)*127.5+127.5).astype(np.uint8)

cv2.namedWindow('ori', 0)
cv2.namedWindow('pred', 0)
cv2.namedWindow('anomaly', 0)
cv2.resizeWindow('ori', 500, 500)
cv2.resizeWindow('pred', 500, 500)
cv2.resizeWindow('anomaly', 500, 500)

original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

original_x = cv2.resize(original_x, (256,256), interpolation=cv2.INTER_NEAREST)
similar_x = cv2.resize(similar_x, (256,256), interpolation=cv2.INTER_NEAREST)
show = cv2.resize(show, (256,256), interpolation=cv2.INTER_NEAREST)

cv2.imshow('ori', original_x)
cv2.imshow('pred', similar_x)
cv2.imshow('anomaly', show)
# cv2.imwrite('anomaly_result/qurey.png', original_x)
# cv2.imwrite('anomaly_result/similar.png', similar_x)
# cv2.imwrite('anomaly_result/anomaly_result.png', show)
key = cv2.waitKey()
if key == 27:
    exit()

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


### 4. tsne feature view

# from sklearn.manifold import TSNE

# ### t-SNE embedding 

# # generating anomaly image for test (radom noise image)

# random_image = np.random.uniform(0,1, (100, 28,28, 1))
# print("a sample from generated anomaly images(random noise image)")
# plt.figure(figsize=(2, 2))
# plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
# plt.show()

# # intermidieate output of discriminator
# model = anogan.feature_extractor()
# feature_map_of_random = model.predict(random_image, verbose=1)
# feature_map_of_minist = model.predict(X_test_original[y_test != 1][:300], verbose=1)
# feature_map_of_minist_1 = model.predict(X_test[:100], verbose=1)

# # t-SNE for visulization
# output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
# output = output.reshape(output.shape[0], -1)
# anomaly_flag = np.array([1]*100+ [0]*300)

# X_embedded = TSNE(n_components=2).fit_transform(output)
# plt.title("t-SNE embedding on the feature representation")
# plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
# plt.scatter(X_embedded[100:400,0], X_embedded[100:400,1], label='mnist(anomaly)')
# plt.scatter(X_embedded[400:,0], X_embedded[400:,1], label='mnist(normal)')
# plt.legend()
# plt.show()