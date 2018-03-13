from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import anogan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int)
parser.add_argument('--label_idx', type=int)
args = parser.parse_args()

### 0. prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# X_train = X_train.reshape(60000, 28, 28, 1)
# X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]
X_test_original = X_test.copy()

# X_train = X_train[(y_train==1)] # | (y_train==0)]
X_test = X_test[(y_test==1)] # | (y_test==0)]

# print (X_train.shape)
# print (X_test.shape)

# X_train = np.append(X_train, X_test, axis=0)
# print (X_train.shape)

### 1. train generator & discriminator
# Model_d, Model_g = anogan.train(64, X_train)
# Model_d, Model_g = anogan.load_model()

### 2. test generator
# generated_img = anogan.generate(64)
# img = anogan.combine_images(generated_img)
# img = img[:,:,0]
# img = (img+1)/2
# # cv2.namedWindow('generated', 0)
# # cv2.resizeWindow('generated', 256, 256)

# plt.figure(num=4, figsize=(4, 4))
# plt.title('trained generator')
# plt.imshow(img, cmap=plt.cm.gray)
# # plt.show()

# exit()

### 3. other class anomaly detection

# g = anogan.generator_model()
# g.load_weights('assets/generator.h5')
# d = anogan.discriminator_model()
# d.load_weights('assets/discriminator.h5')

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=300, d=d)

    # print (test_img.dtype, similar_img.dtype)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(28,28,1) - similar_img.reshape(28,28,1)
    np_residual = (np_residual - np_residual.min())/(np_residual.max()-np_residual.min())

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)

    # cv2.namedWindow('ori', 0)
    # cv2.namedWindow('pred', 0)
    # cv2.namedWindow('anomaly', 0)
    # cv2.resizeWindow('ori', 500, 500)
    # cv2.resizeWindow('pred', 500, 500)
    # cv2.resizeWindow('anomaly', 500, 500)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    # original_x = cv2.resize(original_x, (256,256), interpolation=cv2.INTER_NEAREST)
    # similar_x = cv2.resize(similar_x, (256,256), interpolation=cv2.INTER_NEAREST)
    # show = cv2.resize(show, (256,256), interpolation=cv2.INTER_NEAREST)

    # cv2.imshow('ori', original_x)
    # cv2.imshow('pred', similar_x)
    # cv2.imshow('anomaly', show)
    # cv2.imwrite('anomaly_result/qurey.png', original_x)
    # cv2.imwrite('anomaly_result/similar.png', similar_x)
    # cv2.imwrite('anomaly_result/anomaly_result.png', show)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     exit()
    return ano_score, original_x, similar_x, show

    ### matplot view
    # plt.figure(1, figsize=(2, 2))
    # plt.title('query image')
    # plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
    # # plt.show()
    # print("anomaly score : " + str(ano_score))
    # plt.figure(2, figsize=(2, 2))
    # plt.title('generated image')
    # plt.imshow(similar_img.reshape(28,28), cmap=plt.cm.gray)
    # residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
    # plt.figure(3, figsize=(2, 2))
    # plt.title('anomaly detection')
    # plt.imshow(residual, cmap='jet', alpha=.5)
    # plt.show()


### compute anomaly score - sample from test set

test_img = X_test_original[y_test==1][30]

### compute anomaly score - sample from strange image

test_img = X_test_original[y_test==0][30]

### compute anomaly score - sample from strange image
img_idx = args.img_idx
label_idx = args.label_idx
test_img = X_test_original[y_test==label_idx][img_idx]

start = cv2.getTickCount()
score,qurey,pred,diff = anomaly_detection(test_img)
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
cv2.imwrite('./anomaly_result/qurey_%d_%03d.png'%(label_idx, img_idx), qurey)
cv2.imwrite('./anomaly_result/pred_%d_%03d.png'%(label_idx, img_idx), pred)
cv2.imwrite('./anomaly_result/diff_%d_%03d.png'%(label_idx, img_idx), diff)

### multi image

# num_imgs = 100
# for label_idx in range(10):
#     label = label_idx
#     list_test_img = X_test_original[y_test==label][0:num_imgs]
#     list_score = []
#     list_qurey = []
#     list_pred = []
#     list_diff = []

#     for idx, test_img in enumerate(list_test_img):
#         start = cv2.getTickCount()
#         score,qurey,pred,diff = anomaly_detection(test_img, g, d)
#         list_qurey.append(qurey)
#         list_pred.append(pred)
#         list_diff.append(diff)
#         list_score.append(score)
#         time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
#         print ('%d label, %d : done'%(label, idx), '%.2fms'%time)

#     np_qurey = np.array(list_qurey)
#     np_pred = np.array(list_pred)
#     np_diff = np.array(list_diff)

#     combined_qurey = anogan.combine_images(np_qurey)
#     combined_pred = anogan.combine_images(np_pred)
#     combined_diff = anogan.combine_images(np_diff)

#     print ('qurey shape:', combined_qurey.shape)
#     print ('pred  shape:', combined_pred.shape)
#     print ('diff  shape:', combined_diff.shape)

#     cv2.imwrite('./anomaly_result/qurey_%d.png'%label, combined_qurey)
#     cv2.imwrite('./anomaly_result/pred_%d.png'%label, combined_pred)
#     cv2.imwrite('./anomaly_result/diff_%d.png'%label, combined_diff)

### 4. tsne feature view

### t-SNE embedding 
### generating anomaly image for test (radom noise image)

# from sklearn.manifold import TSNE
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