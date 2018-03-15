from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math

from keras.utils. generic_utils import Progbar

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def discriminator_model():
    inputs = Input((28, 28, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(10,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('./assets/discriminator.h5')
    g.load_weights('./assets/generator.h5')
    return g, d

def train(BATCH_SIZE, X_train):
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.0004)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    
    for epoch in range(10):
        print ("Epoch is", epoch)
        # print ("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            d_loss = d.train_on_batch(X, y)
            # noise = np.random.uniform(0, 1, (BATCH_SIZE, 100))
            # if index % 2 == 0:
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
        print ('')

        g.save_weights('assets/generator.h5', True)
        d.save_weights('assets/discriminator.h5', True)
    return d, g


def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('assets/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
    generated_images = g.predict(noise)
    return generated_images

def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

def feature_extractor(d=None):
    if d is None:
        # print ('discriminator model loading')
        d = discriminator_model()
        d.load_weights('assets/discriminator.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

def anomaly_detector(g=None, d=None):
    # K.set_learning_phase(0)
    if g is None:
        # print ('generator model loading')
        g = generator_model()
        g.load_weights('assets/generator.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    # gInput = Dense((100))(gInput)
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    # print ('learning phase:', K.learning_phase())
    K.set_learning_phase(0)
    # model.summary()
    # exit()
    return model

def debug(model):
    debug_model = Model(inputs=model.layers[1].input, outputs=model.layers[2].output)
    debug_model.compile(loss='mse', optimizer='sgd')
    return debug_model

def compute_anomaly_score(model, x, iterations=1500, d=None):
    num_z = 10
    z = np.random.uniform(0, 1, size=(num_z, 1, 10))
    debug_model = debug(model)
    # model.summary()
    list_similar_data = []
    list_loss = []
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)
    for idx in range(1):
        # similar_data_pre, _ = model.predict(z[idx])
        # print (similar_data_pre.shape)
        # print ('handcraft pre:', np.sum(abs(x[0] - similar_data_pre[0])))
        # print ('ori z:', z[idx])
        # print ('pred pre z:', debug_model.predict(z[idx]))
        loss = model.fit(z[idx], [x, d_x], batch_size=1, epochs=iterations, verbose=0)
        # loss = model.train_on_batch(z[idx], [x, d_x])
        similar_data, _ = model.predict(z[idx])
        last_z = debug_model.predict(z[idx])
        # print ('pred after z:', last_z)
        
        # print ('eval post:', model.evaluate(z[idx], [x, d_x], 1, 1))
        # print ('handcraft post:', np.sum(abs(x[0] - similar_data[0])))
        # print ('diff:', np.sum(abs(similar_data[0] - similar_data_pre[0])))
        # list_similar_data.append(similar_data)
        # print ('loss', loss)
        list_loss.append(loss.history['loss'][-1])
        # print (loss.history['loss'])
        loss = loss.history['loss'][-1]
    
    # print (list_loss)
    # exit()
    
    return loss, similar_data, last_z
