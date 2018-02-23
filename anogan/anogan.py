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
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

# def generator_model():
#     generator = Sequential()
#     generator.add(Dense(128*7*7, input_dim=100, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#     generator.add(LeakyReLU(0.2))
#     generator.add(Reshape((7, 7, 128)))
#     generator.add(UpSampling2D(size=(2, 2)))
#     generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
#     generator.add(LeakyReLU(0.2))
#     generator.add(UpSampling2D(size=(2, 2)))
#     generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
#     generator.compile(loss='binary_crossentropy', optimizer='adam')
#     return generator

# def discriminator_model():
#     discriminator = Sequential()
#     discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28,28, 1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#     discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(0.3))
#     discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
#     discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(0.3))
#     discriminator.add(Flatten())
#     discriminator.add(Dense(1, activation='sigmoid'))
#     discriminator.compile(loss='binary_crossentropy', optimizer='adam')
#     return discriminator

def generator_model():
    inputs = Input((100,))
    fc1 = Dense(input_dim=100, units=128*7*7)(inputs)
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
    model = Sequential()
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
    ganInput = Input(shape=(100,))
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
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    for epoch in range(20):
        print ("Epoch is", epoch)
        # print ("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 100))
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
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 100))
    generated_images = g.predict(noise)
    return generated_images

def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

def feature_extractor():
    d = discriminator_model()
    d.load_weights('assets/discriminator.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='sgd')
    return intermidiate_model

def anomaly_detector():
    g = generator_model()
    g.load_weights('assets/generator.h5')
    g.trainable = False
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
    
    aInput = Input(shape=(100,))
    gInput = Dense((100), trainable=True)(aInput)
    # gInput = Dense((100))(gInput)
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [1, 0], optimizer='sgd')
    # model.summary()
    # exit()
    return model

def debug_model(model):
    debug_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    debug_model.compile(loss='mse', optimizer='sgd')
    return debug_model

def compute_anomaly_score(model, x):
    num_z = 10
    z = np.random.uniform(0, 1, size=(num_z, 1, 100))
    model.summary()
    list_similar_data = []
    list_loss = []
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    for idx in range(1):
        similar_data_pre, _ = model.predict(z[idx])
        print (similar_data_pre.shape)
        print (np.sum(abs(x[0] - similar_data_pre[0])))
        debug = debug_model(model)
        hidden = debug.predict(z[idx])
        print ('hidden shape:', hidden[0].shape)
        print (z[idx][0,:10])
        print (hidden[0][:10])
        loss = model.fit(z[idx], [x, d_x], batch_size=1, epochs=10, verbose=0)
        # loss = model.train_on_batch(z[idx], [x, d_x])
        similar_data, _ = model.predict(z[idx])
        debug = debug_model(model)
        hidden = debug.predict(z[idx])
        print (hidden[0][:10])
        print (model.evaluate(z[idx], [x, d_x], 1, 1))
        print (similar_data.shape)
        print (np.sum(abs(x[0] - similar_data[0])))
        print (np.sum(abs(similar_data[0] - similar_data_pre[0])))
        list_similar_data.append(similar_data)
        # print ('loss', loss)
        list_loss.append(loss.history['loss'][-1])
        print (loss.history['loss'])
        loss = loss.history['loss'][-1]
    
    # print (list_loss)
    # exit()
    
    return loss, similar_data
