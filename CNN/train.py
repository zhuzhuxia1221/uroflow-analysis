import numpy as np
from tensorflow import keras
import tensorflow as tf
from model import build_resnet 
from tool import readts  

import os

np.random.seed(813306)  

nb_epochs = 100 

f_list = ['SMU'] 
for each in f_list:
    f_name = each
    x_train, y_train = readts(f_name + '/' + f_name + '_TRAIN')
    x_test, y_test = readts(f_name + '/' + f_name + '_TEST')

    nb_classes = len(np.unique(y_test))  
    batch_size = min(x_train.shape[0] / 10, 16)  

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    x_test = (x_test - x_train_mean) / (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,)) 
    x_test = x_test.reshape(x_test.shape + (1,))

    x, y = build_resnet(x_train.shape[1:], 64, nb_classes)
    model = keras.models.Model(inputs=x, outputs=y)
    optimizer = keras.optimizers.Adam()  
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                  patience=50, min_lr=0.0001)
    checkpoint_path = "./results/cp.ckpt"  
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)

    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                     verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr, cp_callback])

    os.listdir(checkpoint_dir)
