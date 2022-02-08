"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

import tensorflow as tf 

# Modèlisation

from tensorflow.keras import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense

from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D


from tensorflow.keras.layers import  Lambda, Reshape
from tensorflow import squeeze

from tensorflow.keras.optimizers import Adam

# Reproductibilité
from numpy.random import seed
seed(23)
from tensorflow import random
random.set_seed(23)



def build_model_cnn(nb_class, target_size):
    """
    La fonction build_model_cnn construit un modèle cnn de classification d'images pour obtenir leur transcription
    Paramètres :
        nb_class: int nombre de classes
        target_size : tuple correspondant aux dimensions de l'image souhaitées
    Renvoie:
        probabilités pour chaque classe
    """
    # Inputs
    inputs_data = Input(shape =(target_size[1],target_size[0],1), name = 'input_im')
    
    
    # CNN
    x = Conv2D(filters=64, kernel_size=(9,9), strides=(2,2), name = 'conv_1')(inputs_data)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same", name = 'conv_2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", name = 'conv_3')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", name = 'conv_4')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", name = 'conv_5')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = MaxPooling2D(pool_size=(2,2), name = 'max_pool')(x)
    
    x = GlobalAveragePooling2D(name = 'average_pool')(x)
    
    x = Dense(512, name = 'dense_1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Dropout(0.2, name = 'dropout_1')(x)
 
    y_pred = Dense(nb_class, activation='softmax', name = 'classifier')(x)
 
    
    # Définission du modèle
    model = Model(inputs=[inputs_data], outputs=y_pred, name="cnn")
    
    # compiler le model
    model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    return model