"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

import tensorflow  
import string

# Modèlisation
from tensorflow.keras import Model

from tensorflow.keras.layers import  Input, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, LSTM, Dense

from tensorflow.keras.layers import  MaxPooling2D, Dropout, Bidirectional

from tensorflow.keras.layers import  Lambda, Reshape
from tensorflow import squeeze

# Loss
from tensorflow.keras.backend import ctc_batch_cost

# Décodage
import tensorflow.keras.backend as K
#from  generator_rnn import transcript_vect_mot

# Construction de la couche CTC
class CTCLayer(tensorflow.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred, y_lengths):
        # Calcul de la loss value et ajouter à la couche avec fonction 'self.add_loss()'
        batch_len = tensorflow.cast(tensorflow.shape(y_true)[0], dtype="int64")
        input_length = tensorflow.cast(tensorflow.shape(y_pred)[1], dtype="int64")       

        input_length = input_length * tensorflow.ones(shape=(batch_len, 1), dtype="int64")
        label_length = y_lengths * tensorflow.ones(shape = [1], dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Retourner seulement les prédiction calculées au final
        return y_pred

    
# Contruction du modèle RNN
def build_model_rnn(target_size, batch_size):
    """
    La fonction build_model_rnn construit un modèle rnn pour obtenir la transcription des écritures manuscrites sur une image.
    Paramètres :
        target_size : tuple correspondant aux dimensions de l'image souhaitées
    Renvoie:
        probabilités pour chaque classe
    """
    # Inputs
    inputs_data = Input(shape = (target_size[1],target_size[0], 1), batch_size=batch_size, 
                        name = 'input_im', dtype = 'float32')
    labels = Input(shape = (None,), batch_size=batch_size, name = 'labels', dtype = 'float32')
    y_lengths = Input(name = 'label_length',batch_size=batch_size, shape = (None,), dtype = 'int64')
    
    # CNN
    x = Conv2D(filters=64, kernel_size=(9,9),strides=(2,2), padding="same", name = 'conv_1')(inputs_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2,2), name = 'max_pool1')(x)
    #x = Dropout(0.2)(x)
    
    x = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="valid", name = 'conv_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name = 'pool2')(x)
    
    
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid", name = 'conv_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output_cnn = MaxPooling2D(pool_size=(2,2), name = 'max_pool2')(x)
    
    # reshape to enter RNN
    x = Reshape((1,output_cnn.shape[2],-1))(output_cnn)
    reshape_cnn = Lambda(lambda x: squeeze(x, 1))(x)
    
    # Couche dense 
    dense = Dense(256, name = 'dense_1')(reshape_cnn)
    dense = Activation("relu")(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    
    dense = Dense(128, name = 'dense_2')(reshape_cnn)
    dense = Activation("relu")(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    
    # RNN
    blstm = Bidirectional(LSTM(64, activation='relu', return_sequences=True, dropout=0.2,
                               name="blstm1"))(dense)
    blstm2 = Bidirectional(LSTM(64, activation='relu', return_sequences=True, dropout=0.2,
                               name="blstm2"))(blstm)
    
    # output layer
    y_pred = Dense(len(list(string.printable[:-17]))+1, activation='softmax', name="dense")(blstm2)
    
    # ctc layer pour calcul de la CTC loss à chaque step
    output_ctc = CTCLayer(name="ctc_batch_cost")(labels, y_pred, y_lengths)
    
    # Définission du modèle
    model = Model(inputs=[inputs_data, labels, y_lengths], outputs=output_ctc, name="rnn2")
    
    # compiler le model
    model.compile(optimizer=tensorflow.keras.optimizers.Adam())
    
    print(model.summary())
    
    return model