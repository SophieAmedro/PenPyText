"""
Created on Tue Jul 13 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

import numpy as np
import pandas as pd

import string

import matplotlib.pyplot as plt
import cv2

import tensorflow 
from tensorflow import keras

# fonctions de récupération et préparation des données
import traintestsplit as tts
from words_txt_to_df import txt_to_df
from nettoyage_fichiers import clean_data, error_image
from harmonisation import harmony_clean
from keep_n_chars import max_n_chars

# Générateur de batchs
from  generator_rnn import DatasetGenerator

# Modèlisation
import rnn_model
from cer_callback import CerCallback

# Décodage
import rnn_pred 

# Reproductibilité
from numpy.random import seed
seed(64)
from tensorflow import random
random.set_seed(8)

def main():
    print("Lecture de words.txt et transformation en dataframe") 
    #[word_id,ok_err,gray_level,transcript,data_path]
    df_words = txt_to_df('words.txt')

    print("Suppression des erreurs de lecture d'image du df")
    df_words = clean_data(df_words)

    print("génération d'un dataframe contenant la répartition du dataset [line_id,set]")
    df_tts = tts.text_to_splitDataframe()

    print("Split des données")
    trainset, testset, validationset = tts.split_data(df_tts, df_words)
    
    # Variables utiles
    BATCH_SIZE = 100 # 32 par défaut
    TARGET_SIZE = (128,32) 
    MAX_LENGTH = 10 # 21 par défaut
    EPOCHS = 25
    
    # Suppression des mots plus longs que MAX_LENGTH 
    trainset = max_n_chars(trainset, MAX_LENGTH)
    testset = max_n_chars(testset, MAX_LENGTH)
    validationset = max_n_chars(validationset, MAX_LENGTH)
    
    # Préparation des données
    print("Préparation des données")
    
    # Tranformation des images par lots
    train_generator = DatasetGenerator(dataframe=trainset,
                                       directory="",
                                       x_col = "data_path",
                                       y_col = "transcript",
                                       targetSize = TARGET_SIZE,
                                       nb_canaux = 1, # images grayscale par défaut
                                       batchSize = BATCH_SIZE,
                                       shuffle = False,
                                       max_y_length = MAX_LENGTH)

    # Idem pour le jeu de validation
    valid_generator = DatasetGenerator(dataframe=validationset,
                                       x_col = "data_path",
                                       y_col = "transcript", 
                                       targetSize = TARGET_SIZE,  
                                       shuffle = False, 
                                       max_y_length = MAX_LENGTH)

    # Et le jeu de test
    test_generator = DatasetGenerator(dataframe=testset,
                                       x_col = "data_path",
                                       y_col = "transcript", 
                                       targetSize = TARGET_SIZE,  
                                       shuffle = False, 
                                       max_y_length = MAX_LENGTH)
    
    print("Construction du modèle")
    rnn = rnn_model.build_model_rnn(TARGET_SIZE)
    
    print("Entrainement")
    current_pred = keras.models.Model(rnn.get_layer(name="input_im").input, rnn.get_layer(name="dense").output)
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(filepath = 'rnn.weights.h5', 
                                                   monitor = 'val_loss', 
                                                   mode = 'min',
                                                   save_best_only=True), 
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=5,
                                                     factor=0.1,
                                                     verbose=2,
                                                     mode='min')]
    history = rnn.fit(train_generator,
                      steps_per_epoch = len(trainset)//train_generator.batchSize,
                      validation_data = valid_generator,
                      validation_steps = len(validationset)//valid_generator.batchSize,
                      epochs = EPOCHS, callbacks = callbacks)
    
    # Tracer l'évolution des pertes tout au long de l'entraînement.
    plt.figure(figsize=(12,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    
    print("prédictions")
    rnn.load_weights('rnn.weights.h5')
    predictions = rnn.predict(test_generator)
    pred_key = rnn_pred.pred_top5(predictions, MAX_LENGTH)
    pred_words = rnn_pred.df_bilan_top5(testset, pred_key)
    
    print("Top 5 predictions \n",pred_words)
    
if __name__ == '__main__':
    main()
        
