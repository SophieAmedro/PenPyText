"""
Created on Tue Jul 13 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""

# Importations
import pandas as pd
import numpy as np

import tensorflow
import argparse

# fonctions
import traintestsplit as tts
from words_txt_to_df import txt_to_df
from nettoyage_fichiers import clean_data, error_image
from harmonisation import harmony_clean

# modèle cnn
import preprocess_cnn as ppcnn
from cnn_model import build_model_cnn

# prédictions
import cnn_pred

# Visualisation
import matplotlib.pyplot as plt

# Reproductibilité
from numpy.random import seed
seed(12)
from tensorflow import random
random.set_seed(23)


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
    TARGET_SIZE = (128,32)
    BACTH = 100
    EPOCHS = 25

    print("Préparation des données d'entrainement, validation et test")
    Xy_train = ppcnn.data_Xy(trainset, TARGET_SIZE, trainset, BACTH)
    Xy_valid = ppcnn.data_Xy(validationset, TARGET_SIZE, trainset, BACTH)
    Xy_test = ppcnn.data_Xy(testset, TARGET_SIZE, trainset, BACTH)

    print("Construction du modèle")
    NB_CLASS = len(ppcnn.labels_dict(trainset))
    cnn = build_model_cnn(NB_CLASS, TARGET_SIZE)

    print(" Entrainement du modèle")
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(filepath = 'cnn.weights.h5',
                                                   save_weights_only=True,
                                                   monitor = 'val_accuracy', 
                                                   mode = 'max',
                                                   save_best_only=True)]
    history = cnn.fit(Xy_train, 
                      epochs = EPOCHS, 
                      validation_data = Xy_valid, 
                      callbacks = callbacks)
    
    # Tracer l'évolution des précisions tout au long de l'entraînement.
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy by epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.show()
    

    print("prédictions")
    cnn.load_weights('cnn.weights.h5')
    pred_key = cnn_pred.predictions_top5(cnn, testset, target_size = TARGET_SIZE, batch_size = BACTH)
    dict_code = ppcnn.labels_dict(trainset)
    pred_words = cnn_pred.decoder_pred(pred_key, dict_code)

    # prédiction top 5
    df_top5  = cnn_pred.df_bilan_top5(testset, pred_words)
    print("Top 5 predictions \n",df_top5)
    
if __name__ == '__main__':
    main()
        
    
    
    

