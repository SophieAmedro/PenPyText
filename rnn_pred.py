"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""
import numpy as np
import pandas as pd

import tensorflow

# Décodage
import tensorflow.keras.backend as K
from  generator_rnn import transcript_vect_mot

def pred_top5(predictions, max_length: int):
    """
    La fonction pred_top5 décode avec la ctc_decode le top 5 des prédictions pour chaque image
    Paramètres :
        predictions: array contenant les classes encodées prédites
        max_length: int: nombre de caractère maximum des prédictions
    Renvoie:
        results_pred_top5: array des cinq premières prédictions 
    """
    results_pred_top5 = []
    for i in range(5):
        labels = K.get_value(K.ctc_decode(predictions, 
                                          input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                                          greedy=False, beam_width = max_length, top_paths = 5)[0][i])
        results_pred_top5.append(labels)
    return results_pred_top5


def df_bilan_top5(dataset, pred_top5):
    """
    La fonction df_bilan_top5 renvoie un dataframe comprenant les chemin des images et leur 5 meilleures transcriptions
    Paramètres :
        dataset : dataframe contenant les informations image_path, gray_level des images à transcrire
        pred_top5: array des cinq premières transcriptions
    Renvoie:
        bilan: dataframe
    """
    top5_rnn = pd.DataFrame(columns = ['data_path', 'predict_1', 'predict_2', 'predict_3',
                                              'predict_4','predict_5'])
    top5_rnn.data_path = dataset.data_path
    top5_rnn[top5_rnn.columns[1:]] = np.nan

    for t in range(1, top5_rnn.shape[1]):
        top = t-1 # la première colonne de top5_rnn est la vraie pred
        for i in range(len(pred_top5[top])):
            k = 0
            result = []
            while (pred_top5[top][i,k] != -1):
                result.append(pred_top5[top][i,k])
                k +=1
            mot = transcript_vect_mot(result)
            top5_rnn[top5_rnn.columns[t]][i] = ''.join(mot)
    return top5_rnn