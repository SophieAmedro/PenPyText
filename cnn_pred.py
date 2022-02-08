"""
Created on Tue Dec 02 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""
import numpy as np
import pandas as pd
import preprocess_cnn as ppcnn


def predictions_top5(model, dataset, target_size = (128,32), batch_size = 100):
    """
    La fonction predictions_top5 retourne le top 5 des classes prédites pour chaque image
    Paramètres :
        model: modèle cnn entraîné
        dataset : dataframe contenant les informations image_path, gray_level des images à transcrire
        target_size : tuple  taille des images 
        batch_size: int taille des batchs pour exécuter le modèle de prédiction
    Renvoie:
        predicted_class_indices: array des cinq premières classes prédites
    """
    X = ppcnn.features(dataset, target_size)
    nb_viz = len(X)
    probas_pred = model.predict(X, steps = nb_viz // batch_size)
    # indices/classes des 5 plus grandes probas dans l'ordre croissant des probas
    predicted_class_indices=np.argpartition(probas_pred,-5,axis=1)[:,-5:] 
    return predicted_class_indices


def decoder_pred(pred, dict_code):
    """
    La fonction decoder_pred décode les classes prédites en mot à l'aide d'un dictionnaire 
    Paramètres :
        pred: array contenant les classes encodées prédites 
        dict_code : dictionnaire contenant la correspondance code = int / mots = str
    Renvoie:
        words_pred: array des cinq premières transcriptions
    """
    words_pred = np.copy(pred).astype('str')
    for i in range(len(pred)): # parcourt de chaque ligne
        for k in range(5):# décoder chaque nombre en mot 
            idx = pred[i,k]
            words_pred[i,k] = list(dict_code.keys())[list(dict_code.values()).index(idx)]
    return words_pred


def df_bilan_1(dataset, words_preds):
    """
    La fonction df_bilan_1 renvoie un dataframe comprenant les chemins des images et leur meilleure transcription
    Paramètres :
        dataset : dataframe contenant les informations image_path, gray_level des images à transcrire
        words_pred: array des cinq premières transcriptions
    Renvoie:
        bilan: dataframe
    """
    bilan = pd.DataFrame({"data_path":dataset.data_path[:len(words_preds)],
                          "predict_1":words_preds[:,4],
                          "transcript": dataset.transcript[:len(words_preds)]})
    return bilan

def df_bilan_top5(dataset, words_preds):
    """
    La fonction df_bilan_top5 renvoie un dataframe comprenant les chemin des images et leur 5 meilleures transcriptions
    Paramètres :
        dataset : dataframe contenant les informations image_path, gray_level des images à transcrire
        words_pred: array des cinq premières transcriptions
    Renvoie:
        bilan: dataframe
    """
    bilan = pd.DataFrame({"data_path":dataset.data_path[:len(words_preds)],
                          "predict_1":words_preds[:,4], 
                          "predict_2":words_preds[:,3],
                          "predict_3":words_preds[:,2],
                          "predict_4":words_preds[:,1],
                          "predict_5":words_preds[:,0], 
                          "transcript": dataset.transcript[:len(words_preds)]})
    return bilan
       
 
    