"""
Created on Tue Aug 19 2021

@author: samy_ait-ameur, sophie_amedro, stephane_tchatat
"""
import pandas as pd
import glob
import os
import words_txt_to_df
import harmonisation
import cv2

def text_to_splitDataframe():
    """
     Cette fonction lit les fichiers .txt repartissant les données en différents ensembles (testset, trainset, 
     validationset1, validationset2) et retourne un dataframe contenant l'ensemble des informations.
     Les fichiers .txt sont contenus dans le repertoire data/largeWriterIndependentTextLineRecognitionTask
    :param
    :return: Dataframe trainset, testset, validationset1, validationset2
    """
    # Le répertoire largeWriterIndependentTextLineRecognitionTask contient les fichiers  testset, trainset, validationset1,
    # validationset2 à utiliser pour séparer notre jeux de données
    root = 'data/largeWriterIndependentTextLineRecognitionTask/'
    # Dictionnaire stockant le contenu des fichiers de split
    split_dict ={}
    for f in glob.glob(root+"*.txt"):
        #récupération du nom du fichier lu
        name = os.path.basename(f).split('.')[0]
        if (name != 'LargeWriterIndependentTextLineRecognitionTask'):
            #lecture du fichier du répertoire
            with open(f, "r") as file :
                for line in file:
                    line = str(line.strip())
                    #Enregistre le contenu dans le dictionnaire en utilisant le nom du fichier comme flag pour la donnée
                    split_dict[line] = name
    df = pd.DataFrame(list(split_dict.items()),
                   columns=['line_id', 'set'])
    return df

def split_data(repartition, data_words):
    """
    Cette fonction prend en entrée la répartition à appliquer aux données, l'applique à l'ensemble des données
    et crée les dataframes pour les images appartenant aux groupes respectifs trainset, testset, validationset.
        :parameter repartition: donnant la répartition des données
                   data_words: dataframe contenant les informations sur les images
        :return: dataframes 
    """
    
    # Calcul du line_id auquel appartient chaque mot
    data_words['line_id'] = data_words['word_id'].apply(lambda x: x[:-3])
    #Jointure entre les données et la dataframe de répartition  sur la colonne 'line_id'
    df = data_words.merge(right= repartition, how='inner', on='line_id')
    
    # Création des dictionnaires pour récupération des informations
    train_dict = {}
    test_dict = {}
    #valid1_dict = {}
    valid_dict = {}
    
    
    #Lectures des lignes de mon dataset
    for i in range(len(df)):
        
        #Compléter les dictionnaires
        if (df.loc[i,'set'] == 'trainset') | (df.loc[i,'set'] == 'validationset1'):
        #if (df.loc[i,'set'] == 'trainset'):
            train_dict[i] = {'data_path': df.loc[i,'data_path'], 
                             'transcript': df.loc[i,'transcript'],
                             'gray_level': df.loc[i,'gray_level'],
                             'ok_err': df.loc[i, 'ok_err']}
        elif (df.loc[i,'set'] == 'testset'):
            test_dict[i] = {'data_path': df.loc[i,'data_path'], 
                            'transcript': df.loc[i,'transcript'],
                            'gray_level': df.loc[i,'gray_level'],
                            'ok_err': df.loc[i, 'ok_err']}
        elif (df.loc[i,'set'] == 'validationset2'):
            valid_dict[i] = {'data_path': df.loc[i,'data_path'], 
                             'transcript': df.loc[i,'transcript'],
                             'gray_level': df.loc[i,'gray_level'],
                             'ok_err': df.loc[i, 'ok_err']}
   
    trainset = pd.DataFrame.from_dict(train_dict, "index").reset_index()
    testset = pd.DataFrame.from_dict(test_dict, "index").reset_index()
    validationset = pd.DataFrame.from_dict(valid_dict, "index").reset_index()
    
    # Suppression des images avec erreur de segmentation sur le trainset
    trainset = trainset[trainset['ok_err'] == 'ok']
    trainset = trainset.reset_index()
    
    return trainset, testset, validationset



