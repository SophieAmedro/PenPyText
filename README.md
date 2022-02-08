# projet_penpytext
Océrisation de documents manuscrits (projet DataScientest)

# Données

Pour réaliser ce projet, nous utiliserons le dataset IAM Handwriting Database 3.0 disponible sur le site:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

# Installations et dépendances à installer au préalable

- OpenCV  
- tensorflow 2.4
- jiwer
- streamlit

Requirements:
- python==3.9.7
- matplotlib==3.2.1
- numpy==1.19.5
- pandas==1.3.3
- opencv==4.5.1
- tensorflow==2.5.0

# Téléchargement des données

Pour obtenir les données du dataset IAM Handwriting Database 3.0

- Créez un dossier sur votre disque nommé *data*  

- Créez un compte sur https://fki.tic.heia-fr.ch/databases/iam-handwriting-database afin de pouvoir télécharger les données

- Télécharger les fichiers: + ascii.tgz  + words.tgz  + Large Writer Independent Text Line Recognition Task
Déposer ces fichiers dans votre dossier *data* local et les dézipper.  
  
<https://fki.tic.heia-fr.ch/DBs/iamDB/data/ascii.tgz> dans *data/ascii/*. 

<https://fki.tic.heia-fr.ch/DBs/iamDB/data/words.tgz> dans *data/words/*. 

<https://fki.tic.heia-fr.ch/static/zip/largeWriterIndependentTextLineRecognitionTask.zip> dans *data/largeWriterIndependentTextLineRecognitionTask/*. 

# Lancer le projet

- Pour exécuter le modèle CNN.  
 Depuis la racine du projet, exécuter 
 <code>python  main_cnn.py</code>

- Pour exécuter le modèle RNN.    
 Depuis la racine du projet, exécuter 
 <code>python  main_rnn.py</code>
 
- Pour lancer l'application streamlit.  
Depuis le dossier streamlit_app.  
  <code>streamlit run app.py</code>
