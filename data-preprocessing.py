import os, errno
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import csv
import numpy as np
import pickle
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, LabeledSentence, Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from collections import namedtuple
from statistics import mean

import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE



import nltk;
#nltk.download();
from nltk.stem import WordNetLemmatizer;
from nltk.corpus import wordnet;
#nltk.download('wordnet');
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
import csv
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
import timeit

from sklearn.cluster import KMeans

############### Tokenization ######################################
stopwordsList = stopwords.words('english')
class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+');
        
    def split(self,text,stopwordsList):
        # split into single sentence
        sentences = self.splitter.tokenize(text.lower())
        # tokenization in each sentences and ponctuation removal
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        # stop words removal
        filtered_words = [x for x in tokens[0] if x not in stopwordsList]
        #list(set(tokens[0]) - set(stopwords.words('english')))
        return filtered_words

############## POS Tagger + Lemmatization ###########################
class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN
    
    def pos_tag(self,tokens):
        lemmatizer = WordNetLemmatizer();
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in [tokens]]
        # lemmatization using pos tagg   
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        #pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        pos_tokens = [ [ lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens[0]

def tokenization(text):
    tokens=text.split();
    return(tokens)

def cleanning(text):
    splitter = Splitter();
    return splitter.split(text,stopwordsList);

def stemming(text, stemmingMethod):#stemmingMethod = 1:SnowballStemmer("english"), 2:LancasterStemmer(), 3:PorterStemmer()
    tokens = cleanning(text);
    if stemmingMethod==1:
        stemmer = SnowballStemmer("english");
    else: #2 ou 3
        if stemmingMethod==2:
            stemmer = LancasterStemmer();
        else: #3
            stemmer = PorterStemmer();
    stemmed = [stemmer.stem(t) for t in tokens];
    return stemmed;

def lemmatization(text, lemmatizationMethod): #lemmatizationMethod = 1:LemmatizationWithoutPOSTagger, 2:LemmatizationWithPOSTagger
    tokens = cleanning(text);
    if lemmatizationMethod==1:
        lemmatizer = WordNetLemmatizer();
        lemma = [lemmatizer.lemmatize(t) for t in tokens];
    else: #2
        lemmatizer = LemmatizationWithPOSTagger();
        lemma = lemmatizer.pos_tag(tokens);
    return lemma;

########### apprentissage des modeles Doc2Vec pour chaque DataSet  #############

choosed_corpus = 'MSRParaphraseCorpus';#'MSRParaphraseCorpus' ou 'stsbenchmark'
#choosed_preprocessing_method = 22;#0,1,21,22,23,31,32
choosed_iteration = 10;#nombre d'iterations pour l'apprentissage
#choosed_dm = True;# False:CBOW, True:SkipGram 
choosed_alpha = 0.025;
choosed_min_alpha = 0.025;
choosed_size = 20;
choosed_window = 300;
choosed_min_count = 0;



def preprocess(text,choosed_preprocessing_method):
    switcher = {
        0: tokenization(text),
        1: cleanning(text),
        21: stemming(text,1),#snowball stemmer
        22: stemming(text,2),#lancaster stemmer
        23: stemming(text,3),#porter stemmer
        31: lemmatization(text,1),#without pos tagger
        32: lemmatization(text,2)#with pos tagger
    }
    return switcher.get(choosed_preprocessing_method, tokenization(text));

def convertData(choosed_corpus, trainOrTest, choosed_preprocessing_method, crossValidationFold):
    doc2vecSavingDirectory = choosed_corpus+'/Doc2Vec/preprocess='+str(choosed_preprocessing_method)+'/ConvertedData';
    try:
        os.makedirs(doc2vecSavingDirectory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    fileName=trainOrTest+str(crossValidationFold+1);
    filePath=choosed_corpus +'/'+ fileName +'.txt';
    convertedDocs=[];
    boolVec=[];
    with open(filePath, encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile,delimiter ='\t');
            for row in reader:
                if choosed_corpus == 'MSRParaphraseCorpus':
                    words=preprocess(row[3],choosed_preprocessing_method)#sentence 1
                    tags=[eval(row[1])]#identifiant 1
                    convertedDocs.append(TaggedDocument(words, tags))# integrer 1 dans la liste d'apprentissage
                    words=preprocess(row[4],choosed_preprocessing_method)#sentence 2
                    tags=[eval(row[2])]#identifiant 2
                    convertedDocs.append(TaggedDocument(words, tags))# integrer 2 dans la liste d'apprentissage
                    boolVec.append(bool(eval(row[0])))#sauvegarder le resultat desiree (similarite entre 1 et 2) dans boolVec
    with open(doc2vecSavingDirectory+'/'+fileName+'_tagged_documents.txt', "wb") as fp: #sauvegarder les donnees(documents) apres reformation dans un fichier
        pickle.dump(convertedDocs, fp)   #Pickling
    with open(doc2vecSavingDirectory+'/'+fileName+'_resultats_desires.txt', "wb") as fp:#sauvegarder les resultats desires dans un fichier
        pickle.dump(boolVec, fp)#Pickling
    return(convertedDocs, boolVec);



def trainDoc2vecModel(choosed_corpus, choosed_preprocessing_method, crossValidationFold, choosed_iteration, choosed_dm, choosed_alpha, choosed_min_alpha, choosed_size, choosed_window, choosed_min_count):
    doc2vecSavingDirectory = choosed_corpus+'/Doc2Vec/preprocess='+str(choosed_preprocessing_method);
    fileName = 'train'+ str(crossValidationFold+1);
    filePath = doc2vecSavingDirectory+'/ConvertedData/'+ fileName + '_tagged_documents.txt' ;#preciser le chemin
    myDocs = pickle.load(open(filePath, "rb"));#load les documents test reformees pour appliquer le test
    myModel = Doc2Vec(myDocs, dm=choosed_dm, alpha=choosed_alpha, size=choosed_size, window=choosed_window, min_alpha=choosed_min_alpha, min_count=choosed_min_count, workers=8);#definir le modele
    myModel.train(myDocs, total_examples=len(myDocs), epochs=choosed_iteration);
    doc2vecModelsSavingDirectory = doc2vecSavingDirectory+'/Models/'+ fileName ;
    try:
        os.makedirs(doc2vecModelsSavingDirectory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    modelFileName = 'it='+str(choosed_iteration)+'_dm='+str(choosed_dm)+'_alpha='+str(choosed_alpha)+'_minAlpha='+str(choosed_min_alpha)+'_size='+str(choosed_size)+'_window='+str(choosed_window)+'_minCount='+str(choosed_min_count)+'.model'
    myModel.save(doc2vecModelsSavingDirectory+'/'+modelFileName);#sauvegarder le modele entrainee
    return(myModel)  



def evaluateDoc2vecModel(choosed_corpus, choosed_preprocessing_method, crossValidationFold, choosed_iteration, choosed_dm, choosed_alpha, choosed_min_alpha, choosed_size, choosed_window, choosed_min_count):
    
    doc2vecSavingDirectory = choosed_corpus+'/Doc2Vec/preprocess='+str(choosed_preprocessing_method);

    trainFileName = 'train'+ str(crossValidationFold+1);
    trainDocsPath = doc2vecSavingDirectory+'/ConvertedData/'+ trainFileName + '_tagged_documents.txt' ;#preciser le chemin
    trainDocs = pickle.load(open(trainDocsPath, "rb"));#load les documents test reformees pour appliquer le test
    print('trainDocs : ok')

    testFileName = 'test'+ str(crossValidationFold+1);
    testDocsPath = doc2vecSavingDirectory+'/ConvertedData/'+ testFileName + '_tagged_documents.txt' ;#preciser le chemin
    testDocs = pickle.load(open(testDocsPath, "rb"));#load les documents test reformees pour appliquer le test
    print('testDocs : ok')

    myModelFileName = 'it='+str(choosed_iteration)+'_dm='+str(choosed_dm)+'_alpha='+str(choosed_alpha)+'_minAlpha='+str(choosed_min_alpha)+'_size='+str(choosed_size)+'_window='+str(choosed_window)+'_minCount='+str(choosed_min_count)+'.model'
    myModelPath = doc2vecSavingDirectory+'/Models/'+ trainFileName +'/'+myModelFileName ;#preciser le chemin
    myModel = pickle.load(open(myModelPath, "rb"));#load les documents test reformees pour appliquer le test
    print('myModel : ok')

    trainResultatsDesirePath = doc2vecSavingDirectory+'/ConvertedData/'+ trainFileName + '_resultats_desires.txt' ;#preciser le chemin
    trainResultatsDesire = pickle.load(open(trainResultatsDesirePath, "rb"));#load les documents test reformees pour appliquer le test
    print('trainResultatsDesire : ok')
    
    testResultatsDesirePath = doc2vecSavingDirectory+'/ConvertedData/'+ testFileName + '_resultats_desires.txt' ;#preciser le chemin
    testResultatsDesire = pickle.load(open(testResultatsDesirePath, "rb"));#load les documents test reformees pour appliquer le test
    print('testResultatsDesire : ok')

    trainResultatsObtenus = [];
    for i in range(len(trainDocs)-1):
        if i%2==0:
            trainResultatsObtenus.append([myModel.docvecs.similarity_unseen_docs(myModel, trainDocs[i].words, trainDocs[i+1].words, alpha=myModel.alpha, min_alpha=myModel.min_alpha, steps=myModel.iter)])
    print('trainResultatsObtenus : ok')
    print(trainResultatsObtenus[1:100])
    print(trainResultatsDesire[1:100])
    
    clf = KNeighborsClassifier(3).fit(trainResultatsObtenus, trainResultatsDesire);
    print('knn train : ok')

    testResultatsObtenus = [];
    for i in range(len(testDocs)-1):
        if i%2==0:
            testResultatsObtenus.append([myModel.docvecs.similarity_unseen_docs(myModel, testDocs[i].words, testDocs[i+1].words, alpha=myModel.alpha, min_alpha=myModel.min_alpha, steps=myModel.iter)])
    print('testResultatsObtenus : ok')
    predicted = clf.predict(testResultatsObtenus)
    print('knn predict : ok')

    f1_score = metrics.f1_score(testResultatsDesire, predicted)
    print('f1_score = '+ str(f1_score))
 
    return(f1_score);



trainOrTest = 'train'
choosed_dm = True;# False:CBOW, True:SkipGram
choosed_preprocessing_method = [1]
crossValidationFold = 1;

#for crossValidationFold in range(3):
for pre in choosed_preprocessing_method:
    convertData(choosed_corpus, trainOrTest, pre, crossValidationFold)
    #res = evaluateDoc2vecModel(choosed_corpus, pre, crossValidationFold, choosed_iteration, choosed_dm, choosed_alpha, choosed_min_alpha, choosed_size, choosed_window, choosed_min_count)
    #print('F1 score de ' + str(pre) +' = ' +str(res))
