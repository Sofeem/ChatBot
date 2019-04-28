# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:21:42 2019

@author: HP
"""
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


#import numpy as np
#from nltk.corpus import nps_chat


# Dataset
chat_utterances = nltk.corpus.nps_chat.xml_posts()[:10000]
#print(chat_utterances) 
data = []
data1 = []
word_tokens = []
datada = []
datawords = []

# 15 Dialgue act tag
dialogue_acts = ['Accept', 
                 'Bye', 
                 'Clarify', 
                 'Continuer', 
                 'Emotion', 
                 'Emphasis', 
                 'Greet', 
                 'nAnswer', 
                 'Other', 
                 'Reject', 
                 'Statement', 
                 'System', 
                 'whQuestion', 
                 'yAnswer', 
                 'ynQuestion']

for a in dialogue_acts :
    for u in chat_utterances :
        if u.get('class') == a:
           #print ("Example of {}: {}".format(a, u.text))
           data.append(a)
           data1.append(u.text)
           #data[1].append([u.text])
           #data.extend([u.text])
           
          

#print(data)
#print(data1)

# Creating a dataframe object from listoftuples
df1 = pd.DataFrame(data).reset_index(drop=True)
df2 = pd.DataFrame(data1).reset_index(drop=True) 

#print(df1.head)
#print(df2.head)

FinalDataset = pd.concat([df1, df2], axis=1, ignore_index=False)
FinalDataset.columns = ['Dialogue_act','body_text']


#pre/processing & Feature Extraxtion
    
#FinalDataset['body_text'].dropna
FinalDataset['words'] = FinalDataset.body_text.str.strip().str.split('[\W_]+')
print(FinalDataset.head)



rows = []

for i, row in FinalDataset[['Dialogue_act', 'words']].iterrows():
    for word in row.words:
        rows.append([word, row.Dialogue_act])

words = pd.DataFrame(rows, columns=['words', 'Dialogue_act'])
words.head()

dataframe = words[words.words.str.len() > 0]
dataframe.head()



dataframe['words'] = dataframe.words.str.lower()
dataframe.head()

counts = dataframe.groupby('Dialogue_act')\
    .words.value_counts()\
    .to_frame()\
    .rename(columns={'words':'n_w'})
counts.head()



word_sum = counts.groupby(level=0)\
    .sum()\
    .rename(columns={'n_w': 'n_d'})
word_sum

tf = counts.join(word_sum)

tf['tf'] = tf.n_w/tf.n_d

tf.head()

c_d = dataframe.Dialogue_act.nunique()
print (c_d)

idf = dataframe.groupby('Dialogue_act')\
    .Dialogue_act\
    .nunique()\
    .to_frame()\
    .rename(columns={'Dialogue_act':'i_d'})\
    .sort_values('i_d')
idf.head()


idf['idf'] = np.log(c_d/idf.i_d.values)

idf.head()



tf_idf = tf.join(idf)

tf_idf.head()

tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf
FDataset = tf_idf
FDataset['index1'] = FDataset.index
FDataset.reset_index(drop=True,col_level=1)

FDataset = FDataset.sort_values(by = ['index1'])
FDataset = FDataset.reset_index(drop=True)



strd = FDataset.index1.values[:]
strd  = np.sort(strd)

rang = np.size(strd,0)

for i in range (0,rang):
    datada.append(strd[i][0])
    datawords.append(strd[i][1])



print(datawords)



print(datada)

df3 = pd.DataFrame(datawords).reset_index(drop=True)
df4 = pd.DataFrame(datada).reset_index(drop=True) 

FDataset = FDataset.drop("index1", axis=1)

Dataset = pd.concat([df3,df4], axis=1)
Dataset.columns = ['Body_words', 'Dialogue_act']




FinalDatasetub =pd.concat([FDataset,Dataset], axis=1)


# Train Set and Test Data Set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)

'''
rows = list()
for i in range (0,r):
    row = FinalDataset.loc[r,['Dialogue_act', 'words']].iterrows()
    r = row[i]
    for word in r.words:
        rows.append((r.Dialogue_act, word))

words = pd.DataFrame(rows, columns=['book', 'word'])
words.head()'''
#print(FinalDataset)
#print(FinalDataset['body_text'].value_counts())
#print(FinalDataset['Dialogue_act'])

#print(FinalDataset["body_text"])
#Preprocessing
#for w in FinalDataset["body_text"]:
   # word_tokens.append(nltk.word_tokenize(w))
   
#print(word_tokens)

#mylist_string = [str(x) for x in data1]

#print(mylist_string)

#Preprocessing
'''from sklearn.feature_extraction.text import CountVectorizer

word_tokens = nltk.word_tokenize(FinalDataset.iloc[:]["body_text"])
word_tokens[:2]
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(FinalDataset['body_text'])
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())


print(vectorizer.vocabulary_)'''
#Preprocessing

'''stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer() 

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text'''


'''from sklearn.model_selection import train_test_split
X=FinalDataset['body_text']
y=FinalDataset['Dialogue_act']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])

tfidf_train = tfidf_vect_fit.transform(X_train)
tfidf_test = tfidf_vect_fit.transform(X_test)

X_train_vect = pd.DataFrame(tfidf_train.toarray())
X_test_vect =  pd.DataFrame(tfidf_test.toarray())

X_train_vect.head()
classifier = nltk.NaiveBayesClassifier.train(X_train_vect)
print(nltk.classify.accuracy(classifier, X_test_vect))'''