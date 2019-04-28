# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:05:27 2019

@author: Sofee
"""

import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords


data = []
data2 = []

Utteranc_1  = 'Define Another example. Well, I am so interested in the assigment 2 based on Tokenization'
Utterance_2 = 'tokenization is quite difficult to understand. Well, I am so interested in the assigment 2 based on Tokenization'

data.append(Utteranc_1)
data2.append(Utterance_2)
      
NUM_TOPICS = 1

def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = word_tokenize(text)
    text = remove_stop_words(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

Utteranc_1_data = []
for text in data:
    Utteranc_1_data.append(initial_clean(text))
print(Utteranc_1_data)   

Utteranc_2_data = []
for text in data2:
    Utteranc_2_data.append(initial_clean(text))
print(Utteranc_2_data)  
#Utteranc_1 = initial_clean(Utteranc_1)

dictionary = corpora.Dictionary(Utteranc_1_data)
#print(corpus) 
corpus = [dictionary.doc2bow(doc) for doc in Utteranc_1_data]
print(corpus) 

#model = models.doc2vec(corpus)
#dcorpus = model[corpus] 


#tfidf_model = models.TfidfModel(corpus)
#tfidf_corpus = tfidf_model[corpus] 


lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
#res = lda_model.get_document_topics(dictionary)
#print(res)
print("LDA Model 1:")
 
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print(lda_model.print_topic(idx, 4))





#vec_bow_U1 = dictionary.doc2bow(Utteranc_1_data[0])
#vector_utternce1 = lda_model[vec_bow_U1]
#print(vector_utternce1)
#vec_bow_17 = np.asarray(corpus)
#print(vec_bow_17)


#tfidf_model = models.TfidfModel(corpus)
#tfidf_corpus = tfidf_model[corpus]




#lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
#vector_utternce1 = lda_model[corpus]

#print("LDA Model U1:")
 
#for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    #print("Topic #%s:" % idx, lda_model.print_topic(idx, 4))
 
#print("=" * 20)

#print(Utteranc_2_data)
dictionary2 = corpora.Dictionary(Utteranc_2_data)
corpus2 = [dictionary2.doc2bow(doc) for doc in Utteranc_2_data]

#tfidf_model2 = models.TfidfModel(corpus2)
#tfidf_corpus2 = tfidf_model[corpus2] 
#vec_bow_1 =   dictionary.doc2bow(Utteranc_2_data[0])
#print(vec_bow_1)
lda_model2 = models.LdaModel(corpus=corpus2, num_topics=NUM_TOPICS, id2word=dictionary2)


print("LDA Model 2:")
 
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print(lda_model2.print_topic(idx, 4))
#vector_utternce2 = lda_model2[corpus2]
#vec_bow_U2 = dictionary.doc2bow(Utteranc_2_data[0])
#vector_utternce2 = lda_model2[vec_bow_U2]
#print(vector_utternce2)

#print("LDA Model U2: %s" %lda_model2)
#Topic_modeling1 = []
 
#for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    #print("Topic #%s:" % idx, lda_model2.print_topic(idx, 4))
    #Topic_modeling1.append(lda_model2.print_topic(idx, 4))
          










#similarity = 
#simmilarity = gensim.matutils.cossim(vector_utternce2, vector_utternce1)
#print(simmilarity)

#def get_cosine_sim(*strs): 
    #vectors = [t for t in get_vectors(*strs)]
    #return cosine_similarity(vectors)

