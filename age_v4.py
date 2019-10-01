# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:01:55 2018

@author: mw
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy.linalg as LA
import re
from sympy import Point, Line
import time

start = time.time()

documents = open("tes2.csv", encoding="ISO-8859-1").readlines()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
#
#wcss = []
#for i in range(1,21):
#    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#
#jarak = []
#for i in range(0,20):
#    p1 = Point(1, wcss[0])
#    p2 = Point(20, wcss[19])
#    s = Line(p1, p2)
#    p = Point(i+1, wcss[i])
#    jarak.append(s.distance(p))
#
#m = max(jarak)
#z = [i for i, j in enumerate(jarak) if j == m]

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
label_data = model.labels_
label = pd.DataFrame(label_data)
label.columns = ['cluster']

Cluster = [(label.loc[label['cluster'] == y]) for y in range(0, true_k)]
Sample = [Cluster[x].loc[np.random.choice(Cluster[x].index, size=1)] for x in range(len(Cluster))]
Sample_set = [documents[Sample[z].index.values.astype(int)[0]] for z in range(len(Sample))]

result = []
for n in range(len(Sample_set)):

    #define probability array
    prob=[]
    for i in range(4):
     prob.append(1)
     
    #input test set 
    file = Sample_set[n]
    clean = re.sub(r'[^\x00-\x7F]+',' ', file)
    dok = clean.replace('\n',' ')
    dok = dok.split('\n')
    test_set = dok
     
    #check teenager
    file = open('teenagerclean.txt', encoding='latin-1').read()
    clean = re.sub(r'[^\x00-\x7F]+',' ', file)
    doc = clean.replace('\n',' ')
    doc = doc.split('\n')
    train_set = doc
    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
    for vector in trainVectorizerArray:
        for testV in testVectorizerArray:
            cosine = cx(vector, testV)
            prob[0] = cosine
            
    #check young adult
    file = open('youngadultclean.txt', encoding='latin-1').read()
    clean = re.sub(r'[^\x00-\x7F]+',' ', file)
    doc = clean.replace('\n',' ')
    doc = doc.split('\n')
    train_set = doc
    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
    for vector in trainVectorizerArray:
        for testV in testVectorizerArray:
            cosine = cx(vector, testV)
            prob[1] = cosine
            
    #check adult
    file = open('adultclean.txt', encoding='latin-1').read()
    clean = re.sub(r'[^\x00-\x7F]+',' ', file)
    doc = clean.replace('\n',' ')
    doc = doc.split('\n')
    train_set = doc
    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
    for vector in trainVectorizerArray:
        for testV in testVectorizerArray:
            cosine = cx(vector, testV)
            prob[2] = cosine
            
    #check elder
    file = open('elderclean.txt', encoding='latin-1').read()
    clean = re.sub(r'[^\x00-\x7F]+',' ', file)
    doc = clean.replace('\n',' ')
    doc = doc.split('\n')
    train_set = doc
    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
    for vector in trainVectorizerArray:
        for testV in testVectorizerArray:
            cosine = cx(vector, testV)
            prob[3] = cosine
            
    maximum = 0
    for i,value in enumerate(prob):
        if value > maximum:
            maximum = value
            index = i
            
    if index == 0:
        result.append('teenager')
    else: 
        if index == 1:
            result.append('young adult')
        else:
            if index == 2:
                result.append('adult')
            else:
                result.append('elder')
                
teenager = 0
young_adult = 0
adult = 0
elder = 0               
for j in range(len(result)):
    if result[j] == 'teenager':
        teenager = teenager + len(Cluster[j])
    else:
        if result[j] == 'young adult':
            young_adult = young_adult + len(Cluster[j])
        else:
            if result[j] == 'adult':
                adult = adult + len(Cluster[j])
            else:
                elder = elder + len(Cluster[j])

              
Hasil = {'teenager' : teenager, 'young adult' : young_adult, 'adult' : adult, 'elder' : elder}
print(Hasil)

end = time.time()

print(end - start)



