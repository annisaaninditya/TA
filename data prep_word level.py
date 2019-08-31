# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:44:44 2019

@author: annisaaninditya
"""


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import nltk
import re


from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import model_selection, naive_bayes, metrics
from sklearn.feature_selection import chi2

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score


###########################################################################################################################
####################################################Data Preparation####################################################
###########################################################################################################################

df = pd.read_csv('pertanyaan_final_sepatarated3.csv')
df.head()


col = ['Label', 'Pertanyaan']
df = df[col]
df = df[pd.notnull(df['Pertanyaan'])]
df.columns = ['Label', 'Pertanyaan']
df['category_id'] = df['Label'].factorize()[0]
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)
df.head()

fig = plt.figure(figsize=(8,6))
df.groupby('Label').Pertanyaan.count().plot.bar(ylim=0)
plt.show()

###########################################################################################################################
####################################################Signal Preprocessing####################################################
###########################################################################################################################

datas = df.Pertanyaan
labels = df.category_id
labelorder = df['Label']


factory = StemmerFactory()
stemmer = factory.create_stemmer()

# first step tokenizing

for line in datas:
    tokenized_sents = [nltk.word_tokenize(i) for i in datas]

    #for i in tokenized_sents:
    #    print(i)

        
# second step stemming    

for line in tokenized_sents:    
    stemming_sents = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in datas]

    #for i in stemming_sents:
    #    print (i)


# third step stopwords
                        
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('indonesian'))

OAGTokensWOStop = []
for line in stemming_sents:
    temp = []
    for tweet in line:  
        if tweet == ",":
            break
        tweet = re.sub(r'\d+', '', tweet)
        if tweet not in stop_words:
            temp.append(tweet)
    OAGTokensWOStop.append(temp)


for line in OAGTokensWOStop:
    OAGTokensWOStopString = [' '.join(line) for line in OAGTokensWOStop]
    
###########################################################################################################################
####################################################Feature Extraction####################################################
###########################################################################################################################
    
trainDF = pd.DataFrame()
trainDF['text'] = OAGTokensWOStopString
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
                 
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf_vect_fit = tfidf_vect.fit(OAGTokensWOStopString)
tfidf = tfidf_vect_fit.transform(OAGTokensWOStopString)
tf = tfidf_vect.fit_transform(OAGTokensWOStopString)

tfidf_norm = normalize(tfidf, norm='l2', axis=1)
xtrain1 =  tfidf_vect_fit.transform(train_x)
xtrain1_norm = normalize(xtrain1, norm='l2', axis=1)
xvalid1 =  tfidf_vect_fit.transform(valid_x)
xvalid1_norm = normalize(xvalid1, norm='l2', axis=1)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,2), max_features=5000)
tfidf_vect_ngram_fit = tfidf_vect_ngram.fit(OAGTokensWOStopString)
tfidfngram = tfidf_vect_ngram_fit.transform(OAGTokensWOStopString)
xtrain2 =  tfidf_vect_ngram.transform(train_x)
xvalid2 =  tfidf_vect_ngram.transform(valid_x)

# normalize ngram
tfidfngram_norm = normalize(tfidfngram, norm='l2', axis=1)
xtrain2_norm = normalize(xtrain2, norm='l2', axis=1)
xvalid2_norm = normalize(xvalid2, norm='l2', axis=1)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars_fit = tfidf_vect_ngram_chars.fit(OAGTokensWOStopString)
tfidfchars = tfidf_vect_ngram_chars_fit.transform(OAGTokensWOStopString)
xtrain3 =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid3 =  tfidf_vect_ngram_chars.transform(valid_x) 

# normalize charas
tfidfngram_norm_chars = normalize(tfidfchars, norm='l2', axis=1)
xtrain3_norm = normalize(xtrain3, norm='l2', axis=1)
xvalid3_norm = normalize(xvalid3, norm='l2', axis=1)

###########################################################################################################################
#################################################### Term Weighting using Count Vector ####################################################
###########################################################################################################################

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
countvect = count_vect.fit_transform(OAGTokensWOStopString)

# transform the training and validation data using count vectorizer object
cv_fit = count_vect.fit(OAGTokensWOStopString)
xtrain4 =  cv_fit.transform(train_x)
xvalid4 =  cv_fit.transform(valid_x)

train_count = cv_fit.transform(OAGTokensWOStopString)

# normalize

countvect_norm = normalize(countvect, norm='l2', axis=1)
xtrain4_norm = normalize(xtrain4, norm='l2', axis=1)
xvalid4_norm = normalize(xvalid4, norm='l2', axis=1)

###########################################################################################################################
#################################################### DataFrame ####################################################
###########################################################################################################################

doc_index = [i for i in OAGTokensWOStopString]
label_order = [i for i in labelorder]

# dataframe tf-idf
feature_names = tfidf_vect.get_feature_names()
df2 = pd.DataFrame(tf.todense(), index=doc_index, columns=feature_names)
df2.insert(0, 'labels', label_order)

# dataframe ngram level tf-idf 
feature_names3 = tfidf_vect_ngram.get_feature_names()
df3 = pd.DataFrame(tfidfngram.todense(), index=doc_index, columns=feature_names3)
df3.insert(0, 'labels', label_order)

# dataframe characters level tf-idf 
feature_names4 = tfidf_vect_ngram_chars.get_feature_names()
df4 = pd.DataFrame(tfidfchars.todense(), index=doc_index, columns=feature_names4)
df4.insert(0, 'labels', label_order)

# dataframe couunt level tf-idf 
feature_names5 = count_vect.get_feature_names()
df5 = pd.DataFrame(train_count.todense(), index=doc_index, columns=feature_names5)
df5.insert(0, 'labels', label_order)


###########################################################################################################################
####################################################Naive Bayes Classification####################################################
###########################################################################################################################

# naive bayes model building 
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

# Naive Bayes on Word Level TF IDF Vectors
    
# Naive Bayes on Word Level TF IDF Vectors
accuracy_word = train_model(naive_bayes.MultinomialNB(), xtrain1_norm, train_y, xvalid1_norm)
print ("NB, WordLevel TF-IDF: ", accuracy_word)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_ngram = train_model(naive_bayes.MultinomialNB(), xtrain2_norm, train_y, xvalid2_norm)
print ("NB, N-Gram Vectors: ", accuracy_ngram)

# Naive Bayes on Character Level TF IDF Vectors
accuracy_chara = train_model(naive_bayes.MultinomialNB(), xtrain3_norm, train_y, xvalid3_norm)
print ("NB, CharLevel Vectors: ", accuracy_chara)

# Naive Bayes on Count Vectorize
accuracy_CV = train_model(naive_bayes.MultinomialNB(), xtrain4_norm, train_y, xvalid4_norm)
print ("NB, Count Vectors: ", accuracy_CV)

#using naive bayes, ngram feature has the best accuracy, 0.58

###########################################################################################################################
####################################################Text Representation####################################################
###########################################################################################################################

features = tfidf_norm
features.shape

N = 3
for OAGTokensWOStopString, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf_vect.get_feature_names())[indices]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  tripletgrams = [v for v in feature_names if len(v.split(' ')) == 5]
  print("# '{}':".format(OAGTokensWOStopString))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  print("  . Most correlated tripletgrams:\n. {}".format('\n. '.join(tripletgrams[-N:])))
  
###########################################################################################################################
####################################################Model Selection####################################################
###########################################################################################################################

# now checking using other classifier

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

###########################################################################################################################
####################################################MultiNomial ####################################################
###########################################################################################################################
CV = 10
model = MultinomialNB()
accuracies2 = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
avg_score = np.mean(accuracies2)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(2,2))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

###########################################################################################################################
####################################################Model Evaluation####################################################
###########################################################################################################################

from IPython.display import display
for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 1:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Label', 'Pertanyaan']])
      print('')


print(metrics.classification_report(y_test, y_pred, target_names=df['Label'].unique()))