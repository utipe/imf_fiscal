## Topic Classification: Based on the roots of each sentence, including noun (NN), verb (VB), and adjective (JJ)
## Training set: 300 sentences
## Testing set: 300 sentences
## Use Logistic Regression to classify sentence into relevant topics
## 6 topics: fiscal_stance, fiscal_analysis, economic_condition, monetary_policy, other_policies, and risk
#source py3env/bin/activate

# -*- coding: utf-8 -*- 
import os,json,random,pickle,sys
import numpy
from pandas import DataFrame
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from anytree import Node, RenderTree
from functools import reduce
from collections import Counter
from sklearn import metrics
from scipy.stats import sem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import f1_score
from pylab import *
import matplotlib.pyplot as plt
import json
import corenlp
from utils import*

data_dir = "data"
statement_dir = os.path.join(data_dir,"text_clean")
dic_dir = "dic"
output_dir = "output"

sentence_root=pd.read_csv(os.path.join(data_dir,"sentence_root.csv"), index_col=0)
select=pd.Series(np.random.uniform(size=sentence_root.shape[0]), index=sentence_root.index)
sentence_root["select"]=select
sentence_root.dropna(axis=0, how='any',inplace=True)
sentence_root.sort_values(["select"],inplace=True)

train_root = sentence_root
test_root = sentence_root[-300:]

# train on root of the sentence
print("Classifying using sentence root")
vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[])
train_matrix = vectorizer.fit_transform(train_root['sentence'])
words = vectorizer.get_feature_names()

clf_dic = {}
for topic in train_root.columns[1:-1]:
    clf = LogisticRegression(class_weight = 'balanced', C=10)
    clf.fit(train_matrix,train_root[topic])
    clf_dic[topic]=clf

vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[],vocabulary=words)
for topic, clf in clf_dic.items():
	output = []
	for i in list(test_root.sentence):
		features = vectorizer.fit_transform([i])
		y_pred=clf.predict(features)
		output.append(y_pred[0])
	test_root[topic+"_pred"]=pd.Series(output,index=test_root.index)
	f1_bin = f1_score(test_root[topic],test_root[topic+"_pred"])
	print(topic + " f1 score")
	print(f1_bin)

def coef_word(coef, words, n,reverse=True):
	tmp = {}
	for index, imp in enumerate(coef):
		tmp.update({index : imp})
	coef_sort = sorted(tmp.items(), key=lambda x: x[1], reverse=reverse)
    
	for i in range(n):
		wid = coef_sort[i][0]
		print(words[wid],coef_sort[i][1])

for topic,clf in clf_dic.items():
	print("=========",topic)
	coef_word(clf.coef_[0],words,10)

def get_topics_sub(word_text):
	vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[],vocabulary=words)
	features = vectorizer.fit_transform([word_text])
	topics = []
	for topic,clf in clf_dic.items():
		y_pred=clf.predict(features)
		if y_pred[0]:
			topics.append(topic)
	return topics

def extract_root(word_text):
	temp = json.loads(parser.parse(word_text))
	parsetree=temp["sentences"][0]["parsetree"]
	trees = parsetree.split(" (ROOT ")[0][1:-1].split("] [")
	word = temp["sentences"][0]["words"][0][0]
	t1 = [temp["sentences"][0]["words"][0][1]["Lemma"],temp["sentences"][0]["words"][0][1]["PartOfSpeech"][:2]]
	s_struc=[t1]
	for k in trees:
		k=k.split(" ")
		t1=[k[4].split("=")[1],k[3].split("=")[1][:2]]
		s_struc.append(t1)
	out=[n[0] for n in s_struc if n[1] in ["NN","JJ","VB"]]
	return " ".join(out)
