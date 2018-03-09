## Topic Classification: Based on the roots of each sentence, including noun (NN), verb (VB), and adjective (JJ)

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
from utils import *
from word import Word
# how many sentences are classified as monetary
class Monetary_topic(object):
	def __init__(self,lst_sen_name,parser, choice="all"):
		self.lst_sen = pd.read_csv(os.path.join(data_dir,lst_sen_name)).dropna(axis=0,how="any")
		self.choice=choice
		self.process()
	def process(self):
		np.random.seed(1453)
		self.get_root_lst()
		self.get_training_valid_set()
		self.monetary_topic_clf()
		self.validation_result()
		self.print_coef_word()
		#self.dictionary_based_verification()

	def get_root_sen(self,sentence):
		out = []
		temp_json = json.loads(parser.parse(sentence))
		temp_parsetree = temp_json["sentences"][0]["parsetree"]
		temp_trees = temp_parsetree.split(" (ROOT ")[0][1:-1].split("] [")
		temp_words = temp_json["sentences"][0]["words"][0][0]
		t1 = [temp_json["sentences"][0]["words"][0][1]["Lemma"],temp_json["sentences"][0]["words"][0][1]["PartOfSpeech"][:2]]
		s_struc=[t1]
		for k in temp_trees:
			k=k.split(" ")
			t1=[k[4].split("=")[1],k[3].split("=")[1][:2]]
			s_struc.append(t1)
		out=[n[0] for n in s_struc if n[1] in ["NN","JJ","VB","MD"]]
		return " ".join(out)

	def get_noun_only(self,sentence):
		out = []
		temp_json = json.loads(parser.parse(sentence))
		temp_parsetree = temp_json["sentences"][0]["parsetree"]
		temp_trees = temp_parsetree.split(" (ROOT ")[0][1:-1].split("] [")
		temp_words = temp_json["sentences"][0]["words"][0][0]
		t1 = [temp_json["sentences"][0]["words"][0][1]["Lemma"],temp_json["sentences"][0]["words"][0][1]["PartOfSpeech"][:2]]
		s_struc=[t1]
		for k in temp_trees:
			k=k.split(" ")
			t1=[k[4].split("=")[1],k[3].split("=")[1][:2]]
			s_struc.append(t1)
		out=[n[0] for n in s_struc if n[1] in ["NN"]]
		return " ".join(out)

	def get_nominal_only(self,sentence):
		out = []
		temp_json = json.loads(parser.parse(sentence))
		temp_parsetree = temp_json["sentences"][0]["parsetree"]
		temp_trees = temp_parsetree.split(" (ROOT ")[0][1:-1].split("] [")
		temp_nominal = temp_json["sentences"][0]["dependencies"]
		nominal_list={}
		for i in temp_nominal:
			nominal_list[i[2]]=i[0]
		#temp_words = temp_json["sentences"][0]["words"][0][0]
		t1 = [temp_json["sentences"][0]["words"][0][1]["Lemma"],temp_json["sentences"][0]["words"][0][0],temp_json["sentences"][0]["words"][0][1]["PartOfSpeech"][:2]]
		s_struc=[t1]
		for k in temp_trees:
			k=k.split(" ")
			t1=[k[4].split("=")[1],k[0].split("=")[1],k[3].split("=")[1][:2]]
			s_struc.append(t1)
		out=[n[0] for n in s_struc if n[2] in ["NN"] and nominal_list[n[1]][:5] in ["nsubj"]]
		return " ".join(out)

	def get_root_lst(self):
		if self.choice=="noun":
			self.lst_sen["sentence_noun"] = self.lst_sen["sentence"].apply(lambda x: self.get_noun_only(x))
		elif self.choice == "nominal":
			self.lst_sen["sentence_nominal"] = self.lst_sen["sentence"].apply(lambda x: self.get_nominal_only(x))
		else:
			self.lst_sen["sentence_root"] = self.lst_sen["sentence"].apply(lambda x: self.get_root_sen(x))
		self.lst_sen.dropna(0,how="any",inplace=True)
		
	def get_training_valid_set(self):
		self.lst_sen["select"]=pd.Series(np.random.uniform(size=self.lst_sen.shape[0]), index=self.lst_sen.index)
		self.lst_sen.sort_values(["select"],inplace=True)
		test_size = int(0.55*self.lst_sen.shape[0])
		self.train_root = self.lst_sen[:test_size]
		self.test_root = self.lst_sen[test_size:]

	def monetary_topic_clf(self):
		vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[])
		if self.choice=="noun":
			train_matrix = vectorizer.fit_transform(self.train_root['sentence_noun'])
		elif self.choice == "nominal":
			train_matrix = vectorizer.fit_transform(self.train_root['sentence_nominal'])
		else:
			train_matrix = vectorizer.fit_transform(self.train_root['sentence_root'])
		self.words = vectorizer.get_feature_names()
		self.clf_dic = {}
		for topic in self.train_root.columns[1:-2]:
			temp_clf = LogisticRegression(class_weight = 'balanced', C=10)
			temp_clf.fit(train_matrix,self.train_root[topic])
			self.clf_dic[topic]=temp_clf

	def validation_result(self):
		vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[],vocabulary=self.words)
		if self.choice=="noun":
			for topic, clf in self.clf_dic.items():
				output = []
				for i in list(self.test_root.sentence_noun):
					features = vectorizer.fit_transform([i])
					y_pred=clf.predict(features)
					output.append(y_pred[0])
				self.test_root[topic+"_pred"]=pd.Series(output,index=self.test_root.index)
				f1_bin = f1_score(self.test_root[topic],self.test_root[topic+"_pred"])
				print(topic + " f1 score")
				print(f1_bin)
		elif self.choice=="nominal":
			for topic, clf in self.clf_dic.items():
				output = []
				for i in list(self.test_root.sentence_nominal):
					features = vectorizer.fit_transform([i])
					y_pred=clf.predict(features)
					output.append(y_pred[0])
				self.test_root[topic+"_pred"]=pd.Series(output,index=self.test_root.index)
				f1_bin = f1_score(self.test_root[topic],self.test_root[topic+"_pred"])
				print(topic + " f1 score")
				print(f1_bin)
		else:
			for topic, clf in self.clf_dic.items():
				output = []
				for i,r in self.test_root.iterrows():
					features = vectorizer.fit_transform([r["sentence_root"]])
					y_pred=clf.predict(features)
					output.append(y_pred[0])
				self.test_root[topic+"_pred"]=pd.Series(output,index=self.test_root.index)
				f1_bin = f1_score(self.test_root[topic],self.test_root[topic+"_pred"])
				print(topic + " f1 score")
				print(f1_bin)
	def coef_word(self,coef, words, n,name,reverse=True):
		tmp = {}
		topword = {"topword":[]}
		for index, imp in enumerate(coef):
			tmp.update({index : imp})
		coef_sort = sorted(tmp.items(), key=lambda x: x[1], reverse=reverse)
		for i in range(n):
			wid = coef_sort[i][0]
			print(words[wid],coef_sort[i][1])
			topword["topword"].append(words[wid])
		topword = pd.DataFrame(topword)
		topword.to_csv(os.path.join(output_dir,name+"_topword.csv"))
	def print_coef_word(self):
		for topic,clf in self.clf_dic.items():
			print("=========",topic)
			self.coef_word(clf.coef_[0],self.words,topic,100)

	def dictionary_based_clf(self, sentence):
		self.top_word = pd.read_csv(os.path.join(data_dir,"top_word.csv"))
		self.view_word = pd.read_csv(os.path.join(data_dir,"view_word.csv"))
		count_mon = 0
		for i,r in self.view_word.iterrows():
			if r.view_word in sentence:
				count_mon +=1
		if count_mon>0:
			for j,k in self.top_word.iterrows():
				if k.top_word in sentence:
					return True
			return False
		else:
			return False

	def dictionary_based_verification(self):
		self.test_root["dictionary_score"] = self.test_root.sentence_root.apply(lambda x: self.dictionary_based_clf(x))
		self.test_root["compare"] = self.test_root.monetary_topic==self.test_root.dictionary_score
		print(pd.crosstab(index = self.test_root.monetary_topic, columns = self.test_root.dictionary_score))
		print("Number of correctly classified sentences")
		print(self.test_root.compare.sum())
		self.test_root.to_csv(os.path.join(output_dir,"test_compare.csv"))
		f1_bin = f1_score(self.test_root.monetary_topic,self.test_root.dictionary_score)
		print("f1 score")
		print(f1_bin)

def main():
	print("Classifying topic using noun only")
	#chk = Monetary_topic("hirose_san/monetary_hirosesan_sub2_clean.csv",parser,"noun")
	#chk = Monetary_topic("fiscal_test.csv",parser,"noun")
	print("Classifying topic using the nominal parts only")
	#chk1 = Monetary_topic("hirose_san/monetary_hirosesan_sub2_clean.csv",parser,"nominal")
	#chk1 = Monetary_topic("fiscal_test.csv",parser,"nominal")
	print("Classifying topic using noun, verb, adjective, and auxiliary")
	chk2 = Monetary_topic("hirose_san/sub2_test.csv",parser,"all")
	#chk2 = Monetary_topic("fiscal_test.csv",parser,"all")
	#print(chk.train_root.head())

if __name__ == '__main__':
    main()