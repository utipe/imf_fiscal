## Topic Classification: Based on the roots of each sentence, including noun (NN), verb (VB), and adjective (JJ)

from collections import defaultdict
from anytree import Node, RenderTree
from functools import reduce
from collections import Counter
from sklearn import metrics
from scipy.stats import sem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from pylab import *
from utils import *
from word import Word
from copy import deepcopy
# how many sentences are classified as monetary
class Monetary_topic(object):
	def __init__(self,lst_sen_name,parser, choice="all"):
		self.lst_sen = pd.read_csv(os.path.join(data_dir,lst_sen_name)).dropna(axis=0,how="any")
		self.choice=choice
		self.process()
	def process(self):
		np.random.seed(1453)
		#self.get_root_lst()
		#self.get_training_valid_set()
		#self.monetary_topic_clf()
		#self.validation_result()
		#self.print_coef_word()
		#self.dictionary_based_verification()
		#self.dictionary_based_trial_error()
		#self.test_clf()

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
		test_size = int(0.7*self.lst_sen.shape[0])
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
		tmp_clf = LogisticRegression(class_weight = 'balanced', C=10)
		tmp_clf.fit(train_matrix,self.train_root["monetary_topic"])
		self.clf_dic["monetary_topic"]=tmp_clf

	def get_mon_top(self,sentence):
		vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',use_idf=False,stop_words=[],vocabulary=self.words)
		sen_adj = self.get_root_sen(sentence)
		for topic, clf in self.clf_dic.items():
			feature = vectorizer.fit_transform(sen_adj)
			y_pred = clf.predict(features)
		return y_pred

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
				out_csv = {"sentence":[],"Hirose_san_score":[],"model_score":[],"sentence_root":[]}
				output = []
				for i,r in self.test_root.iterrows():
					out_csv["sentence"].append(r["sentence"])
					out_csv["sentence_root"].append(r["sentence_root"])
					out_csv["Hirose_san_score"].append(r["monetary_topic"])
					features = vectorizer.fit_transform([r["sentence_root"]])
					y_pred=clf.predict(features)
					output.append(y_pred[0])
					out_csv["model_score"].append(y_pred[0])
				self.test_root[topic+"_pred"]=pd.Series(output,index=self.test_root.index)
				out_csv = pd.DataFrame(out_csv)
				out_csv.to_csv(os.path.join(output_dir,"compare.csv"))
				f1_bin = f1_score(self.test_root[topic],self.test_root[topic+"_pred"])
				print(topic + " f1 score")
				print(f1_bin)
	def coef_word(self,coef, words, n,reverse=True):
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
		topword.to_csv(os.path.join(output_dir,"topword.csv"))
	def print_coef_word(self):
		for topic,clf in self.clf_dic.items():
			print("=========",topic)
			self.coef_word(clf.coef_[0],self.words,100)

	def dictionary_based_clf(self, sentence, num=0,view=0):
		if num == 0:
			self.top_word = pd.read_csv(os.path.join(data_dir,"top_word.csv"))
		else:
			self.top_word = pd.read_csv(os.path.join(data_dir,"top_word.csv"))[:-num]
		if view==0:
			self.view_word = pd.read_csv(os.path.join(data_dir,"view_word.csv"))
		else:
			self.view_word = pd.read_csv(os.path.join(data_dir,"view_word.csv"))[:-view]
		count_mon = 0
		for i,r in self.view_word.iterrows():
			if r.view_word in sentence:
				count_mon +=1
		if count_mon>0:
			for j,k in self.top_word.iterrows():
				if k["top_word"] in sentence:
					return True
			return False
		else:
			return False

	def dic_clf_out(self,sentence):
		sen_process = self.get_root_sen(sentence)
		return self.dic_clf_try(sentence), sen_process

	def dic_clf_try(self,sentence,word_lst_ori=pd.read_csv(os.path.join(data_dir,"top_word.csv"))[:-41],word=None,view=0):
		word_lst = deepcopy(word_lst_ori)
		if word:
			word_lst.loc[len(word_lst)] = word
		if view==0:
			view_word = pd.read_csv(os.path.join(data_dir,"view_word.csv"))
		else:
			view_word = pd.read_csv(os.path.join(data_dir,"view_word.csv"))[:-view]
		count_mon = 0
		for i,r in view_word.iterrows():
			if r.view_word in sentence:
				count_mon +=1
		if count_mon>0:
			for j,k in word_lst.iterrows():
				if k["top_word"] in sentence:
					return True
					#return 1
			return False
			#return 0
		else:
			return False
			#return 0

	def test_clf(self):
		top_word_try = pd.read_csv(os.path.join(data_dir,"top_word.csv"))[:-41]
		print(top_word_try)
		out_list1 = {"no_view_word":[],"f1_score":[],"f1_compare":[]}
		self.train_root["temp_score"] = self.train_root.sentence_root.apply(lambda x: self.dic_clf_try(x,top_word_try,None,0))
		f1_bin_c = f1_score(self.train_root.monetary_topic,self.train_root.temp_score)
		for i in range(0,9):
			print(top_word_try.head())
			self.train_root["dic_score_{0}".format(i)] = self.train_root.sentence_root.apply(lambda x: self.dic_clf_try(x,top_word_try,None,i))
			f1_bin = f1_score(self.train_root.monetary_topic,self.train_root["dic_score_{0}".format(i)])
			out_list1["no_view_word"].append(10-i)
			out_list1["f1_score"].append(f1_bin)
			out_list1["f1_compare"].append(f1_bin_c)
		out_list1=pd.DataFrame(out_list1)
		out_list1.to_csv(os.path.join(output_dir,"out_list1.csv"))
		self.train_root.to_csv(os.path.join(output_dir,"train_out.csv"))

	def dictionary_based_trial_error(self, top_word_try=None, trial_list = None):
		out_list = {"add_top_word":[],"f1_score":[],"pre_score":[],"re_score":[],"f1_compare":[],"pre_compare":[],"re_compare":[]}
		out_list1 = {"no_view_word":[],"f1_score":[],"pre_score":[],"re_score":[]}
		if not top_word_try:
			top_word_try = pd.read_csv(os.path.join(data_dir,"top_word.csv"))[:-41]
		if not trial_list:
			trial_list = pd.read_csv(os.path.join(data_dir,"top_word.csv"))[-41:]
		self.test_root["dic_score"] = self.test_root.sentence_root.apply(lambda x: self.dic_clf_try(x,top_word_try,None,0))
		self.test_root["compare"] = self.test_root.monetary_topic==self.test_root["dic_score"]
		f1_bin_c = f1_score(self.test_root.monetary_topic,self.test_root["dic_score"])
		prec_c = precision_score(self.test_root.monetary_topic,self.test_root["dic_score"])
		recall_c = recall_score(self.test_root.monetary_topic,self.test_root["dic_score"])
		for i,r in trial_list.iterrows():
			self.test_root["dic_score"+r.top_word] = self.test_root.sentence_root.apply(lambda x: self.dic_clf_try(x,top_word_try,r.top_word,0))
			self.test_root["compare"+r.top_word] = self.test_root.monetary_topic==self.test_root["dic_score"+r.top_word]
			f1_bin = f1_score(self.test_root.monetary_topic,self.test_root["dic_score"+r.top_word])
			prec = precision_score(self.test_root.monetary_topic,self.test_root["dic_score"+r.top_word])
			recall = recall_score(self.test_root.monetary_topic,self.test_root["dic_score"+r.top_word])
			out_list["add_top_word"].append(r.top_word)
			out_list["f1_score"].append(f1_bin)
			out_list["pre_score"].append(prec)
			out_list["re_score"].append(recall)
			out_list["f1_compare"].append(f1_bin_c)
			out_list["pre_compare"].append(prec_c)
			out_list["re_compare"].append(recall_c)
		print(top_word_try)
		for i in range(0,9):
			self.test_root["dic_score_{0}".format(i)] = self.test_root.sentence_root.apply(lambda x: self.dic_clf_try(x,top_word_try,None,i))
			self.test_root["compare_{0}".format(i)] = self.test_root.monetary_topic==self.test_root["dic_score_{0}".format(i)]
			f1_bin = f1_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			prec = precision_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			recall = recall_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			out_list1["no_view_word"].append(10-i)
			out_list1["f1_score"].append(f1_bin)
			out_list1["pre_score"].append(prec)
			out_list1["re_score"].append(recall)

		out_list = pd.DataFrame(out_list)
		plt.figure(figsize = (30,20))
		plt.plot(out_list.f1_score)
		plt.plot(out_list.f1_compare)
		plt.ylabel("F1 Score")
		plt.xticks(out_list.index,out_list.add_top_word)
		savefig(os.path.join(output_dir,"add_f1.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list.pre_score)
		plt.plot(out_list.pre_compare)
		plt.ylabel("Precision Score")
		plt.xticks(out_list.index,out_list.add_top_word)
		savefig(os.path.join(output_dir,"add_prec.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list.re_score)
		plt.plot(out_list.re_compare)
		plt.ylabel("Recall Score")
		plt.xticks(out_list.index,out_list.add_top_word)
		savefig(os.path.join(output_dir,"add_rec.png"))
		plt.show()

		out_list1 = pd.DataFrame(out_list1)
		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.f1_score)
		plt.ylabel("F1 Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"add_f1_view.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.pre_score)
		plt.ylabel("Precision Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"add_rec_view.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.re_score)
		plt.ylabel("Recall Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"add_rec_view.png"))
		plt.show()
		out_list.to_csv(os.path.join(output_dir,"out_list.csv"))
		out_list1.to_csv(os.path.join(output_dir,"out_list1.csv"))
		self.test_root.to_csv(os.path.join(output_dir,"test_compare.csv"))

	def dictionary_based_verification(self):
		out_list = {"no_top_word":[],"f1_score":[],"pre_score":[],"re_score":[]}
		out_list1 = {"no_view_word":[],"f1_score":[],"pre_score":[],"re_score":[]}
		for i in range(0,49):
			self.test_root["dic_score_{0}".format(i)] = self.test_root.sentence_root.apply(lambda x: self.dictionary_based_clf(x,i+1,0))
			self.test_root["compare_{0}".format(i)] = self.test_root.monetary_topic==self.test_root["dic_score_{0}".format(i)]
			print(pd.crosstab(index = self.test_root.monetary_topic, columns = self.test_root["dic_score_{0}".format(i)]))
			print("Number of correctly classified sentences")
			print(self.test_root["compare_{0}".format(i)].sum())
			f1_bin = f1_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			prec = precision_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			recall = recall_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			print("f1 score")
			print(f1_bin)
			out_list["no_top_word"].append(50-i)
			out_list["f1_score"].append(f1_bin)
			out_list["pre_score"].append(prec)
			out_list["re_score"].append(recall)
		for i in range(0,8):
			self.test_root["dic_score_{0}".format(i)] = self.test_root.sentence_root.apply(lambda x: self.dictionary_based_clf(x,0,i+1))
			self.test_root["compare_{0}".format(i)] = self.test_root.monetary_topic==self.test_root["dic_score_{0}".format(i)]
			print(pd.crosstab(index = self.test_root.monetary_topic, columns = self.test_root["dic_score_{0}".format(i)]))
			print("Number of correctly classified sentences")
			print(self.test_root["compare_{0}".format(i)].sum())
			f1_bin = f1_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			prec = precision_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			recall = recall_score(self.test_root.monetary_topic,self.test_root["dic_score_{0}".format(i)])
			print("f1 score")
			print(f1_bin)
			out_list1["no_view_word"].append(10-i)
			out_list1["f1_score"].append(f1_bin)
			out_list1["pre_score"].append(prec)
			out_list1["re_score"].append(recall)

		out_list = pd.DataFrame(out_list)
		plt.figure()
		plt.plot(out_list.no_top_word,out_list.f1_score)
		plt.ylabel("F1 Score")
		plt.xlabel("Number of top words used")
		savefig(os.path.join(output_dir,"f1.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list.no_top_word,out_list.pre_score)
		plt.ylabel("Precision Score")
		plt.xlabel("Number of top words used")
		savefig(os.path.join(output_dir,"prec.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list.no_top_word,out_list.re_score)
		plt.ylabel("Recall Score")
		plt.xlabel("Number of top words used")
		savefig(os.path.join(output_dir,"rec.png"))
		plt.show()

		out_list1 = pd.DataFrame(out_list1)
		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.f1_score)
		plt.ylabel("F1 Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"f1_view.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.pre_score)
		plt.ylabel("Precision Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"rec_view.png"))
		plt.show()

		plt.figure()
		plt.plot(out_list1.no_view_word,out_list1.re_score)
		plt.ylabel("Recall Score")
		plt.xlabel("Number of view words used")
		plt.xticks(out_list1.no_view_word)
		savefig(os.path.join(output_dir,"rec_view.png"))
		plt.show()
		out_list.to_csv(os.path.join(output_dir,"out_list.csv"))
		self.test_root.to_csv(os.path.join(output_dir,"test_compare.csv"))

def main():
	print("Classifying topic using noun only")
	#chk = Monetary_topic("hirose_san/sub2_test.csv",parser,"noun")
	#chk = Monetary_topic("fiscal_test.csv",parser,"noun")
	print("Classifying topic using the nominal parts only")
	#chk1 = Monetary_topic("hirose_san/sub2_test.csv",parser,"nominal")
	#chk1 = Monetary_topic("fiscal_test.csv",parser,"nominal")
	print("Classifying topic using noun, verb, adjective, and auxiliary")
	chk2 = Monetary_topic("hirose_san/monetary_hirosesan_sub2_clean.csv",parser,"all")
	chk2.get_mon_top("In Japan , the Bank of Japan should stand ready for further easing.")
	#chk2 = Monetary_topic()
	#chk2 = Monetary_topic("hirose_san/sub2_test.csv",parser,"all")
	#print(chk.train_root.head())
	print(chk2.dic_clf_out("In Japan , the Bank of Japan should stand ready for further easing , preferably by extending purchases under its quantitative and qualitative monetary easing program to assets"))

if __name__ == '__main__':
    main()