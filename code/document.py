# -*- coding: utf-8 -*- 
from utils import *
from sentence import Sentence
from topic import *

# documents file
class Document(object):
    def __init__(self,document_dir,divide_word_n,devide_topic_n):
        self.document_dir = document_dir
        self.document_name = self.document_dir.split("/")[-1][:-4]
        self.divide_word_n = divide_word_n # true/false 
        self.devide_topic_n = devide_topic_n # true/false
        self.separate2sentences()
        self.calculate_sentences()
        self.calculate_document()
        
    def separate2sentences(self):
        f = open(self.document_dir)
        self.contents = f.read()
        f.close()
        # pre process
        self.contents = self.contents.replace(u"\xa0", u" ")
        self.contents.replace(u"U.S.",u"United States").replace(u"U.K.",u"United Kingdom")
        # break down into sections and then break down into sentence
        self.sentences_dic = {}
        self.total_sentence_num = 0
        sections = self.contents.split(u"\n")
        for i,section in enumerate(sections):
            section = section.strip()
            if section[:6] == u"Voting":#remove voting section
                break
            else:
                _sentences = section.split(u". ")
                for j,sentence in enumerate(_sentences):
                    sentence = sentence.strip()
                    if sentence != u"":
                        index = self.document_name + "_"+str(i+1)+"_"+str(j+1)
                        self.sentences_dic[index] = sentence
                        self.total_sentence_num += 1
                        
    def calculate_sentences(self):
        self.topic_scores_dic = {}
        self.subsentence_detail_dic = {}
        self.score_detail_dic = {}
        self.word_count_dic = {}
        for index,sentence in self.sentences_dic.items():
            s = Sentence(sentence,parser,"all_dic.csv",self.devide_topic_n,self.divide_word_n)
            self.topic_scores_dic[index] = s.topic_score
            self.score_detail_dic[index] = s.model_details()
            self.subsentence_detail_dic[index] = s.subsentence_detail
            
    def calculate_document(self):
        self.d_topics = {}
        for topics in self.topic_scores_dic.values():#per sentence
            for topic,value in topics.items():#per sub sentence
                if topic in self.d_topics:
                    self.d_topics[topic].append(value)
                else:
                    self.d_topics[topic] = [value]

    def get_details(self):
        details_dic = {"index":[],"sentence":[],"subsentence_detail":[],"topics":[],"score_detail":[]}
        _topics = ["fiscal_stance","fiscal_analysis","economic_condition","monetary_policy","other_policies","risk","no_topic","fiscal_stance_dum"]
        for _topic in _topics:
            details_dic[_topic] = []
            
        for index,topics in self.topic_scores_dic.items():
            details_dic["index"].append(index)
            for _topic in _topics:
                if _topic in topics.keys():
                    details_dic[_topic].append(topics[_topic])
                else:
                    details_dic[_topic].append(0)
            details_dic["sentence"].append(self.sentences_dic[index])
            details_dic["subsentence_detail"].append(self.subsentence_detail_dic[index])
            details_dic["score_detail"].append(self.score_detail_dic[index])
            details_dic["topics"].append(topics)
        headers = ["index","sentence","subsentence_detail","score_detail","fiscal_stance","fiscal_analysis","economic_condition","monetary_policy","other_policies","risk","no_topic","fiscal_stance_dum"]
        return DataFrame(details_dic).ix[:,headers]
