# -*- coding: utf-8 -*- 
from utils import *
from sen_mon import *
from topic_monetary import *
import re
from copy import deepcopy

# documents file
class Document(object):
    def __init__(self,document_dir,divide_word_n,devide_topic_n):
        self.document_dir = document_dir
        self.document_name = self.document_dir.split("/")[-1][:-4]
        self.divide_word_n = divide_word_n # true/false 
        self.devide_topic_n = devide_topic_n # true/false
        self.monetary = Monetary_topic("hirose_san/monetary_hirosesan_sub2_clean.csv",parser,"all")
        self.parse_sen = ParcingSentence(nlp)
        self.imf_view_df = pd.read_csv(os.path.join(dic_dir,"imf_view.csv"))
        self.separate2sentences()
        self.calculate_sentences()
        self.calculate_document()

    def parenthetic_contents(self,string):
        stack = []
        for i, c in enumerate(string):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                start = stack.pop()
                yield (len(stack), string[start + 1: i])

    def convert_sentence(self,lst):
        p = re.compile("-\w*-")
        out = []
        temp = lst.split("(")
        for i in temp:
            temp1 = i.split(" ")
            if len(temp1)>1:
                #if ")" in temp1[1]:
                temp1 = temp1[1].split(")")
                if temp1[0]!="" and not p.search(temp1[0]):
                    out.append(temp1[0])
        return " ".join(out)

    def extract_and_but_or(self,orig_sen):
        sen = deepcopy(orig_sen)
        if sen[-1]!=".":
            sen+="."
        sen_json = json.loads(parser.parse(sen))['sentences'][0]["parsetree"].split(" (ROOT ")[1]
        sen_json = "(ROOT "+sen_json
        temp = list(self.parenthetic_contents(sen_json))
        temp1 = [i[1] for i in temp if i[0]==2]
        temp = [i for i,j in enumerate(temp1) if j=="CC but" or j=="CC or" or j=="CC and"]
        temp.append(len(temp1))
        sen_out = []
        k=0
        for i in temp:
            if i>k and i<=len(temp1):
                temp2 = [self.convert_sentence(j) for j in temp1[k:i]]
                sen_out.append(" ".join(temp2))
            k=i+1
        if orig_sen[-1]!=".":
            sen_out[-1]=sen_out[-1][:-1]
        return sen_out

    def extract_subphrase(self,sen):
        p =re.compile("-\w*-")
        sen_json = json.loads(parser.parse(sen))
        temp = sen_json['sentences'][0]["parsetree"].split(" (ROOT ")[1].split("SBAR")
        sentence_out = []
        for j in temp:
            temp1 = j.split("(")
            out = []
            for i in temp1:
                temp2 = i.split(" ")
                if len(temp2)>1:
                    if ")" in temp2[1]:
                        temp2 = temp2[1].split(")")[0]
                        if (temp2 is not "") and (not p.search(temp2)):
                            out.append(temp2)
            if len(out)>1:
                sentence_out.append(" ".join(out))
        return sentence_out
        
    def separate2sentences(self):
        f = open(self.document_dir)
        self.contents = f.read()
        f.close()
        # pre process
        self.contents = self.contents.replace(u"\xa0", u" ")
        self.contents.replace(u"U.S.",u"United States").replace(u"U.K.",u"United Kingdom")
        self.sentences_dic = {}
        self.total_sentence_num = 0
        sections = self.contents.split(u"\n")
        for i,section in enumerate(sections):
            section = section.strip()
            _sentences = section.split(u". ")
            for j,sentence in enumerate(_sentences):
                sentence = sentence.strip()
                if sentence != u"":
                    _subsen = self.extract_and_but_or(sentence)
                    for k, subsen in enumerate(_subsen):
                        _subsen1 = self.extract_subphrase(subsen)
                        for l, subsen1 in enumerate(_subsen1):
                            if subsen1 != u"":
                                if subsen1[-1]==",":
                                    subsen1 = subsen1[:-1]
                                else:
                                    subsen1 = subsen1
                                temp = subsen1.split(" ")
                                if len(temp)>2:
                                    index = self.document_name + "_"+str(i+1)+"_"+str(j+1)+"_"+str(k+1)+"_"+str(l+1)
                                    self.sentences_dic[index] = subsen1
                                    self.total_sentence_num += 1
                        
    def calculate_sentences(self):
        self.topic_scores_dic = {}
        self.subsentence_detail_dic = {}
        self.score_detail_dic = {}
        self.tense = {}
        self.score_view = {}
        for index,sentence in self.sentences_dic.items():
            temp, sen_adj = self.monetary.dic_clf_out(sentence)
            #temp = self.monetary.get_mon_top(sentence)
            self.topic_scores_dic[index] = temp
            if temp:
                self.parse_sen.parsing(sentence)
                self.subsentence_detail_dic[index] = self.parse_sen.output_tree()
                if self.parse_sen.score>0:
                    self.score_detail_dic[index] = 1
                elif self.parse_sen.score<0:
                    self.score_detail_dic[index] = -1
                else:
                    self.score_detail_dic[index] = 0
                self.tense[index] = self.parse_sen.tense
                temp1 = deepcopy(self.score_detail_dic[index])
                for i,r in self.imf_view_df.iterrows():
                    if r["word"] in sen_adj:
                        temp1 = temp1*r["view"]
                self.score_view[index] = temp1
            else:
                self.subsentence_detail_dic[index] = 0
                self.score_detail_dic[index] =0
                self.tense[index] = 0
                self.score_view[index] = 0
            
    def calculate_document(self):
        self.d_topics = {"monetary_policy":[], "monetary_view_score":[]}
        for i,r in self.topic_scores_dic.items():
            if r:
                self.d_topics["monetary_policy"].append(self.score_detail_dic[i])
                self.d_topics["monetary_view_score"].append(self.score_view[i])

    def get_details(self):
        details_dic = {"index":[],"sentence":[],"subsentence_detail":[],"monetary_policy":[],"monetary_score":[],"monetary_view_score":[],"tense":[]}
            
        for i,r in self.topic_scores_dic.items():
            details_dic["index"].append(i)
            details_dic["sentence"].append(self.sentences_dic[i])
            details_dic["subsentence_detail"].append(self.subsentence_detail_dic[i])
            details_dic["monetary_score"].append(self.score_detail_dic[i])
            details_dic["monetary_view_score"].append(self.score_view[i])
            details_dic["monetary_policy"].append(r)
            details_dic["tense"].append(self.tense[i])
        headers = ["index","sentence","subsentence_detail","monetary_policy","monetary_score", "monetary_view_score","tense"]
        return DataFrame(details_dic).ix[:,headers]
