# -*- coding: utf-8 -*- 
from utils import *
from doc_mon import Document
from sen_mon import *
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class Corpus(object):
    def __init__(self,corpus_dir,divide_word_n,devide_sentence_n,devide_topic_n):
        np.random.seed(342)
        self.corpus_dir = corpus_dir
        self.divide_word_n = divide_word_n # true/false 
        self.devide_sentence_n = devide_sentence_n # true/false
        self.devide_topic_n = devide_topic_n # true/false
        self.documents = os.listdir(self.corpus_dir)
        self.hirose_san_score = pd.read_csv(os.path.join(data_dir,"results_hirose_san.csv"))
        self.hirose_san_score.dropna(0,how="any",inplace=True)
        self.hirose_san_score = self.hirose_san_score.loc[self.hirose_san_score.hirose_san_score!=-9]
        self.calculate_documents()
        self.output_scores("monetary_imf")
        self.output_details("monetary_imf")
        self.accuracy_adjust()

    def calculate_documents(self):
        self.topic_scores = {}
        self.details_df = []
        for document in self.documents:
            d = Document(os.path.join(self.corpus_dir,document),self.divide_word_n,self.devide_topic_n)
            self.topic_scores[d.document_name] = d.d_topics
            self.details_df.append(d.get_details())
    def output_scores(self,output_name):
        output_dic = {"name":[],"monetary_policy":[],"monetary_policy_adj":[],"monetary_view":[],"monetary_view_adj":[]}
        for name,d_topics in self.topic_scores.items():
            output_dic["name"].append(name)
            if len(d_topics["monetary_policy"])>0:
                output_dic["monetary_policy_adj"].append(sum(d_topics["monetary_policy"])/len(d_topics["monetary_policy"]))
                output_dic["monetary_view_adj"].append(sum(d_topics["monetary_view_score"])/len(d_topics["monetary_policy"]))
            else:
                output_dic["monetary_policy_adj"].append(0)
                output_dic["monetary_view_adj"].append(0)
            output_dic["monetary_policy"].append(sum(d_topics["monetary_policy"]))
            output_dic["monetary_view"].append(sum(d_topics["monetary_view_score"]))
        df = DataFrame(output_dic)
        df.set_index('name').to_csv(os.path.join(output_dir,output_name,"topic_score.csv"))
        
    def output_details(self,output_name):
        df = pd.concat(self.details_df)
        df.to_csv(os.path.join(output_dir,"details",output_name,"results.csv"),index=False)

    def accuracy_adjust(self):
        # merge the data to check with hirose-san score
        temp = pd.read_csv(os.path.join(output_dir,"details","monetary_imf","results.csv"))
        self.out_df = pd.merge(temp, self.hirose_san_score, on="index", how="inner")
        self.out_df["select"] = pd.Series(np.random.uniform(size=self.out_df.shape[0]), index=self.out_df.index)
        self.out_df.sort_values(["select"],inplace=True)
        self.out_df["match_score"] = np.where(self.out_df["monetary_score"]==self.out_df["hirose_san_score"],1,0)
        self.out_df["match_view"] = np.where(self.out_df["monetary_view_score"]==self.out_df["hirose_san_score"],1,0)
        test_size = int(0.55*self.out_df.shape[0])
        self.out_train = self.out_df[:test_size]
        self.out_test = self.out_df[test_size:]
        self.out_train.to_csv(os.path.join(output_dir,"details","monetary_imf","out_train.csv"),index=False)
        temp1 = self.out_test.match_score.sum()/len(self.out_test)
        temp2 = self.out_test.match_view.sum()/len(self.out_test)
        print("With just monetary direction only")
        print("Accuracy rate: ", temp1)
        print("With monetary direction and IMF view")
        print("Accuracy rate: ", temp2)

def main():
    divide_word_n = False
    devide_sentence_n = False
    devide_topic_n = False
    c = Corpus(monetary_dir,divide_word_n,devide_sentence_n,devide_topic_n)
    
if __name__ == '__main__':
    main()
