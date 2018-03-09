# -*- coding: utf-8 -*- 
from utils import *
from document import Document

class Corpus(object):
    def __init__(self,corpus_dir,divide_word_n,devide_sentence_n,devide_topic_n):
        self.corpus_dir = corpus_dir
        self.divide_word_n = divide_word_n # true/false 
        self.devide_sentence_n = devide_sentence_n # true/false
        self.devide_topic_n = devide_topic_n # true/false
        self.documents = os.listdir(self.corpus_dir)
        self.calculate_documents()

    def calculate_documents(self):
        self.topic_scores = {}
        self.details_df = []
        for document in self.documents:
            d = Document(os.path.join(self.corpus_dir,document),self.divide_word_n,self.devide_topic_n)
            self.topic_scores[d.document_name] = d.d_topics
            self.details_df.append(d.get_details())
    def output_scores(self,output_name):
        output_dic = {"name":[]}
        _topics = ["fiscal_stance","fiscal_analysis","economic_condition","monetary_policy","other_policies","risk","no_topic","fiscal_stance_dum"]
        for topic in _topics:
            output_dic[topic] = []
        for name,d_topics in self.topic_scores.items():
            output_dic["name"].append(name)
            for topic in _topics:
                if topic in d_topics.keys():
                    if self.devide_sentence_n:
                        output_dic[topic].append(sum(d_topics[topic])/len(d_topics[topic]))
                    else:
                        output_dic[topic].append(sum(d_topics[topic]))
                else:
                    output_dic[topic].append(0)
        df = DataFrame(output_dic)
        df.set_index('name').to_csv(os.path.join(output_dir,output_name,"topic_score.csv"))
        
    def output_weight(self,output_name):
        output_dic = {"name":[]}
        _topics = ["fiscal_stance","fiscal_analysis","economic_condition","monetary_policy","other_policies","risk","no_topic","fiscal_stance_dum"]
        for topic in _topics:
            output_dic[topic] = []
        for name,d_topics in self.topic_scores.items():
            output_dic["name"].append(name)
            total = sum([len(value) for value in d_topics.values()])
            for topic in _topics:
                if topic in d_topics.keys():
                    output_dic[topic].append(len(d_topics[topic])/total*1.0)
                else:
                    output_dic[topic].append(0)
        df = DataFrame(output_dic)
        df.set_index('name').to_csv(os.path.join(output_dir,output_name,"topic_weight.csv"))
    
    def output_total(self,output_name):
        output_dic = {"name":[],"fiscal_stance":[]}
        for name,d_topics in self.topic_scores.items():
            output_dic["name"].append(name)
            if self.devide_sentence_n:
                output_dic["fiscal_stance"].append(sum(d_topics["fiscal_stance"])/len(d_topics["fiscal_stance"]))
            else:
                output_dic["fiscal_stance"].append(sum(d_topics["fiscal_stance"]))
        df = DataFrame(output_dic)
        df.set_index('name').to_csv(os.path.join(output_dir,output_name,"total_score.csv"))
    
    def output_results(self,output_name):
        self.topic_weight_df = self.dic2df(self.topic_weight)
        self.topic_score_df = self.dic2df(self.topic_scores)
        self.total_score_df = self.dic2df(self.total_scores)
        
        self.topic_weight_df.to_csv(os.path.join(output_dir,output_name,"topic_weight.csv"))
        self.topic_score_df.to_csv(os.path.join(output_dir,output_name,"topic_score.csv"))
        self.total_score_df.to_csv(os.path.join(output_dir,output_name,"total_score.csv"))
        
    def dic2df(self,dics):
        dfs = []
        for name,dic in dics.items():
            df = DataFrame(dic,index=[name])
            dfs.append(df)
        return pd.concat(dfs).fillna(0)
        
    def output_details(self,output_name):
        df = pd.concat(self.details_df)
        df.to_csv(os.path.join(output_dir,"details",output_name,"results.csv"),index=False)
        
def main():
    divide_word_n = False
    devide_sentence_n = False
    devide_topic_n = False
    c = Corpus(statement_dir,divide_word_n,devide_sentence_n,devide_topic_n)
    c.output_weight("statement_output_not_divided_w&s&t")
    c.output_scores("statement_output_not_divided_w&s&t")
    c.output_details("statement_output_not_divided_w&s&t")
    c.output_total("statement_output_not_divided_w&s&t")

    divide_word_n = False
    devide_sentence_n = True
    devide_topic_n = False
    c = Corpus(statement_dir,divide_word_n,devide_sentence_n,devide_topic_n)
    c.output_weight("statement_output_not_divided_w&t")
    c.output_scores("statement_output_not_divided_w&t")
    c.output_details("statement_output_not_divided_w&t")
    c.output_total("statement_output_not_divided_w&t")
    
if __name__ == '__main__':
    main()
