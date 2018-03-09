from anytree import Node, RenderTree
from pycorenlp import StanfordCoreNLP
import pandas as pd
from functools import reduce
from utils import *
nlp = StanfordCoreNLP('http://localhost:9000')
dic_df = pd.read_csv(os.path.join(dic_dir,"all_dic_mon.csv"))
VB_dic = {
    "VB":"",
    "VBD":"past",
    "VBG":"present",
    "VBN":"past",
    "VBP":"present",
    "VBZ":"present"
}

class Word(Node):
    def __init__(self,name,word,_type,tokens):
        super(Word,self).__init__(name)
        self.word = word
        self.parse_relation = _type
        self.get_tokens(tokens)
        self.get_polarity_and_degree()
        #self.update_score()
    def get_tokens(self,tokens):
        self.token = [token for token in tokens if token["index"] == self.name][0]["pos"]
        self.lemma = [token for token in tokens if token["index"] == self.name][0]["lemma"]
    def get_polarity_and_degree(self):
        self.polarity = 0.0
        self.degree = 0.0
        if self.word != "":
            record = dic_df[(dic_df["word"]==self.lemma)&(dic_df["type"]==self.token[:2])]
            if len(record) == 1:
                self.polarity = record["fomc_p"].iloc[0]
                self.degree = record["fomc_d"].iloc[0]
        self.update_polarity = self.polarity
        self.update_degree = self.degree
    def update_score(self):
        if self.lemma == "unwind":
            for child in self.children:
                if child.update_polarity!=0:
                    self.update_polarity *= child.update_polarity
        elif self.update_polarity != 0:#自身に極性値を持ち場合
            for child in self.children:#子ノードで程度を持つ場合
                if child.update_degree != 0:# and child.parse_relation in ['advmod','amod']:
                    self.update_polarity = self.update_polarity * child.update_degree
                if child.lemma=="less":
                    if self.token == "NN":
                        self.update_polarity *= -1
            for child in self.children:#反転する場合
                if child.parse_relation == "neg":
                    self.update_polarity *= -1
                elif  (child.update_polarity != 0) and (child.parse_relation in ['compound', 'nsubj']):#,'dobj']) or child.parse_relation[:4] == "nmod"):
                    #反転時(名詞が名詞にかかる時(compound)または名詞が動詞にかかる(nsubj)場合のみ反転情報は伝搬
                    self.update_polarity *= child.update_polarity

            for child in self.children:#子ノードで極性を持つ場合（:#反転を除く）
                if (child.update_polarity != 0) and (child.parse_relation not in ['compound', 'nsubj']):#,'dobj'] and child.parse_relation[:4] != "nmod"):
                    self.update_polarity += child.update_polarity
            for child in self.children:
                if child.lemma=="where" or child.lemma=="while" or child.lemma=="fundamental" or child.lemma == "fiscal":
                    self.update_polarity=0
                if child.lemma == "gap" or child.lemma == "growth":
                    for c in child.children:
                        if c.lemma == "output" or c.lemma == "potential":
                            self.update_polarity=0
                if child.lemma=="stance":
                    for c in child.children:
                        if c.lemma=="fiscal":
                            self.update_polarity = 0
            for child in self.children:
                if child.lemma == "premature" or child.lemma == "rate" or child.lemma=="too":
                    self.update_polarity *= -1
        else:#自身に極性値を持たない場合
            polarity_not_nsubj = sum([child.update_polarity for child in self.children if child.update_polarity != 0 and child.parse_relation != "nsubj"])
            polarity_nsubj = sum([child.update_polarity for child in self.children if child.update_polarity != 0 and child.parse_relation == "nsubj"])
            self.update_polarity = polarity_not_nsubj * polarity_nsubj if polarity_nsubj != 0 and polarity_not_nsubj != 0 else polarity_not_nsubj + polarity_nsubj #子ノードの極性値合計を自身の極性値とする。
            #程度値
            degree_lists = [child.update_degree for child in self.children if child.update_degree != 0]
            if self.degree != 0:
                degree_lists.append(self.degree)
            if len(degree_lists) != 0:
                if self.update_polarity == 0:#子に極性値を持たない場合は上に伝搬
                    if self.parse_relation[:3] != "acl":
                        self.update_degree = reduce(lambda x,y:x*y,degree_lists)
                else:#子に極性値を持つ場合は、兄弟間でsum(極性値)×product(程度値)
                    self.update_polarity = self.update_polarity * reduce(lambda x,y:x*y,degree_lists)
                    self.update_degree = 0
            for child in self.children:
                if child.lemma=="where" or child.lemma=="while" or child.lemma=="fundamental" or child.lemma=="with":
                    print(self.lemma)
                    self.update_polarity=0
            if self.lemma=="rate":
                self.update_polarity *= -1
            if self.lemma=="complicate":
                self.update_polarity=0
        
class ParcingSentence(object):
    def __init__(self,nlp):
        self.nlp = nlp
    def parsing(self,text):
        self.text = (text)
        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse,lemma','outputFormat': 'json'
        })

        self.deps = output["sentences"][0]["enhancedPlusPlusDependencies"]
        self.tokens = output["sentences"][0]["tokens"]
        self.get_tree()
        self.check_tense()
        self.update_polarity_score()
    def get_tree(self):
        root_dep = [dep for dep in self.deps if dep['dep']=='ROOT'][0]
        self.root_node = Word(root_dep['dependent'],root_dep['dependentGloss'],root_dep['dep'],self.tokens)
        self.get_child(self.root_node)
    def get_child(self,p_node):
        children = [dep for dep in self.deps if (dep['governor']==p_node.name) & (dep["dependent"]!= p_node.name)]
        for child in children:
            node = Word(child['dependent'],child['dependentGloss'],child['dep'],self.tokens)
            node.parent = p_node
            self.get_child(node)
    def update_polarity_score(self):
        all_nodes = [node for _, _, node in RenderTree(self.root_node)]
        max_depth = max([node.depth for node in all_nodes])
        for depth in range(max_depth)[::-1]:#底からたどる
            nodes = [node for node in all_nodes if node.depth == depth]
            for node in nodes:
                if len(node.children) != 0:#子ノードを持つ場合
                    node.update_score()
        self.score = self.root_node.update_polarity

    def check_tense(self):
        #rootの時制をみる
        self.tense = ""
        self.get_tense(self.root_node)
        if self.tense == "":#rootだけで判断できなかった場合について子供を見る
            for child_node in self.root_node.children:
                self.get_tense(child_node)
    def get_tense(self,node):
        if node.token[:2] == "VB":
            self.tense = VB_dic[node.token]
            if self.tense == "":
                self.find_aux(node)
        else:
            self.find_aux(node)
        
    def find_aux(self,p_node):#助動詞をみて判断する
        auxs = [node for node in p_node.children if node.parse_relation == "aux"]
        if len(auxs) != 0:
            if auxs[0].word in ["would","should","will","could","may","might","can"]:
                self.tense = "future"
    def output_tree(self):
        for pre, fill, node in RenderTree(self.root_node):
            print("%s%s,%s,%s,%s,%s,%s,%s" % (pre, node.lemma,node.parse_relation,node.token,node.polarity,node.update_degree,node.update_polarity,node.degree))
            
def main():
    parsing_sentence = ParcingSentence(nlp)
    # the code seems to be unable to parse the following sentences
    text = "Emerging economies still adjusting to the lower commodity price environment should avoid expansionary fiscal policies and focus on fostering longer-term adjustment instead."
    parsing_sentence.parsing(text)
    parsing_sentence.output_tree()
    #parsing_sentence.check_tense()
    print("tense",parsing_sentence.tense)
    print("score",parsing_sentence.score)

def get_tense(sentence):
    try:
        parsing_sentence = ParcingSentence(nlp)
        parsing_sentence.parsing(sentence)
        parsing_sentence.check_tense()
        return parsing_sentence.tense
    except:
        print(sentence)
        return ""
def main2():
    #parsing_sentence = ParcingSentence(nlp)
    df = pd.read_csv("yellen_eco.csv",index_col = 0)
    df["tense"] = df["sentence"].apply(lambda x:get_tense(x))
    df.to_csv("yellen_eco_tense.csv")
if __name__ == "__main__":
    main()
    #main2()