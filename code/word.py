# -*- coding: utf-8 -*- 
from utils import *
class Word(Node):
    def __init__(self,name,words,dic_df):
        super(Word,self).__init__(name)
        self.get_word_and_type(words)
        self.get_polarity_and_degree(dic_df)
   
    def get_word_and_type(self,words):
        self.word = ""
        self.type = ""
        for word_row in words:
            if self.name == word_row[0]:
                self.word = word_row[4]
                self.type = word_row[5]

    def get_polarity_and_degree(self,dic_df):
        self.polarity = 0
        self.degree = 0
        if self.word != "":
            record = dic_df[(dic_df["word"]==self.word)&(dic_df["type"]==self.type[:2])]
            if len(record) == 1:
                self.polarity = record["fomc_p"].iloc[0]
                self.degree = record["fomc_d"].iloc[0]
        self.update_polarity = self.polarity
        self.update_degree = self.degree
                
    def update_score(self):
        if self.update_polarity != 0:
            for child in self.children:
                if child.update_degree != 0:
                    self.update_polarity = self.update_polarity * child.update_degree                    
            for child in self.children:
                if child.parse_relation == "neg":
                    self.update_polarity *= -1
                elif  (child.update_polarity != 0) and (child.parse_relation in ['compound', 'nsubj','dobj','acl'] or child.parse_relation[:4] in ["nmod","amod"]):
                    self.update_polarity *= child.update_polarity
            for child in self.children:
                if (child.update_polarity != 0) and (child.parse_relation not in ['compound', 'nsubj','dobj','acl'] and child.parse_relation[:4] not in ["nmod","amod"]):
                    self.update_polarity += child.update_polarity
        else:
            polarity_not_nsubj = sum([child.update_polarity for child in self.children if child.update_polarity != 0 and child.parse_relation[:5] != "nsubj"])
            polarity_nsubj = sum([child.update_polarity for child in self.children if child.update_polarity != 0 and child.parse_relation[:5] == "nsubj"])
            self.update_polarity = polarity_not_nsubj * polarity_nsubj if polarity_nsubj != 0 and polarity_not_nsubj != 0 else polarity_not_nsubj + polarity_nsubj
            degree_lists = [child.update_degree for child in self.children if child.update_degree != 0]
            if self.degree != 0:
                degree_lists.append(self.degree)
            if len(degree_lists) != 0:
                if self.update_polarity == 0:
                    if self.parse_relation[:3] != "acl":
                        self.update_degree = reduce(lambda x,y:x*y,degree_lists)
                else:
                    self.update_polarity = self.update_polarity * reduce(lambda x,y:x*y,degree_lists)
                    self.update_degree = 0
            for child in self.children:
                if child.parse_relation == "neg":
                    self.update_polarity *= -1
        child_word = [child.word for child in self.children]
        if self.word in ["flexible","flexibility","smooth","smoothing","attuned"]:
            if "pace" in child_word:
                self.update_polarity*=-0.25
        if self.word in ["pace"]:
            if "balanced" in child_word or "gradual" in child_word or "measured" in child_word:
                self.update_polarity*=-0.25
        if self.word in ["plan","reform","plans","target"]:
            if "consolidation" in child_word or "entitlement" in child_word or "fiscal" in child_word or "deficit" in child_word:
                self.update_polarity=-1
        if self.word in ["target","targets"]:
            if "fiscal" in child_word:
                self.update_polarity=-1
        if self.word in ["friendly"]:
            if "growth" in child_word:
                self.update_polarity=0.5
        if self.word in ["ensure"]:
            if "to" in child_word:
                self.update_polarity=0.0

    def get_subsentece_root_nodes(self):
        subsentece_root_nodes = []
        for c_node in self.children:
            if is_subsentence_root(c_node.parse_relation):
                subsentece_root_nodes.append(c_node)
            else:
                subsentece_root_nodes = subsentece_root_nodes + c_node.get_subsentece_root_nodes()
        return subsentece_root_nodes
