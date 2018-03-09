# -*- coding: utf-8 -*- 
import os,json,random,pickle,sys
import numpy
from pandas import DataFrame
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from collections import defaultdict
from anytree import Node, RenderTree
from functools import reduce
import json
import corenlp

os.chdir("..")
data_dir = "data"
statement_dir = os.path.join(data_dir,"text_clean")
g8_dir = os.path.join(data_dir,"g8")
monetary_dir = os.path.join(data_dir,"monetary_clean")
dic_dir = "dic"
output_dir = "output"

def is_parallel(relation):
    if relation == "parataxis" or relation[:4]  == "conj" or relation[:5] in ["advcl"]:
    #or ("nsubj" in [child_node.parse_relation[:5]  for child_node in node.children] and node.parse_relation[:5] != "nsubj")
        return True
    else:
        return False
def is_subsentence_root(relation):
    if relation in ["root","parataxis","acl:relcl"] or relation[:4]  == "conj" or relation[:5]  in ["ccomp","advcl"]:
        return True
    else:
        return False

def load_stanford_parser():
    parser = corenlp.StanfordCoreNLP("stanford-corenlp-full-2016-10-31")
    return parser
parser = load_stanford_parser()
