# -*- coding: utf-8 -*- 
from utils import *
from word import Word
from topic import *

class SubSentence(Node):
    def __init__(self,name,root_word_node):
        super(SubSentence,self).__init__(name)
        self.score = 0
        self.topics = []
        self.words = ""
        self.root_word_node = root_word_node        
    def separate_to_subsentence(self):
        for node in self.root_word_node.get_subsentece_root_nodes():
            subsentence = SubSentence(node.name+"_"+node.parse_relation,node)
            subsentence.parent = self
        for s_node in self.children:
            s_node.root_word_node.parent = None
            s_node.separate_to_subsentence()
        self.all_nodes = [node for pre, fill, node in RenderTree(self.root_word_node)]
    def output_tree_all(self):
        print(self.name)
        print("=======")
        for pre, fill, node in RenderTree(self.root_word_node):
            print(pre+node.name,node.type,node.parse_relation,node.update_polarity,node.update_degree,node.polarity,node.degree)
        for s_node in self.children:
            s_node.output_tree_all()

    def get_subsentence_root(self):
        subroot = [node.word for node in self.all_nodes]
        return " ".join(subroot)
# classify the topic for subsentence
    def topic_classify(self):
        topic_list = self.get_subsentence_root()
        child_top = self.get_children_topics()
        top_list = get_topics_sub(topic_list)
        for top in child_top:
            if top not in top_list:
                top_list.append(top)
        self.topics = top_list

    def update_polarity_score(self,divide_word_n):
        max_depth = max([node.depth for node in self.all_nodes])
        self.topic_classify()
        for depth in range(max_depth)[::-1]:
            nodes = [node for node in self.all_nodes if node.depth == depth]
            for node in nodes:
                if len(node.children) != 0:
                    node.update_score()
        self.score = self.root_word_node.update_polarity
        polarity_words = [node.word for node in self.all_nodes if node.polarity != 0]
        if len(polarity_words) == 1:
            if polarity_words[0] in ["unemployment"]:
                self.score = 0.0
        if divide_word_n:
            self.score = self.score / len(self.all_nodes)
        if "fiscal_stance" not in self.topics:
            self.score = 0.0

    def get_children_score(self):
        children_score = 0
        for child in self.children:
            score = child.score + child.get_children_score()
            if is_parallel(child.root_word_node.parse_relation):
                children_score = children_score + score
            else:
                children_score = children_score + score / 2.0
        return children_score
    def get_children_topics(self):
        topics = []
        for child in self.children:
            children_topics = child.topics if len(child.topics) != 0 else child.get_children_topics()
            for topic in children_topics:
                if topic not in topics:
                    topics.append(topic)
        return topics
    def get_topics_from_non_nsubj_part(self):
        non_nsubj_words = [node.word.lower() for node in self.all_nodes if node.word.lower() not in self.words]
        topics = get_topics_sub(" ".join(non_nsubj_words))
        return topics
    def get_children_topics_from_non_nsubj_part(self):
        topics = []
        for child in self.children:
            children_topics = child.get_topics_from_non_nsubj_part() if len(child.get_topics_from_non_nsubj_part()) != 0 else child.get_topics_from_non_nsubj_part()
            for topic in children_topics:
                if topic not in topics:
                    topics.append(topic)
        return topics
class Sentence(object):
    def __init__(self,sentence,parser,dic_name,divided_topic_num,divide_word_n):
            self.sentence = sentence.lower()
            self.divided_topic_num = divided_topic_num
            self.divide_word_n = divide_word_n
            self.dic_df = pd.read_csv(os.path.join(dic_dir,dic_name))
            self.process()
    def process(self):
        self.parsing_sentence()
        self.make_tree_dic()
        self.extract_words()
        self.make_tree()
        #self.output_tree_all()
        self.separate_to_subsentence()
        self.calculate_subsentence()
        self.calculate_sentence()
        self.get_subsentence_detail()
  
    def parsing_sentence(self):
        self.result_json = json.loads(parser.parse(self.sentence))
        self.dependencies = self.result_json["sentences"][0]['dependencies']
        self.indexeddependencies = self.result_json["sentences"][0]['indexeddependencies']
        self.words = self.result_json["sentences"][0]["words"]
    def make_tree_dic(self):
        parsetree = self.result_json["sentences"][0]["parsetree"]
        trees = parsetree.split(" (ROOT ")[0][1:-1].split("] [")
        self.tree_dic = {}
        for tree in trees:
            try:
                # extract the "text" to "lemma"
                parts = tree.split(" ")[:5]
                _dic = {}
                for part in parts:
                    _part = part.split("=")
                    _dic[_part[0]] = _part[1]
                self.tree_dic[_dic["Text"]] = _dic
            except:
                pass
        for word in self.words:
            self.tree_dic[word[0]] = word[1]
    def extract_words(self):
        self.extracted_words = []
        for parse_index,parse_results in enumerate(self.dependencies):
            # example of 1 member: ['compound', 'conditions', 'Labor']
            word = parse_results[2]
            parse_type = parse_results[0]
            to_word = parse_results[1]
            # {'NamedEntityTag': 'O', 'CharacterOffsetBegin': '0', 'CharacterOffsetEnd': '5', 'PartOfSpeech': 'NN', 'Lemma': 'labor'}
            word_info = self.tree_dic[word]
            lemma = word_info[u'Lemma']
            POS = word_info[u'PartOfSpeech']
            # index ['compound', 'conditions-3', 'Labor-1']
            self.extracted_words.append([self.indexeddependencies[parse_index][2],word,parse_type,to_word,lemma,POS])
    def make_tree(self):
        self.nodes = []
        self.subsentences = []
        self.root = Word("ROOT-0",self.extracted_words,self.dic_df)
        self.root.parse_relation = "base_root"
        self.nodes.append(self.root)
        for parse_index, parse_result in enumerate(self.indexeddependencies):
            node_1 = self.make_node(parse_result[1])
            node_2 = self.make_node(parse_result[2])
            node_2.parse_relation = parse_result[0]
            if node_2.parent == None and node_1.parent != node_2:
                node_2.parent = node_1
            else:
                if node_2.parent == node_1.parent:
                    node_2.parent = node_1
                    #print(parse_result)
    def make_node(self,indexed_word):
        if indexed_word not in [node.name for node in self.nodes]:
            node = Word(indexed_word,self.extracted_words,self.dic_df)
            self.nodes.append(node)
        else:
            node = [node for node in self.nodes if node.name == indexed_word][0]
        return node
    def separate_to_subsentence(self):
        self.subsentence_root = SubSentence("subsentence_root",self.root)
        self.subsentence_root.separate_to_subsentence()
        self.subsentences = [node for pre, fill, node in RenderTree(self.subsentence_root)]
    def calculate_subsentence(self):
        for subsentence in self.subsentences:
            subsentence.update_polarity_score(self.divide_word_n)
            subsentence.topic_classify()

    def calculate_sentence(self):
        self.topic_score = {}
        score = 0.0
        if len(self.subsentence_root.children) > 0:
            score = sum([i.score for i in self.subsentence_root.children])
            root_topics=[]
            for child in self.subsentence_root.children:
                for tpic in child.topics:
                    if tpic not in root_topics:
                        root_topics.append(tpic)
            if score == 0.0 and root_topics == []:
                self.subsentence_root = self.subsentence_root.children[0]
                self.calculate_sentence()
            else:
                if root_topics == []:
                    print("no topic !")
                    print(self.sentence)
                for topic in root_topics:
                    self.topic_score[topic] = score
                for child_subsentence in self.subsentence_root.children:
                    score = child_subsentence.score + child_subsentence.get_children_score()
                    for topic in child_subsentence.topics:
                        if topic in self.topic_score.keys():
                            self.topic_score[topic] = self.topic_score[topic] + score
                        else:
                            self.topic_score[topic] = score
        print(self.topic_score)
        if "fiscal_stance" not in self.topic_score:
            self.topic_score["fiscal_stance"]=0.0
            self.topic_score["fiscal_stance_dum"]=0.0
        else:
            self.topic_score["fiscal_stance_dum"]=1.0

    def model_details(self):
        details = {"polarity":[],"degree":[]}
        for node in self.nodes:
            if node.polarity != 0:
                details["polarity"].append((node.word,node.polarity))
            elif node.degree != 0:
                details["degree"].append((node.word,node.degree))
        return details
    def output_tree_all(self):
        root = [node for node in self.nodes if node.parse_relation == "root"][0]
        for pre, fill, node in RenderTree(root):
            print(pre+node.name,node.type,node.parse_relation,node.update_polarity,node.update_degree,node.polarity,node.degree)
    def output_tree_only_polarity(self):
        print(self.root.update_polarity)
        for pre, fill, node in RenderTree(self.root):
            if node.type != "" or node.name == u"root":
                print(pre+node.word,node.type,node.parse_relation,node.update_polarity,node.update_degree,node.polarity,node.degree)
    def get_subsentence_detail(self):
        self.subsentence_detail = ""
        for pre, fill, node in RenderTree(self.subsentence_root):
            if node != self.subsentence_root:
                if is_parallel(node.root_word_node.parse_relation):
                    self.subsentence_detail = self.subsentence_detail + pre + node.root_word_node.word+" "+str(node.get_children_score())+" "+str(node.score)+" "+",".join(node.topics) +"//"+ "\n"
                else:
                    self.subsentence_detail = self.subsentence_detail + pre + node.root_word_node.word+" "+str(node.get_children_score())+" "+str(node.score)+" "+",".join(node.topics) + "\n"

s1 = "There is a million stars on the sky of which names we have not known."
s = Sentence(s1,parser,"all_dic.csv",False,False)
for k,v in s.tree_dic.items():
    if v["PartOfSpeech"][:2]=="NN" or v["PartOfSpeech"][:2]=="VB":
        print(v["Lemma"])
#print(s.subsentence_detail)
#print(s.topic_score)
#print(s1)
#print(len(s.subsentences))
#s.subsentence_root.output_tree_all()
