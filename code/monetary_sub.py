from utils import *
from word import Word
#from topic import *
#from sentence import *
import re
from copy import deepcopy

class break_subsentence(object):
	# need to make sure each sentence end with a period
	def __init__(self, mon_dir):
		self.mon_txt = pd.read_csv(mon_dir).dropna(axis=0, how="all")

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
	def get_subphrase(self,output_name):
		df_out = {"sentence":[],"subsentences":[]}
		#self.mon_txt["subphrase"]=self.mon_txt["sentence"].apply(lambda x: self.extract_subphrase(x))
		self.mon_txt["subphrase"]=self.mon_txt["sentence"].apply(lambda x: self.extract_and_but_or(x))
		for i,r in self.mon_txt.iterrows():
			for j in r["subphrase"]:
				df_out["sentence"].append(r["sentence"])
				df_out["subsentences"].append(j)
		df_out1 = pd.DataFrame(df_out)
		df_out1.to_csv(os.path.join(output_dir,output_name,"monetary_subsentence1.csv"))
		df_out2 = {"sentence":[],"subsentence1":[],"subsentence2":[]}
		df_out1["subsentence2"] = df_out1["subsentences"].apply(lambda x: self.extract_subphrase(x))
		for i,r in df_out1.iterrows():
			for j in r["subsentence2"]:
				df_out2["sentence"].append(r["sentence"])
				df_out2["subsentence1"].append(r["subsentences"])
				df_out2["subsentence2"].append(j)
		df_out2 = pd.DataFrame(df_out2)
		df_out2.to_csv(os.path.join(output_dir,output_name,"monetary_subsentence2.csv"))
		return df_out
def main():
	chk = break_subsentence(os.path.join(data_dir,"hirose_san/monetary(hirosesan-scored).csv"))
	chk.get_subphrase("details")

if __name__ == '__main__':
    main()