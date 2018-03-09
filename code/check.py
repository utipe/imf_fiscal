from utils import *
from word import Word
from topic import *
from sentence import *
class check_hirose_san(object):
	def __init__(self, check_dir):
		self.check_text = pd.read_csv(check_dir).dropna(axis=0, how="any")

	def get_score(self,sen):
		s = Sentence(sen,parser,"all_dic.csv",False,False).topic_score["fiscal_stance"]
		if s>0:
			temp = 1
		elif s<0:
			temp = -1
		else:
			temp = 0
		return temp

	def get_score_doc(self,output_name):
		self.check_text["model_score"] = self.check_text["sentence"].apply(lambda x: self.get_score(x))
		self.check_text["check_Hirosesan"] = self.check_text["model_score"]==self.check_text["Hirose"]
		out = self.check_text.sum(axis=0)["check_Hirosesan"]*100.0/self.check_text.shape[0]
		self.check_text["Accuracy_rate_Hirosesan"]=out
		self.check_text["check"] = self.check_text["model_score"]==self.check_text["fiscal_stance"]
		out1 = self.check_text.sum(axis=0)["check"]*100.0/self.check_text.shape[0]
		self.check_text["Accuracy_rate"]=out1
		# calculate the correct percentage of only positive sentence:
		temp = self.check_text.loc[self.check_text["Hirose"]==1]
		out_temp = temp.sum(axis=0)["check_Hirosesan"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_positive_Hirosesan"]=out_temp
		temp = self.check_text.loc[self.check_text["fiscal_stance"]==1]
		out_temp = temp.sum(axis=0)["check"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_positive"]=out_temp
		# calculate the correct percentage of only negative sentence
		temp = self.check_text.loc[self.check_text["Hirose"]==-1]
		out_temp = temp.sum(axis=0)["check_Hirosesan"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_negative_Hirosesan"]=out_temp
		temp = self.check_text.loc[self.check_text["fiscal_stance"]==-1]
		out_temp = temp.sum(axis=0)["check"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_negative"]=out_temp
		# calculate the correct percentage of 0 sentence
		temp = self.check_text.loc[self.check_text["Hirose"]==0]
		out_temp = temp.sum(axis=0)["check_Hirosesan"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_zero_Hirosesan"]=out_temp
		temp = self.check_text.loc[self.check_text["fiscal_stance"]==0]
		out_temp = temp.sum(axis=0)["check"]*100.0/temp.shape[0]
		self.check_text["Accuracy_rate_zero"]=out_temp

		self.check_text.to_csv(os.path.join(output_dir,output_name,"modelcheck.csv"))
		print("Accuracy rate model vs. Hirose-san:")
		print(out)

def main():
	chk = check_hirose_san(os.path.join(data_dir,"hirose_san/fiscal_sentence_score(hirosesan-check).csv"))
	chk.get_score_doc("details")

if __name__ == '__main__':
    main()