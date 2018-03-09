from bs4 import BeautifulSoup
import requests,re,os
import textract
os.chdir("..")
DATA_DIR = "data/fiscal_pdf"
SAVE_DIR = "data/fiscal_txt"
pre_link="http://www.imf.org"
target_dir = "http://www.imf.org/external/np/g20/"
g8_link = "http://www.g8.utoronto.ca/summit/index.htm"

# Extract all IMF Staff Note from IMF website
html = requests.get(target_dir)
soup = BeautifulSoup(html.content, "lxml")
link_imf = soup.findAll("a",text=re.compile('IMF Staff Note to G-20'))

def extract_pdf(r,link,datdir,savedir):
    name = link.split("/")[-1].split(".")[0][:6]
    name_format = "20"+name[-2:]+name[:2]+name[2:4]
    name_pdf = name_format + ".pdf"
    name_txt = name_format + ".txt"
    save_dir = os.path.join(datdir,name_pdf)
    save_txt = os.path.join(savedir,name_txt)
    with open(save_dir, 'wb') as f:
        f.write(r.content)
    text = textract.process(save_dir)
    with open(save_txt, 'wb') as f:
        f.write(text)

for i in link_imf:
    if "pdf" in i["href"]:
        if "www" in i["href"]:
            r = requests.get(i["href"], stream=True)
            extract_pdf(r,i["href"],DATA_DIR,SAVE_DIR)
        else:
            link_temp=pre_link + i["href"]
            r = requests.get(link_temp,stream=True)
            extract_pdf(r,i["href"],DATA_DIR,SAVE_DIR)
    else:
        if "www" in i["href"]:
            r = requests.get(i["href"], stream=True)
        else:
            link_temp=pre_link + i["href"]
            r = requests.get(link_temp,stream=True)
        temp=BeautifulSoup(r.content, "lxml")
        link_temp=temp.find(text=re.compile("Read the"))
        link_temp=link_temp.parent.a["href"]
        if "external" in link_temp:
            link_temp=pre_link+link_temp
        else:
            link_temp=target_dir+link_temp
        r = requests.get(link_temp,stream=True)
        extract_pdf(r,link_temp,DATA_DIR,SAVE_DIR)            
