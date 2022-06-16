import pandas as pd

dict={}
firstLine=open("./dataset_WWW_22/features.csv","r",encoding="utf-8").readlines()[0]
for item in firstLine.split(","):
    print(item)
    dict[item]="str"
csvfile=pd.read_csv("./dataset_WWW_22/features.csv",low_memory=False,dtype=dict)
find_qtd=csvfile.loc[csvfile['qtd_tweetid'] == '1357784545291104258']
find_rply=csvfile.loc[csvfile['reply_statusid'] == '1357784545291104258']
find_rt=csvfile.loc[csvfile['rt_tweetid'] == '1357784545291104258']
print(find_qtd["ns_label"])
print(find_rply["ns_label"])
print(find_rt["ns_label"])

# dict={"sourceidAAAAAA":{"qtd":12,"reply":20,"rt":5,"total":37,reliable:0,qtd_text:[],sourceText:}}