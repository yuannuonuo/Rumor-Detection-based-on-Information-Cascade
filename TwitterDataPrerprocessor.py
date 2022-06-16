import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import gensim
import random
import shutil
import TweetProcess
import numpy as np
from gensim.models.doc2vec import Doc2Vec

TRAINDATADIR="./Data_Twitter3/ExperimentalData/Train/"
TESTDATADIR="./Data_Twitter3/ExperimentalData/Test/"
DOCUMENTVECTORDIMENSION=100
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！—【】（）、。：；’‘……￥·"""
dicts = {i: '' for i in punctuation}
punc_table = str.maketrans(dicts)

def MaxMinNormalization(x,Min,Max):
    x = (x - Min) / (Max - Min);
    return x

def loadStopwords(path):
    stopwordsList=[]
    stopwordsFile=open(path,"r",encoding="utf-8")
    stopwordsFileLines=stopwordsFile.readlines()
    for line in stopwordsFileLines:
        stopwordsList.append(line.replace("\n","").replace("\t",""))
    return stopwordsList

def loadFile(filePath):
    file=open(filePath,"r",encoding="utf-8")
    return file

def readDataFromJsonFile(filePath):
    file=loadFile(filePath)
    jsonData=json.load(file)
    return jsonData

def readDataFromDir(path):
    dataList=[]
    for root,dirs,files in os.walk(path):
        for filePath in files:
            completeFilePath=os.path.join(root,filePath)
            jsonData=readDataFromJsonFile(completeFilePath)
            dataList.append(jsonData)
    return dataList

def getCommentsFromJson(jsonData):
    commentsList=jsonData["comments"]
    return commentsList

def getRepostsFromJson(jsonData):
    repostsList=jsonData["reposts"]
    return repostsList

def splitSentence(sentence):
    stop_words = set(stopwords.words('english'))
    sentence=TweetProcess.text_preprocess(sentence)
    newSplitResult=""
    # try:
    word_tokens = word_tokenize(sentence)
    print(word_tokens)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    newSplitResult=" ".join(filtered_sentence)
    # except:
    #     newSplitResult=""
    return newSplitResult

def vertorize1():
    count=0
    allTextFile=open("./Data_Twitter3/alltext.txt", "a+", encoding = "utf-8")
    logFile = open("./Data_Twitter3/logFile.txt", "a+", encoding="utf-8")
    for root,dirs,files in os.walk("./Data_Twitter3/GroupedData/fake_news"):
        for file in files:
            currentFileLines = open(os.path.join(root,file), "r", encoding = "utf-8").readlines()
            for lineCounter in range(len(currentFileLines)):
                line=currentFileLines[lineCounter]
                itemsList=line.split("\t")
                if len(itemsList)>=1:
                    text=itemsList[1].replace("\n","").replace("\t","")
                    if text!="":
                        count+=1
                        allTextFile.write(text+"\n")
                        logFile.write("fake\t"+str(file)+"\t"+str(lineCounter)+"\n")
    for root, dirs, files in os.walk("./Data_Twitter3/GroupedData/real_news"):
        for file in files:
            currentFileLines = open(os.path.join(root, file), "r", encoding="utf-8").readlines()
            for lineCounter in range(len(currentFileLines)):
                line = currentFileLines[lineCounter]
                itemsList = line.split("\t")
                if len(itemsList) >= 1:
                    text = itemsList[1].replace("\n","").replace("\t","").strip()
                    if text != "" and len(text.split(" "))>=2 and text!=" " and text!="  ":
                        count += 1
                        allTextFile.write(text + "\n")
                        logFile.write("real\t" + str(file) + "\t" + str(lineCounter)+"\n")

def vertorize2():
    # 加载数据
    documents = []
    # 使用count当做每个句子的“标签”，标签和每个句子是一一对应的
    count = 0
    allDataFileLines = open("./Data_Twitter3/alltext.txt", mode='r', encoding="utf-8").readlines()
    print(len(allDataFileLines))
    for line in allDataFileLines:
        words = line.split(" ")
        # 这里documents里的每个元素是二元组，具体可以查看函数文档
        documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
        count += 1
        if count % 1000 == 0:
            print('{} sentences has loaded...'.format(count))
    print(len(documents))
    # 模型训练
    model = Doc2Vec(documents, dm=1, vector_size=DOCUMENTVECTORDIMENSION, window=10, min_count=0, workers=4,
                        epochs=15)
    # 保存模型
    model.save('./models_Twitter3/d2v_alllines.model')
    return model

# def

def extractData(path):
    tag=-1
    if "fake" in path:
        tag=0
    elif "real" in path:
        tag=1
    for root,dirs,files in os.walk(path):
        for fileCounter in range(len(files)):
            extractDataFile = None
            if tag==0:
                extractDataFile = open("./Data_Twitter3/ExtractedData/fake_news/" + str(fileCounter+1) + ".txt","a+",encoding="utf-8")
            elif tag==1:
                extractDataFile = open("./Data_Twitter3/ExtractedData/real_news/" + str(fileCounter + 1) + ".txt", "a+", encoding = "utf-8")
            file=open(os.path.join(root,files[fileCounter]),"r",encoding="utf-8")
            print("Extracting file "+str(fileCounter+1)+"("+str(files[fileCounter])+")/"+str(len(files))+"\t"+os.path.join(root,files[fileCounter])+" to target directory!")
            # try:
            fileJsonData=json.load(file)
            text = str(fileJsonData[str(files[fileCounter]).split(".json")[0]]["sourceText"])
            text = splitSentence(text)
            print(text)
            dateStr = str(fileJsonData[str(files[fileCounter]).split(".json")[0]]["reply_text"][0]).split("$$@@##$$@@##")[1]
            try:
                timeStamp = dateStr
                extractDataFile.write(str(tag) + "\tSOURCE" +"\t"+text.replace("\t","")+"\t"+timeStamp+"\n")
                commentsList = fileJsonData[str(files[fileCounter]).split(".json")[0]]["reply_text"]
                repostsList = fileJsonData[str(files[fileCounter]).split(".json")[0]]["rt_text"]
                quotedList = fileJsonData[str(files[fileCounter]).split(".json")[0]]["qtd_text"]
                for comment in commentsList:
                    commentDateStr = str(comment).split("$$@@##$$@@##")[1]
                    commentText = splitSentence(str(comment).split("$$@@##$$@@##")[0])
                    try:
                        commentTimeStamp = commentDateStr
                        if len(commentText) > 0 and commentText != "" and commentText != " ":
                            extractDataFile.write(str(tag) + "\tCOMMENT" + "\t" + commentText.replace("\t", "") + "\t" + commentTimeStamp + "\n")
                    except:
                        continue
                for repost in repostsList:
                    repostDateStr = str(repost).split("$$@@##$$@@##")[1]
                    repostText = splitSentence(str(repost).split("$$@@##$$@@##")[0])
                    try:
                        repostTimeStamp = repostDateStr
                        if len(repostText) > 0 and repostText != "" and repostText != " ":
                            extractDataFile.write(str(tag) + "\tREPOST" + "\t" + repostText.replace("\t", "") + "\t" + repostTimeStamp + "\n")
                    except:
                        continue
                for quote in quotedList:
                    quoteDateStr = str(quote).split("$$@@##$$@@##")[1]
                    quoteText = splitSentence(str(quote).split("$$@@##$$@@##")[0])
                    try:
                        quoteTimeStamp = quoteDateStr
                        if len(quoteText) > 0 and quoteText != "" and quoteText != " ":
                            extractDataFile.write(str(tag) + "\tREPOST" + "\t" + quoteText.replace("\t", "") + "\t" + quoteTimeStamp + "\n")
                    except:
                        continue
            except:
                continue
            # except:
            #     continue

def combineComments(commentsLinelist,tag):
    combinedComments=[]
    if len(commentsLinelist)<=9 and len(commentsLinelist)>0:
        for commentCounter in range(len(commentsLinelist)):
            commentText=str(commentsLinelist[commentCounter].split("\t")[2])
            combinedComments.append(str(tag)+"\t"+commentText+"\n")
        for i in range(9-len(commentsLinelist)):
            combinedComments.append(str(tag)+"\t"+"\n")
    elif len(commentsLinelist)>9:
        commentsDict={}
        maxTimestamp=-99999999999999
        minTimestamp = 99999999999999
        for commentCounter in range(len(commentsLinelist)):
            commentTimestamp = float(str(commentsLinelist[commentCounter].split("\t")[3].replace("\n","")))
            if commentTimestamp>maxTimestamp:
                maxTimestamp=commentTimestamp
            if commentTimestamp<minTimestamp:
                minTimestamp=commentTimestamp
        for commentCounter in range(len(commentsLinelist)):
            commentText = str(commentsLinelist[commentCounter].split("\t")[2])
            commentTimestamp = float(commentsLinelist[commentCounter].split("\t")[3].replace("\n",""))
            commentsDict[commentText]=MaxMinNormalization(commentTimestamp,minTimestamp,maxTimestamp)
        sortedComments = sorted(commentsDict.items(), key=lambda x: x[1], reverse=False)
        newCommentsDict={}
        for i in range(9):
            newCommentsDict[str(i)]=[]
        for commentItem in sortedComments:
            num=int(float(commentItem[1]) // (1 / 9))
            if num>=9:
               num=8
            newCommentsDict[str(num)].append(commentItem[0])
        for key in range(9):
            if len(newCommentsDict[str(key)])>0:
                combinedComments.append(str(tag)+"\t"+" ".join(newCommentsDict[str(key)])+"\n")
            else:
                combinedComments.append(str(tag) + "\t" + "\n")
    return combinedComments



def groupData(path):
    tag = -1
    if "fake" in path:
        tag = 0
    elif "real" in path:
        tag = 1
    for root,dirs,files in os.walk(path):
        for filePath in files:
            extractDataFile = None
            if tag == 0:
                extractDataFile = open("./Data_Twitter3/GroupedData/fake_news/" + str(filePath) , "a+",
                                       encoding="utf-8")
            elif tag == 1:
                extractDataFile = open("./Data_Twitter3/GroupedData/real_news/" + str(filePath) , "a+",
                                       encoding="utf-8")
            file = open(os.path.join(root, filePath), "r", encoding="utf-8")
            fileLines=file.readlines()
            extractDataFile.write(str(tag) + "\t" + fileLines[0].split("\t")[2] + "\n")
            if len(fileLines)<=1:
                for i in range(9):
                    extractDataFile.write(str(tag) + "\t" + "\n")
            else:
                combinedComments=combineComments(fileLines[1:],tag)
                for line in combinedComments:
                    extractDataFile.write(line)


def divideIntoTrainTest():
    posList = [counter for counter in range(757)]
    negList = [counter for counter in range(757)]
    negTrainList = random.sample(negList, 530)
    posTrainList = random.sample(negList, 530)
    negTestList = []
    for counter in range(len(negList)):
        if negList[counter] not in negTrainList:
            negTestList.append(negList[counter])
    posTestList = []
    for counter in range(len(posList)):
        if posList[counter] not in posTrainList:
            posTestList.append(posList[counter])
    for root, dirs, files in os.walk("./Data_Twitter3/GroupedData/real_news/"):
        for fileCounter in range(len(files)):
            # print(files[fileCounter])
            if fileCounter in posTrainList:
                shutil.copyfile("./Data_Twitter3/GroupedData/real_news/" + files[fileCounter],"./Data_Twitter3/ExperimentalData/Train/real "+files[fileCounter])
            elif fileCounter in posTestList:
                shutil.copyfile("./Data_Twitter3/GroupedData/real_news/" + files[fileCounter],"./Data_Twitter3/ExperimentalData/Test/real " + files[fileCounter])
    for root, dirs, files in os.walk("./Data_Twitter3/GroupedData/fake_news/"):
        for fileCounter in range(len(files)):
            # print(files[fileCounter])
            if fileCounter in negTrainList:
                shutil.copyfile("./Data_Twitter3/GroupedData/fake_news/" + files[fileCounter],"./Data_Twitter3/ExperimentalData/Train/fake "+files[fileCounter])
            elif fileCounter in negTestList:
                shutil.copyfile("./Data_Twitter3/GroupedData/fake_news/" + files[fileCounter],"./Data_Twitter3/ExperimentalData/Test/fake " + files[fileCounter])
def divideIntoTestDev():
    negTestList = [counter for counter in range(227)]
    posTestList = [counter for counter in range(227)]
    negDevList = random.sample(negTestList, 76)
    posDevList = random.sample(posTestList, 76)
    for root,dirs,files in os.walk("./Data_Twitter3/ExperimentalData/Test/"):
        for fileCounter in range(len(files[:227])):
            if fileCounter in negDevList:
                shutil.copyfile("./Data_Twitter3/ExperimentalData/Test/"+files[fileCounter],"./Data_Twitter3/ExperimentalData/Dev/"+files[fileCounter])
            else:
                shutil.copyfile("./Data_Twitter3/ExperimentalData/Test/" + files[fileCounter],
                                "./Data_Twitter3/ExperimentalData/Test1/" + files[fileCounter])
        for fileCounter in range(227,454):
            if (fileCounter-227) in posDevList:
                shutil.copyfile("./Data_Twitter3/ExperimentalData/Test/" + files[fileCounter],"./Data_Twitter3/ExperimentalData/Dev/" + files[fileCounter])
            else:
                shutil.copyfile("./Data_Twitter3/ExperimentalData/Test/" + files[fileCounter],
                                "./Data_Twitter3/ExperimentalData/Test1/" + files[fileCounter])


def transformDataIntoXY(dataDir):
    X=[]
    Y=[]
    model = Doc2Vec.load('./models_Twitter2/d2v_alllines.model')
    print(len(model.docvecs))
    logDict={}
    logFileLines=open("./Data_Twitter2/logFile.txt",mode="r",encoding="utf-8").readlines()
    for lineCounter in range(len(logFileLines)):
        logDict[logFileLines[lineCounter].replace("\n","")]=lineCounter
    for root, dirs, files in os.walk(dataDir):
        for fileName in files:
            fileDir=dataDir+fileName
            file=open(fileDir,mode="r",encoding="utf-8")
            fileLines=file.readlines()
            if len(fileLines)>=10:
                flag=int(fileLines[0].split("\t")[0])
                if flag==0:
                    Y.append(np.array([float(1),float(0)]))
                elif flag==1:
                    Y.append(np.array([float(0),float(1)]))
                currentFileVectors=[]
                for lineCounter in range(len(fileLines)):
                    text = fileLines[lineCounter].split("\t")[1].replace("\n", "").replace("\t", "").strip()
                    if text != "" and len(text.split(" ")) >= 2 and text != " " and text != "  " and text!="   ":
                        print(lineCounter,len(fileLines[lineCounter]),repr(text))
                        keyStr=fileName.replace(" ","\t")+"\t"+str(lineCounter)
                        index=logDict[keyStr]
                        currentFileVectors.append(model[index])
                    else:
                        currentFileVectors.append(np.array([float(0.0) for i in range(DOCUMENTVECTORDIMENSION)]))
                X.append(np.array(currentFileVectors))
    return np.array(X),np.array(Y)

def getTestTrainData():
    trainX,trainY=transformDataIntoXY(TRAINDATADIR)
    testX,testY=transformDataIntoXY(TESTDATADIR)
    return trainX,trainY,testX,testY

def analyzeData(dataList):
    commentsNumDict={"0":0,"1-25":0,"26-50":0,"51-75":0,"76-100":0,">100":0}
    largestCommentsNum=-999
    smallestCommentsNum=999
    totalCommentsNum=0
    for jsonData in dataList:
        commentsList=getCommentsFromJson(jsonData)
        commentsNum=len(commentsList)
        if commentsNum>largestCommentsNum:
            largestCommentsNum=commentsNum
        if commentsNum<smallestCommentsNum:
            smallestCommentsNum=commentsNum
        if commentsNum == 0:
            commentsNumDict["0"]+=1
        elif commentsNum >= 1 and commentsNum <= 25:
            commentsNumDict["1-25"]+=1
        elif commentsNum >= 26 and commentsNum <= 50:
            commentsNumDict["26-50"] += 1
        elif commentsNum >= 51 and commentsNum <= 75:
            commentsNumDict["51-75"] += 1
        elif commentsNum >= 76 and commentsNum <= 100:
            commentsNumDict["76-100"] += 1
        elif commentsNum > 100:
            commentsNumDict[">100"] += 1
        totalCommentsNum+=commentsNum
    print("total comments number",totalCommentsNum)
    print("average comments number:",totalCommentsNum/len(dataList))
    print("max comments number:", largestCommentsNum)
    print("min comments number:", smallestCommentsNum)
    print("comments number distribution:",commentsNumDict)


if __name__ == '__main__':
    # extractData("./Data_Twitter/OriginalData/fake_news/")
    # extractData("./Data_Twitter/OriginalData/real_news/")
    divideIntoTestDev()
    # trainX, trainY, testX, testY = getTestTrainData()
    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)
    # divideIntoTrainTest()
    # vertorize2()
    # groupData("./Data_Twitter3/ExtractedData/fake_news")
    # groupData("./Data_Twitter3/ExtractedData/real_news")
    # fakeWeiboPath="./Data/OriginalData/fake_news/"
    # fakeWeiboDataList=readDataFromDir(fakeWeiboPath)
    # realWeiboPath ="./Data/OriginalData/real_news/"
    # realWeiboDataList = readDataFromDir(realWeiboPath)
    # #
    # # # print("fake")
    # # # collectAllText(fakeWeiboDataList)
    # # # print("real")
    # # # collectAllText(realWeiboDataList)
    # #
    # print("fake Weibo",len(fakeWeiboDataList))
    # analyzeData(fakeWeiboDataList)
    # print("real Weibo",len(realWeiboDataList))
    # analyzeData(realWeiboDataList)
