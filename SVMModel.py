import os
import DataPreprocessor4
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import stochastic_gradient
from sklearn.naive_bayes import MultinomialNB     # 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer


TRAINDATADIR="./ExperimentData/Train/"
TESTDATADIR="./ExperimentData/Test/"

def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)

def transformDataIntoXY(dataDir):
    X=[]
    Y=[]
    for root, dirs, files in os.walk(dataDir):
        for fileName in files:
            fileDir = dataDir + "/" + fileName
            file = open(fileDir, mode="r", encoding="utf-8")
            fileLines = file.readlines()
            flag = int(fileLines[0].split("\t")[1])
            if flag == 0:
                Y.append(0)
            elif flag == 1:
                Y.append(1)
            currentFileLines = ""
            for lineCounter in range(len(fileLines)):
                if fileLines[lineCounter].split("\t")[2] != "\n":
                    currentFileLines+=(fileLines[lineCounter].split("\t")[2]+" ")
            X.append(currentFileLines)
    return X,np.array(Y)

if __name__ == '__main__':
    dataPreprocessor1=DataPreprocessor4.DataPreprocessor4()
    trainX,trainY,testX,testY,devX, devY=dataPreprocessor1.getTrainTestDevData()
    a=[]
    b=[]
    c=[]
    d=[]
    for item in trainX:
        a.append(item.ravel())
    trainX=np.array(a)
    for item in trainY:
        if item[0]==0:
            c.append(1)
        else:
            c.append(0)
    trainY=np.array(c)
    for item in testY:
        if item[0]==0:
            d.append(1)
        else:
            d.append(0)
    testY=np.array(d)
    for item in testX:
        b.append(item.ravel())
    testX=np.array(b)
    clf_rbf = SVC(kernel='rbf',random_state=0)
    clf_rbf.fit(trainX, trainY)
    predictY=clf_rbf.predict(testX)
    # print()
    score_rbf = clf_rbf.score(testX, testY)
    print("The mean acc of svm_rbf is : %f" % score_rbf)
    print(classification_report(predictY,testY,digits=4))

    clf_dtc = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=4,n_estimators=25)
    clf_dtc = clf_dtc.fit(trainX, trainY)
    rfc = rfc.fit(trainX, trainY)
    # 查看模型效果
    score_c = clf_dtc.score(testX, testY)
    score_r = rfc.score(testX, testY)
    predictY_dtc = clf_dtc.predict(testX)
    predictY_rfc = rfc.predict(testX)
    # 打印最后结果
    print("Single Tree:", classification_report(testY,predictY_dtc,digits=4))
    print("The mean acc of DTC:",str(score_c))
    print("Random Forest:"+str(score_r), classification_report(testY,predictY_rfc,digits=4))
    print("The mean acc of RFC:", str(score_r))

    # print('Accuracy score:', accuracy_score(testY, predictY))
    # print('Recall:', recall_score(testY, predictY, average='weighted'))
    # print('F1-score:', f1_score(testY, predictY, average='weighted'))
    # print('Precision score:', precision_score(testY, predictY, average='weighted'))

    # svm_c(trainX,testX,trainY,testY)
    #SVM 0.64~0.65

    # 使用逻辑斯蒂回归
    # lr = LogisticRegression()  # 初始化LogisticRegression
    # lr.fit(trainX, trainY)  # 使用训练集对测试集进行训练
    # lr_y_predit = lr.predict(testX)  # 使用逻辑回归函数对测试集进行预测
    # print('Accuracy of LR Classifier:%f' % lr.score(testX, testY))  # 使得逻辑回归模型自带的评分函数score获得模型在测试集上的准确性结果
    # print(classification_report(testY, lr_y_predit))
    # #0.64

    # trainX, trainY = transformDataIntoXY(TRAINDATADIR)
    # testX, testY = transformDataIntoXY(TESTDATADIR)

    # xData = np.concatenate([trainX, testX])
    # tokenizer = Tokenizer(num_words=15001)
    # tokenizer.fit_on_texts(xData)
    # trainX = tokenizer.texts_to_sequences(trainX)
    # testX = tokenizer.texts_to_sequences(testX)

    # vec = CountVectorizer()
    # trainX = vec.fit_transform(trainX)
    # testX = vec.transform(testX)

    # 3.使用朴素贝叶斯进行训练
    # mnb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
    # mnb.fit(trainX, trainY)  # 利用训练数据对模型参数进行估计
    # y_predict = mnb.predict(testX)  # 对参数进行预测
    #
    # print('The Accuracy of Naive Bayes Classifier is:', mnb.score(testX, testY))
    # print(classification_report(testY, y_predict))




