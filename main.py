# -*- coding: utf-8 -*-
# @Time    : 2022/5/22 15:50
# @Author  : Yuan Yue (Acstream)
# @Email   : yuangyue@qq.com
# @File    : main.py

import sys
import DBCN
import BiLSTM
import DBCN_BiLSTM
import tensorflow as tf
import DataPreprocessor1
import DataPreprocessor11
import DataPreprocessor2
import DataPreprocessor3
import DataPreprocessor4
import DataPreprocessor41
import DataPreprocessor5
import DataPreprocessor6

if __name__ == '__main__':
    dataset = str(sys.argv[1])
    combination = str(sys.argv[2])
    modelName = str(sys.argv[3])
    trainOrTest = str(sys.argv[4])
    filePath = str(sys.argv[5])

    dataPreprocessor=None
    testModelSavingPath=""

    if dataset=="--twitter":
        if combination=="--sourceposts":
            dataPreprocessor = DataPreprocessor1.DataPreprocessor1()
            testModelSavingPath = "./models/networks/"
        elif combination=="--sourceposts-replies":
            dataPreprocessor = DataPreprocessor11.DataPreprocessor11()
            testModelSavingPath = "./models/networks/"
        elif combination == "--sourceposts-reposts":
            dataPreprocessor = DataPreprocessor2.DataPreprocessor2()
            testModelSavingPath = "./models2/networks/"
        elif combination == "--sourceposts-replies-reposts":
            dataPreprocessor = DataPreprocessor3.DataPreprocessor3()
            testModelSavingPath = "./models3/networks/"
    elif dataset=="--weibo":
        if combination == "--sourceposts":
            dataPreprocessor = DataPreprocessor4.DataPreprocessor4()
            testModelSavingPath = "./models4/networks/"
        elif combination == "--sourceposts-replies":
            dataPreprocessor = DataPreprocessor41.DataPreprocessor41()
            testModelSavingPath = "./models4/networks/"
        elif combination == "--sourceposts-reposts":
            dataPreprocessor = DataPreprocessor5.DataPreprocessor5()
            testModelSavingPath = "./models5/networks/"
        elif combination == "--sourceposts-replies-reposts":
            dataPreprocessor = DataPreprocessor6.DataPreprocessor6()
            testModelSavingPath = "./models6/networks/"

    sess = tf.Session()
    if trainOrTest=="--train":
        if modelName=="--dbcn":
            print("Start Training the Second Ablation Model-CSRD Without the BiLSTM Part!")
            dbcn=DBCN.DBCN_Model()
            if filePath=="--":
                DBCN.DBCN_Model.modelSavingPath="./models/networks/dbcn/"
            else:
                DBCN.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.train(sess,dataPreprocessor)
            print("Second Ablation Model Training Completed!")
        elif modelName=="--dbcnbilstm":
            print("Start Training CSRD!")
            dbcnbilstm=DBCN_BiLSTM.DBCN_BiLSTM()
            if filePath=="--":
                DBCN_BiLSTM.DBCN_BiLSTM.modelSavingPath="./models/networks/dbcnbilstm/"
            else:
                DBCN_BiLSTM.DBCN_BiLSTM.modelSavingPath=filePath
            dbcnbilstm.modelSavingPath=filePath
            dbcnbilstm.train(sess,dataPreprocessor)
            print("CSRD Training Completed!")
        elif modelName == "--bilstm":
            print("Start Training the First Ablation Model-CSRD Without the Dilated Convolution Part!")
            dbcnbilstm = BiLSTM.BiLSTM()
            if filePath == "--":
                BiLSTM.BiLSTM.modelSavingPath = "./models/networks/bilstm/"
            else:
                BiLSTM.BiLSTM.modelSavingPath = filePath
            dbcnbilstm.modelSavingPath = filePath
            dbcnbilstm.train(sess, dataPreprocessor)
            print("First Ablation Model Training Completed!")
       elif modelName == "--dbcnbilstmatt":
            print("Start Training Attention-weight Model for CSRD!")
            dbcnbilstm = DBCN_BiLSTM_Att.DBCN_BiLSTM_Att()
            if filePath == "--":
                BiLSTM.BiLSTM.modelSavingPath = "./models/networks/dbcnbilstmatt/"
            else:
                BiLSTM.BiLSTM.modelSavingPath = filePath
            dbcnbilstm.modelSavingPath = filePath
            dbcnbilstm.train(sess, dataPreprocessor)
            print("Attention-weight Model Training Completed!")
    elif trainOrTest=="--test":
        if modelName=="--dbcn":
            print("Start Testing the Second Ablation Model-CSRD Without the BiLSTM Part!")
            dbcn=DBCN.DBCN_Model()
            if combination=="--sourceposts":
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath+"/dbcn_withoutcomments/"
            else:
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath + "/dbcn/"
            dbcn.modelSavingPath=filePath
            dbcn.test(sess,dataPreprocessor)
            print("Second Ablation Model Test Completed!")
        elif modelName=="--dbcnbilstm":
            print("Start Testing CSRD!")
            dbcn = DBCN.DBCN_Model()
            if combination == "--sourceposts":
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath + "/dbcnbilstm_withoutcomments/"
            else:
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath + "/dbcnbilstm/"
            dbcn.modelSavingPath = filePath
            dbcn.test(sess, dataPreprocessor)
            print("CSRD Test Completed!")
        elif modelName == "--bilstm":
            print("Start Testing the First Ablation Model-CSRD Without the Dilated Convolution Part!")
            dbcn = DBCN.DBCN_Model()
            if combination == "--sourceposts":
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath + "/bilstm_withoutcomments/"
            else:
                DBCN.DBCN_Model.modelSavingPath = testModelSavingPath + "/bilstm/"
            dbcn.modelSavingPath = filePath
            dbcn.test(sess, dataPreprocessor)
            print("First Ablation Model Test Completed!")
