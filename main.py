# -*- coding: utf-8 -*-
# @Time    : 2022/5/22 15:50
# @Author  : Yuan Yue (Acstream)
# @Email   : yuangyue@qq.com
# @File    : main.py

import sys
import DBCN
import DBCN_BiLSTM
import tensorflow as tf
import DataPreprocessor1

if __name__ == '__main__':
    modelName="--dbcnbilstm"
    trainOrTest="--train"
    filePath="--"

    sess = tf.Session()
    dataPreprocessor = DataPreprocessor1.DataPreprocessor1()
    if trainOrTest=="--train":
        if modelName=="--dbcn":
            print("Start Training DBCN!")
            dbcn=DBCN.DBCN_Model()
            if filePath=="--":
                DBCN.DBCN_Model.modelSavingPath="./models/networks/dbcn/"
            else:
                DBCN.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.train(sess,dataPreprocessor)
            print("DBCN Training Completed!")
        elif modelName=="--dbcnbilstm":
            print("Start Training DBCN_BiLSTM!")
            dbcnbilstm=DBCN_BiLSTM.DBCN_BiLSTM()
            if filePath=="--":
                DBCN_BiLSTM.DBCN_BiLSTM.modelSavingPath="./models/networks/gru2/"
                # DBCN.DBCN_Model.modelSavingPath = "./models_Twitter/networks/dbcnbilstm_withoutcomments/"
            else:
                DBCN_BiLSTM.DBCN_BiLSTM.modelSavingPath=filePath
            dbcnbilstm.modelSavingPath=filePath
            dbcnbilstm.train(sess,dataPreprocessor)
            print("DBCN_BiLSTM! Training Completed!")
        # elif modelName=="--odcn":
        #     print("Start Training ODCN!")
        #     odcn = ODCN_Model.ODCN_Model()
        #     if filePath == "--":
        #         ODCN_Model.ODCN_Model.modelSavingPath = "./trained models/new networks/odcn/"
        #     else:
        #         ODCN_Model.ODCN_Model.modelSavingPath = filePath
        #     odcn.modelSavingPath = filePath
        #     odcn.train(sess, dataPreprocessor)
        #     print("ODCN Training Completed!")
    elif trainOrTest=="--test":
        if modelName=="--dbcn":
            print("Start Testing DBCN!")
            dbcn=DBCN.DBCN_Model()
            if filePath=="--":
                DBCN.DBCN_Model.modelSavingPath="./models/networks/dbcn/"
            else:
                DBCN.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.test(sess,dataPreprocessor)
            print("DBCN Test Completed!")
        elif modelName=="--dbcnbilstm":
            print("Start Testing DBCN_BiLSTM!")
            dbcn=DBCN_BiLSTM.DBCN_BiLSTM()
            if filePath=="--":
                # DBCN.DBCN_Model.modelSavingPath="./models_Twitter/networks/dbcnbilstm_withoutcomments/"
                DBCN.DBCN_Model.modelSavingPath = "./models/networks/gru2/"
            else:
                DBCN.DBCN_Model.modelSavingPath=filePath
            dbcn.modelSavingPath=filePath
            dbcn.test(sess,dataPreprocessor)
            print("DBCN_BiLSTM Test Completed!")
        # elif modelName=="--odcn":
        #     print("Start Testing ODCN!")
        #     odcn = ODCN_Model.ODCN_Model()
        #     if filePath == "--":
        #         ODCN_Model.ODCN_Model.modelSavingPath = "./trained models/networks/odcn/"
        #     else:
        #         ODCN_Model.ODCN_Model.modelSavingPath = filePath
        #     odcn.modelSavingPath = filePath
        #     odcn.test(sess, dataPreprocessor)
        #     print("ODCN Test Completed!")