import DataPreprocessor1
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding,Dense, Conv1D,Flatten,GlobalMaxPooling1D,Input,Lambda,Dropout,BatchNormalization
# import KMaxpooling
import math

SHUFFLE_SIZE=20
BATCH_SIZE=16
in_training=True

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

if __name__ == '__main__':
    previous_best_accuracy = -9999999

    modelSavingPath = "./models/networks/cami_withoutcomments/"
    dataPreprocessor=DataPreprocessor1.DataPreprocessor1()
    sess = tf.Session()
    K.set_session(sess)
    w1_matrix = init_weights([5, 100, 1, 10], "W1")
    b1_matrix = tf.Variable(tf.constant(0.1, shape=[10, 100]), "b1")
    w2_matrix = init_weights([2, 100, 10, 4], "W2")
    b2_matrix = tf.Variable(tf.constant(0.1, shape=[4, 100]), "b2")
    trainX,trainY,testX,testY=dataPreprocessor.getTrainTestData()
    trainDataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    testDataset = tf.data.Dataset.from_tensor_slices((testX, testY))
    dataset = trainDataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).repeat()
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    iterations = math.ceil(400 / BATCH_SIZE)
    n_epochs=20
    display_step=1

    inputs = tf.placeholder(tf.float32, shape=(None, 10,100), name="inputs")
    print(inputs)
    labels = tf.placeholder(tf.float32, shape=(None, 2), name="labels")
    embedd = tf.expand_dims(inputs, -1)
    embedd_unstack = tf.unstack(embedd, axis=2)
    w1_unstack = tf.unstack(w1_matrix, axis=1)
    b1_unstack = tf.unstack(b1_matrix, axis=1)
    w2_unstack = tf.unstack(w2_matrix, axis=1)
    b2_unstack = tf.unstack(b2_matrix, axis=1)
    convs1 = []
    convs2 = []
    with tf.name_scope("per_dim_conv_k_maxpooling_1"):
        for i in range(len(embedd_unstack)):
            conv1 = tf.nn.leaky_relu(tf.nn.conv1d(value=embedd_unstack[i], filters=w1_unstack[i], stride=1, padding="VALID") + b1_unstack[i])
            conv1 = tf.transpose(conv1, perm=[0, 2, 1])
            values1 = tf.nn.top_k(conv1, 5, sorted=False).values
            values1 = tf.transpose(values1, perm=[0, 2, 1])
            convs1.append(values1)
        convres1 = tf.stack(convs1, axis=2)
        print(convres1)
    input_unstack=tf.unstack(convres1,axis=2)
    with tf.name_scope("per_dim_conv_k_maxpooling_2"):
        for i in range(len(input_unstack)):
            conv2 = tf.nn.leaky_relu(tf.nn.conv1d(value=input_unstack[i], filters=w2_unstack[i], stride=1, padding="VALID") +b2_unstack[i])
            conv2 = tf.transpose(conv2, perm=[0, 2, 1])
            values2 = tf.nn.top_k(conv2, 2, sorted=False).values
            values2 = tf.transpose(values2, perm=[0, 2, 1])
            convs2.append(values2)
        convres2 = tf.stack(convs2, axis=2)
    bn = BatchNormalization()(convres2)
    flatembedd = Flatten()(bn)
    predictions = Dense(2, activation='tanh')(flatembedd)
    print(predictions)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))
    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', acc)
    # merged_summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)
    train_optim = tf.train.AdamOptimizer().minimize(loss)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_optim = tf.train.AdamOptimizer().minimize(loss)



    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        # for epoch in range(n_epochs):
        #     for iteration in range(iterations):
        #         x_batch, y_batch = sess.run(next_iterator)
        #         sess.run([train_optim], feed_dict={inputs: x_batch, labels: y_batch})
        #         if (epoch + 1) % display_step == 0:
        #             c, d = sess.run([loss, acc], feed_dict={inputs: x_batch, labels: y_batch})
        #             e, f = sess.run([loss, acc], feed_dict={inputs: testX, labels: testY})
        #             print("Epoch:", '%03d' % (epoch + 1), "loss=", "{:.9f}".format(c), "acc=", "{:.9f}".format(d),"testloss=", "{:.9f}".format(e), "testacc=", "{:.9f}".format(f))
        #             if f > previous_best_accuracy:
        #                 # if testAcc > 0.9 and testAcc<0.93:
        #                 saver.save(sess, modelSavingPath+"cami")
        #                 previous_best_accuracy = f
        #                 print("Saving current model!")

        # build the DBCN model,
        saver = tf.train.Saver()
        # get the saver to load the saved model.
        init = tf.global_variables_initializer()
        sess.run(init)
        # initialize all variables in tensorflow.
        saver.restore(sess, tf.train.latest_checkpoint(modelSavingPath))
        print("Loading model completed!")
        # load the saved model into DBCN.
        graph = tf.get_default_graph()
        # get the graph for accessing the testing results.
        testX, testY, testFileNameNoDict = dataPreprocessor.getTestData()
        # get the testing data.
        feed_dict = {"inputs:0": testX}
        # form the feed dictionary of testing.
        denseOutput = graph.get_tensor_by_name("dense_1/Tanh:0")
        # use graph to access the predictions of the model for testing data.
        testY_Predict = sess.run(denseOutput, feed_dict)
        # testing and get the testing results (predictions).
        predict = sess.run(tf.argmax(testY_Predict, 1))
        # transform 2-D predictions into 1-D.
        real = sess.run(tf.argmax(testY, 1))
        # transform 2-D real labels into 1-D.

        # The following codes count the TP, FN, TN and FP numbers then calculate the accuracy, precision, F1, and recall values.
        TP_List = []
        # List for storing TP data.
        FN_List = []
        # List for storing FN data.
        TN_List = []
        # List for storing TN data.
        FP_List = []
        # List for storing FP data.

        # Traversing the testing results for counting TP, FN, TN and FP
        for i in range(len(predict)):
            if predict[i] == real[i] and predict[i] == 1:
                TP_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] != real[i] and predict[i] == 0 and real[i] == 1:
                FN_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] == real[i] and predict[i] == 0:
                TN_List.append(str(testFileNameNoDict[i]).split("-")[0])
            elif predict[i] != real[i] and predict[i] == 1 and real[i] == 0:
                FP_List.append(str(testFileNameNoDict[i]).split("-")[0])
        # calculate and print the evaluating values.
        Precision_R = len(TP_List) / (len(TP_List) + len(FP_List))
        # Precision Rumor.
        Precision_NR = len(TN_List) / (len(TN_List) + len(FN_List))
        # Precision Non Rumor.
        Recall_R = len(TP_List) / (len(TP_List) + len(FN_List))
        # Recall Rumor.
        Recall_NR = len(TN_List) / (len(TN_List) + len(FP_List))
        # Recall Non Rumor.
        F1_R = 2 * Precision_R * Recall_R / (Precision_R + Recall_R)
        # F1 Rumor.
        F1_NR = 2 * Precision_NR * Recall_NR / (Precision_NR + Recall_NR)
        # F1 Non Rumor.
        F1_AVG = (F1_R + F1_NR) / 2
        # F1 Average.
        Accuracy = (len(TP_List) + len(TN_List)) / (len(TP_List) + len(TN_List) + len(FP_List) + len(FN_List))
        # Accuracy.
        print("TP number=" + str(len(TP_List)))
        print("FN number=" + str(len(FN_List)))
        print("TN number=" + str(len(TN_List)))
        print("FP number=" + str(len(FP_List)))
        print("Precision_R=TP/(TP+FP)=" + str("{:.9f}".format(Precision_R)))
        print("Precision_NR=TN/(TN+FN)=" + str("{:.9f}".format(Precision_NR)))
        print("Recall_R=TP/(TP+FN)=" + str("{:.9f}".format(Recall_R)))
        print("Recall_NR=TN/(TN+FP)=" + str("{:.9f}".format(Recall_NR)))
        print("F1_R=2*Precision_R*Recall_R/(Precision_R+Recall_R)=" + str("{:.9f}".format(F1_R)))
        print("F1_NR=2*Precision_NR*Recall_NR/(Precision_NR+Recall_NR)=" + str("{:.9f}".format(F1_NR)))
        print("F1_AVG=(F1_R+F1_NR)/2=" + str("{:.9f}".format(F1_AVG)))
        print("Accuracy=(TP+TN)/(TP+TN+FP+FN)=" + str("{:.9f}".format(Accuracy)))