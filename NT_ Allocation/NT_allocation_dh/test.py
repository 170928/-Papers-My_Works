from optfinder import Optfinder as optfinder
from samplemaker import SampleMaker as samplemaker
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
import generator as gen
import random
import pandas as pd
import xlrd

class Model:

    def __init__(self, User, Nt, Comb , modelname = ""):

        self.modelname = modelname
        self.User = User
        self.Nt = Nt
        self.input_dim = User * Nt
        self.Comb = Comb

        #=================================================================
        self.trainphase = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32,shape=[None,self.input_dim])
        self.logit = self.apply_logit()

        #======logit_softmax가 뉴럴네트워크 최종 아웃풋======
        self.logit_softmax = tf.nn.softmax(self.logit)
        #=================================================================

    def linear(self,X,in_dim ,out_dim,name,active_f = tf.nn.relu):

        with tf.variable_scope(self.modelname+name) as scope:
            W = tf.get_variable(name ='weights',shape=[in_dim,out_dim], initializer=xavier_initializer())
            #X = tf.nn.dropout(X, keep_prob=self.keep_prob)
            X = tf.layers.batch_normalization(X,training=self.trainphase )
            B = tf.Variable(tf.zeros(out_dim))
            if(active_f != None):
                return active_f(tf.matmul(X,W)+B)
            else:
                return tf.matmul(X,W)+B


    def apply_logit(self):
        self.keep_prob = tf.placeholder(tf.float32)
        H = self.linear(self.X,self.input_dim,100,'L1')
        #H = self.linear(H,100,100,"L2")
        #H = self.linear(H,100,100,"L3")
        #H = self.linear(H,100,100,"L4")
        #H = self.linear(H,100,100,"L5")
        #H = self.linear(H,100,100,"L6")
        #H = self.linear(H,100,100,"L7")
        H = self.linear(H,100,50,"Last")
        self.output = self.linear(H,50, self.Comb, "output", active_f=None)
        return self.output


def modelTester():

    User = 3
    Nt =  5
    Comb = 60

    #=====================TrainingSet===========================
    #input = pd.read_excel('input.xlsx')
    #inputs = np.array(input)
    #label = pd.read_excel('comb.xlsx')
    #labels = np.array(label)

    #=======================TestSet=============================
    #tinput = pd.read_excel('input.xlsx')
    #tinputs = np.array(tinput)
    #tlabel = pd.read_excel('comb.xlsx')
    #tlabels = np.array(tlabel)

    saver = tf.train.Saver()
    print("asd")
    with tf.Session() as sess:
        model = Model(User=User, Nt=Nt, Comb=Comb)

        X = tf.placeholder(tf.float32, shape=[None, model.User])
        Y = tf.placeholder(tf.float32, shape=[None, Comb])

        #============================텐서보드 변수 묶어주는 부분=================================
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/dyspan_0513")
        writer.add_graph(sess.graph)  # Show the graph
        #==========================================================================================


        #tf.softmax_croos_entrotpy_with_logits 는 performs a softmax (i.e scaling) on logits
        #trueLabel에는 안해주므로 0~1사이 값으로 변환이 필요
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels= Y, logits = model.logit) )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002,).minimize(cost)
        predict_comb = tf.argmax(model.logit_softmax, 1)
        #==========================tf.argmax(model===) 은 1xNone 개 의 배열이 생성
        correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(model.logit_softmax,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracy_summ = tf.summary.scalar("accuracy", accuracy)

        #========================학습시작=========================
        sess.run(tf.global_variables_initializer())

        for i in range(100000):

            #========================트레이닝 데이터를 매번 새로 생성한다======================
            sampleSize = 1000

            inputs = np.zeros([sampleSize, User * Nt])
            labels = np.zeros([sampleSize, Comb])
            #================Optimal한 sumrate들의 값이 들어있는 행렬 sumrate==================
            sumrate_all = np.zeros([sampleSize, 60])
            sumrate = np.zeros([sampleSize])

            for j in range(0, sampleSize):
                #data_gen = gen.generator(User, Nt, 1, SNR=random.randrange(5, 31, 2))
                data_gen = gen.generator(User, Nt, 1, SNR=15)
                data_gen.optimal(User, Nt)

                sumrate_all[j] = (data_gen.sumrateSet)
                sumrate[j] = (data_gen.sumrateSet[data_gen.comb])
                inputs[j] = (data_gen.norm)
                labels[j] = (data_gen.label)



            summary, _ = sess.run([ merged_summary, optimizer], feed_dict={ model.X: inputs , Y: labels, model.trainphase: True})
            writer.add_summary(summary, global_step=i)

            if i % 100 ==0:

                acc, rCost, pred_comb = sess.run( [accuracy, cost, predict_comb], feed_dict={model.X : inputs, Y: labels, model.trainphase: True})
                print('[', i , ']', "Cost :", rCost, "Accuracy", acc, "\n")
                saver.save(sess, 'dyspan.ckpt')

                # ==================================================================================
                mean_sumrate_sample = np.mean(sumrate)
                tf.summary.scalar("Optimal Sumrate", mean_sumrate_sample)
                pred_sumrate = np.array([sumrate_all[i,pred_comb[i]] for i in range(sampleSize)])
                mean_pred_sumrate = np.mean(pred_sumrate)
                tf.summary.sclar("Predict Sumrate", mean_pred_sumrate)
                # ===================================================================================


if __name__ == "__main__":
    modelTester()
    pass