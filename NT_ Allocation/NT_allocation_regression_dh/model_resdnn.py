from samplemaker import SampleMaker as samplemaker
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
import generator as gen
import random
import pandas as pd
import xlrd
import itertools

class Model:
    def __init__(self, User, Nt , modelname = ""):

        self.modelname = modelname
        self.User = User
        self.Nt = Nt
        self.input_dim = User * Nt
        self.Comb = len(np.array(list(itertools.permutations(np.arange(0, Nt, 1), User))))
        self.trainphase = tf.placeholder(tf.bool)
        self.apply_logit()

    def linear(self,X,in_dim ,out_dim,name,active_f = tf.nn.relu):

        with tf.variable_scope(self.modelname+name) as scope:
            W = tf.get_variable(name ='weights',shape=[in_dim,out_dim], initializer=xavier_initializer())
            X = tf.layers.batch_normalization(X,training=self.trainphase )
            B = tf.Variable(tf.zeros(out_dim))
            if(active_f != None):
                return active_f(tf.matmul(X,W)+B)
            else:
                return tf.matmul(X,W)+B


    def apply_logit(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        H0 = self.linear(self.X,self.input_dim,100,'L1')

        H1 = self.linear(H0,100,100,"L2")
        H2 = self.linear(H1,100,100,"L3")+H0

        H3 = self.linear(H2,100,100,"L4")
        H4 = self.linear(H3,100,100,"L5")+H2

        H5 = self.linear(H4,100,100,"L6")
        H6 = self.linear(H5,100,100,"L7")+H4

        #fianl fc
        H = self.linear(H6,100,50,"Last")
        self.output = self.linear(H,50, 1, "output", active_f=None)

        self.logit = self.output
        self.logit_softmax = tf.nn.softmax(self.logit)


def modelTester():
    pass



if __name__ == "__main__":
    modelTester()

'''

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

    with tf.Session() as sess:

        model = Model(User = User, Nt = Nt, Comb = Comb)

        X = tf.placeholder(tf.float32, shape=[None, model.User])
        Y = tf.placeholder(tf.float32, shape=[None, Comb])



        #tf.softmax_croos_entrotpy_with_logits 는 performs a softmax (i.e scaling) on logits
        #trueLabel에는 안해주므로 0~1사이 값으로 변환이 필요
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels= Y, logits = model.logit) )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002,).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(model.logit_softmax,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #========================학습시작=========================
        sess.run(tf.global_variables_initializer())

        for i in range(100000):

            # =======================트레이닝 데이터를 매번 새로 생성한다======================
            sampleSize = 1000

            inputs = np.zeros([sampleSize, User * Nt])
            labels = np.zeros([sampleSize, Comb])

            for j in range(0, sampleSize):
                #data_gen = gen.generator(User, Nt, 1, SNR=random.randrange(5, 31, 2))
                data_gen = gen.generator(User, Nt, 1, SNR=15)
                data_gen.optimal(User, Nt)

                inputs[j] = (data_gen.norm)
                labels[j] = (data_gen.label)

            # =================================================================================

            rLogit, rCost,_ = sess.run([model.logit, cost, optimizer], feed_dict={ model.X: inputs , Y: labels, model.trainphase: True})
            if i % 100 ==0:
                acc = sess.run( accuracy, feed_dict={model.X : inputs, Y: labels, model.trainphase: True})
                print(i, "Cost :", rCost, "Accuracy", acc, "\n")

            if i == 9999:
                acc = sess.run(accuracy, feed_dict={model.X: inputs, Y: labels, model.trainphase: True})
                print(i, "Cost :", rCost, "Accuracy", acc, "\n")
                print("Test Result", acc)
'''