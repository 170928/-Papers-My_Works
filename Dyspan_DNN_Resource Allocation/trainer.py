import tensorflow as tf
from model_dnn import Model
import numpy as np
from generator import Generator
import argparse
import os

class Trainer:
    def __init__(self,model,args,lr_straegy = None):
        self.args = args
        self.num_user = args.num_user
        self.num_nt = args.num_nt
        self.num_comb = args.num_comb
        self.num_samples = args.num_samples
        self.learning_rate = args.learning_rate
        self.max_itr = args.max_itr if args.max_itr != None else 999999999

        # tensors for train
        self.model = model
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_comb])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.model.logit))
        self.lr_tensor = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_tensor).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.model.logit_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        # tensors for summary
        self.loss_hist = tf.placeholder(tf.float32)
        self.summary_loss_hist = tf.summary.merge([
            tf.summary.scalar('loss', self.loss_hist),
        ])

        self.accuray_hist = tf.placeholder(tf.float32)
        self.summary_acc_hist = tf.summary.merge([
            tf.summary.scalar('accuracy', self.accuray_hist),
        ])

        self.value_hist = tf.placeholder(tf.float32)
        self.summary_value_hist = tf.summary.merge([
            tf.summary.scalar('value', self.value_hist),
        ])


        # tensors for etc
        self.saver = tf.train.Saver(max_to_keep=20)


    def learning_rate_strategy_normal(self,cur_itr):
        return self.learning_rate

    def learning_rate_strategy_poly(self,cur_itr):
        a = self.learning_rate*pow((1- min(cur_itr/self.max_itr,1) ),0.9)
        return a


    def train(self,lr_strategy = None):
        self.lr_strategy = self.learning_rate_strategy_normal if lr_strategy == None else lr_strategy
        with tf.Session() as sess:
        #============== main ===========================
        # 1. session init and graph load
        #===============================================
            sess.run(tf.global_variables_initializer())
            try:    self.saver.restore(sess, self.args.boardroot +'-'+ str(self.args.start_itr))
            except: print("not exist : ", self.args.boardroot +'-'+ str(self.args.start_itr))



        #============== main ===========================
        # 2. declare writers for tensorboard
        #===============================================
            writer_opt = tf.summary.FileWriter(os.path.join(self.args.boardroot,"opt"), sess.graph)
            writer_rand = tf.summary.FileWriter(os.path.join(self.args.boardroot,"rand"), sess.graph)
            writer_rand10 = tf.summary.FileWriter(os.path.join(self.args.boardroot,"rand10"), sess.graph)
            writer_greedy = tf.summary.FileWriter(os.path.join(self.args.boardroot,"greedy"), sess.graph)
            writer_dnn = tf.summary.FileWriter(os.path.join(self.args.boardroot,"dnn"), sess.graph)


        #============== main ===========================
        # 3. train loop
        #===============================================
            for itr in range(self.args.start_itr,self.max_itr):
            #===================== train loop =================
            # 3_1. prepare samples,opts,
            #==================================================
                inputs = np.zeros([self.num_samples, self.num_user * self.num_nt])
                labels = np.zeros([self.num_samples, self.num_comb])

                data_gens = []
                for j in range(0, self.num_samples):
                    #data_gen = gen.generator(User, Nt, 1, SNR=random.randrange(5, 31, 2))
                    data_gen = Generator(self.num_user, self.num_nt, 1, SNR=15)
                    data_gen.optimal(self.num_user, self.num_nt)
                    data_gens.append(data_gen)

                    inputs[j] = (data_gen.norm )
                    labels[j] = (data_gen.label)



            #===================== train loop =================
            # 3_2. print and write logs
            #==================================================
                lr = self.lr_strategy(itr)
                rLogit, rCost, _,acc = sess.run([self.model.logit, self.cost, self.optimizer, self.accuracy],
                                            feed_dict={self.model.X: inputs, self.Y: labels, self.model.trainphase: True,
                                                       self.lr_tensor: lr})

                optvals = np.array([data_gens[i].sumrateSet[np.argmax(labels[i])] for i in range(self.num_samples)])
                randvals = np.array([data_gens[i].getRandOptVal()     for i in range(self.num_samples)])
                rand10vals = np.array([data_gens[i].getRandOptVal(10) for i in range(self.num_samples)])
                greedyvals = np.array([data_gens[i].getGreedyOptVal() for i in range(self.num_samples)])
                dnnvals = np.array([data_gens[i].sumrateSet[np.argmax(rLogit[i])] for i in range(self.num_samples)])

                print(itr, "Cost :", rCost, "\tAccuracy :", round(acc,6), "\tcur_lr:",lr)
                print(itr, "optvals[0]   :",round(optvals[0],6),    "\t[mean]optvals   :",round(np.mean(optvals),6))
                print(itr, "randvals[0]  :",round(randvals[0], 6),  "\t[mean]randvals  :",round(np.mean(randvals), 6))
                print(itr, "rand10vals[0]:",round(rand10vals[0],6), "\t[mean]rand10vals:",round(np.mean(rand10vals),6))
                #print(itr, "greedyvals[0]:",round(greedyvals[0],6), "\t[mean]greedyvals:",round(np.mean(greedyvals),6))
                print(itr, "dnnvals[0]   :",round(dnnvals[0],6),    "\t[mean]dnnvals   :",round(np.mean(dnnvals),6))
                print()

                writer_dnn.add_summary(sess.run(self.summary_loss_hist,feed_dict={self.loss_hist:rCost}),itr)
                writer_dnn.add_summary(sess.run(self.summary_acc_hist,feed_dict={self.accuray_hist:acc}),itr)
                writer_opt.add_summary(sess.run(self.summary_value_hist, feed_dict={self.value_hist: np.mean(optvals)}), itr)
                writer_rand.add_summary(sess.run(self.summary_value_hist, feed_dict={self.value_hist: np.mean(randvals)}), itr)
                writer_rand10.add_summary(sess.run(self.summary_value_hist, feed_dict={self.value_hist: np.mean(rand10vals)}), itr)
                writer_greedy.add_summary(sess.run(self.summary_value_hist, feed_dict={self.value_hist: np.mean(greedyvals)}), itr)
                writer_dnn.add_summary(sess.run(self.summary_value_hist, feed_dict={self.value_hist: np.mean(dnnvals)}), itr)

            #===================== train loop =================
            # 3_3. save
            #==================================================
                if (itr % 10000 == 0 and itr != 0):
                    self.saver.save(sess, self.args.saveroot, global_step=itr)


            #===================== train loop =================
            # 3_4. clean up
            #==================================================
                itr += 1





def Trainer_tester():
    pass
if __name__ == "__main__":
    Trainer_tester()
    pass



'''

    # ===================== main =======================
    # 1. parse args
    # ==================================================
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_user", default=3)
    parser.add_argument("--num_nt", default=5)
    parser.add_argument("--num_comb",default=60)
    parser.add_argument("--num_samples", default=1000)
    parser.add_argument("--num_models", default=10)
    parser.add_argument("--start_itr", default=0)
    parser.add_argument("--max_itr",default=100000)
    parser.add_argument("--learning_rate", default=0.0002)
    parser.add_argument("--saveroot",default='default')

    args = parser.parse_args()
    if args.num_user > args.num_nt:
        raise AssertionError("num user must be less than or equal to num antenna")

    model = Model(User=args.num_user, Nt=args.num_nt, Comb=args.num_comb)
    trainer = Trainer(model = model,args = args)
    trainer.train(trainer.learning_rate_strategy_poly())
'''

'''
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
'''

'''
        writer_value_rand = tf.summary.FileWriter("./board/ensemble2/rand", sess.graph)
        writer_value_rand10 = tf.summary.FileWriter("./board/ensemble2/rand10", sess.graph)
        writer_value_greedy = tf.summary.FileWriter("./board/ensemble2/greedy", sess.graph)
        writers_value_ensed_dnn = [tf.summary.FileWriter("./board/ensemble2/dnn_ens" + str(i), sess.graph) for i in
                                   range(self.num_models)]
        writers_for_each_model = [tf.summary.FileWriter("./board/ensemble2/model" + str(i), sess.graph) for i in
                                  range(self.num_models)]
        '''


'''
        self.value = tf.placeholder(tf.float32)
        self.opt_value_diff = tf.placeholder(tf.float32)
        self.hist_for_ensembled = [
            tf.summary.scalar('value_ens', self.value),
            tf.summary.scalar('opt-value_ens', self.opt_value_diff), ]
        self.merged_ensemble = tf.summary.merge(self.hist_for_ensembled)
        self.accuracy_ensed = tf.placeholder(tf.float32)
        self.merged_ensemble_accuracy = tf.summary.scalar('ens_accuracy', self.accuracy_ensed)
        '''

'''
for multi summary
dict = {}
dict.update({self.loss_hist: rCost})
summary = sess.run(self.summary_loss_hist, feed_dict={self.loss_hist: rCost})
writer.add_summary(summary, itr)
'''