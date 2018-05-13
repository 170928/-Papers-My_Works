import tensorflow as tf
import numpy as np
import argparse
from trainer import Trainer
from model_dnn import Model as Model_dnn
from model_resdnn import Model as Model_resdnn


if __name__ == "__main__":
#===================== main =======================
# 1. parse args
#==================================================
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_user", default=5)
    parser.add_argument("--num_nt", default=5)
    parser.add_argument("--num_samples", default=1000)
    parser.add_argument("--num_models", default=10)
    parser.add_argument("--start_itr", default=0)
    parser.add_argument("--max_itr",default=None) #None to limitless
    parser.add_argument("--learning_rate", default=0.0002)
    parser.add_argument("--saveroot",default='./save/single-5user-normal/')
    parser.add_argument("--boardroot",default='./board/single_5user-normal/')
    args = parser.parse_args()
    if args.num_user > args.num_nt:
        raise AssertionError("num user must be less than or equal to num antenna")

#===================== main =======================
# 2. set confit
#==================================================
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True


#===================== main =======================
# 3. train
#==================================================
    model = Model_dnn(User=args.num_user, Nt=args.num_nt)
    #model = Model_resdnn(User=args.num_user, Nt=args.num_nt, Comb=args.num_comb)

    trainer = Trainer(model=model, args=args)
    trainer.train()
    #trainer.train(lr_strategy=trainer.learning_rate_strategy_poly)
