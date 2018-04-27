import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import models
import random


import DatasetMaker_cifar
import Model_Loader

import math


# ======================== test_model ==================================================================================
def test_harmonyNet(currentEpoch):
    correct = 0
    total = 0

    number_of_correct_for_category = [0 for k in range(datasetMaker.getNumOfClasses())]
    number_of_category = [0 for k in range(datasetMaker.getNumOfClasses())]
    mylist = [0 for k in range(datasetMaker.getNumOfClasses())]

    number_of_correct_for_division = [0 for k in range( len(divisedList) )]
    number_of_division = [0 for k in range(len(divisedList))]
    mylist2 = [0 for k in range(len(divisedList))]


    for image, label in testset_loader:
        x = Variable(image, volatile=True).cuda()
        y_ = Variable(label).cuda()

        output = harmonyNet.forward(x)
        _, output_index = torch.max(output, 1)


        for k in range(batch_size):
            number_of_category[y_.data[k]] += 1
            number_of_division[output_index.data[k]] +=1
            total += 1

            if y_.data[k] in divisedList[output_index.data[k]] :
                number_of_correct_for_category[y_.data[k]] += 1
                number_of_correct_for_division[output_index.data[k]] +=1
                correct += 1



    for k in range(datasetMaker.getNumOfClasses()):   mylist[k] = round(number_of_correct_for_category[k] / number_of_category[k], 3) if number_of_category[k] != 0 else 0.000
    for k in range(len(divisedList)):   mylist2[k] = round(number_of_correct_for_division[k] / number_of_division[k], 3) if number_of_division[k] != 0 else 0.000


    print("epoch" + str(currentEpoch) + "\t", end='')
    print(round((100 * correct / total), 3), "\t", end='')
    for k in range(datasetMaker.getNumOfClasses()): print(mylist[k], "\t", end='')
    print("\t\t", end='')
    for k in range(len(divisedList)): print(mylist2[k], "\t", end='')
    print("")
# ======================== test_model : end ============================================================================







# ======================== save_model ==================================================================================
def save_model(currentEpoch):
    if not os.path.isdir(os.path.join(modelpath, HARMONYNETNAME)):
        os.makedirs(os.path.join(modelpath, HARMONYNETNAME), exist_ok=True)

    torch.save(harmonyNet, os.path.join(modelpath, HARMONYNETNAME, 'cifar_model.pkl'))
    torch.save(harmonyNet, os.path.join(modelpath, HARMONYNETNAME, 'cifar_model_epo' + str(currentEpoch) + '.pkl'))
    print("save : ", currentEpoch)
# ======================== save_model : end ============================================================================










# ======================== train_normal ================================================================================
def train_harmonyNet(trainingSet_loader,useEval = False) :

    for i in range(harmonyNetEpochOffset +1, num_epoch):

        for j, [image, label] in enumerate(trainingSet_loader):
            x = Variable(image, requires_grad=True).cuda()
            y_ = Variable(label).cuda()
            harmonyY_ = Variable(torch.cuda.LongTensor(batch_size)).cuda()

            for k in range(batch_size):
                for d in range(num_of_division) :
                    if y_.data[k] in divisedList[d] : harmonyY_.data[k] =  d

            optimizer.zero_grad()
            outputm = harmonyNet.forward(x)
            loss = loss_func(outputm, harmonyY_)
            loss.backward()
            optimizer.step()


            if j % 100 == 0: print(loss.data, "epoch : " + str(i), j, "model name :" + HARMONYNETNAME)


        if useEval == True :
            test_harmonyNet(currentEpoch=i)
            pass

        if (i % 1 == 0):
            save_model(currentEpoch=i)

# ======================== train_normal : end===========================================================================


#=========================  eval_for_excel =============================================================================
def eval_for_excel(epoch = 0) :
    global harmonyNet
    if epoch <= 0:
        for i in range(epoch+1,num_epoch):
            harmonyNet = model_Loader.loadModel(i, info= False)
            test_harmonyNet(currentEpoch=i)
        pass
#=========================  eval_for_excel : end========================================================================






if __name__ == "__main__" :
    #============================ set hyper parameters      ================================================================
    batch_size = 128
    learning_rate = 0.0002
    num_epoch = 150
    modelpath = os.path.join('.', 'train_logs')
    #==========================   set hyper parameters : end  ==============================================================






    #============================ prepare dataset (CIFAR10) ================================================================
    #datasetMaker
    datasetName = 'cifar10'
    datasetMaker = DatasetMaker_cifar.DatasetMaker_cifar(datasetName = datasetName)
    print("train set : " , datasetMaker.getTrainset().__getitem__(0)[0].size(), datasetMaker.getTrainset().__len__())
    print("test set : " , datasetMaker.getTestset().__getitem__(0)[0].size(), datasetMaker.getTestset().__len__())

    #make dataset
    if datasetName == 'cifar10' :
        dataset1 = datasetMaker.get_slicedSet(datasetMaker.getTrainset() , 0 , 30000)
        dataset2 = datasetMaker.get_slicedSet(datasetMaker.getTrainset() , 30000 , 50000)
        testset  = datasetMaker.getTestset()
        dataset1biased2_3 = datasetMaker.get_biasedSet(dataset1,[2,3], 0)
        dataset1biased2_3_5 = datasetMaker.get_biasedSet(dataset1,[2,3,5], 0)
        dataset1biased_not2_3_5 = datasetMaker.get_biasedSet(dataset1,  [i for i in range(datasetMaker.getNumOfClasses()) if i not in [2,3,5] ], 0.1)
        dataset1biased_not2_3_5_balanced = datasetMaker.get_balancedSet(dataset1biased_not2_3_5)

    #load dataset
    dataset1_loader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    dataset2_loader = torch.utils.data.DataLoader(dataset2,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)
    testset_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)
    dataset1biased2_3_5_loader = torch.utils.data.DataLoader(dataset1biased2_3_5 ,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    dataset1biased_not2_3_5_loader = torch.utils.data.DataLoader(dataset1biased_not2_3_5 ,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    dataset1biased_not2_3_5_balanced_loader = torch.utils.data.DataLoader(dataset1biased_not2_3_5_balanced ,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)

    #========================== prepare dataset (CIFAR10)  : end  ==========================================================





    #========================= model selector ==============================================================================
    #MODELNAME = "CNN_BASIC"


    weakList = [2,3,5]
    divisedList = [  [i for i in range(datasetMaker.getNumOfClasses()) if i not in weakList ]  , weakList ]
    num_of_division = 2

    #BASEMODEL = 'googLeNet'
    BASEMODEL = 'middLeNet'
    #HARMONYNETNAME = "HarmonyNet"
    #HARMONYNETNAME = "HarmonyNet_set1"
    HARMONYNETNAME = "HarmonyNet_inbalnced_235"
    harmonyNetEpochOffset = 0
    model_Loader = Model_Loader.Model_Loader( num_of_classes=len(divisedList),
                                            basemodel=BASEMODEL,
                                            modelName=HARMONYNETNAME,
                                            epochOffset=harmonyNetEpochOffset,
                                            path=modelpath
                                        )
    harmonyNet = model_Loader.loadModel()
    #========================= model selector : end ========================================================================




    #=========================  set optimizer ==============================================================================
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(harmonyNet.parameters(), lr=learning_rate)
    #========================= set optimizer : end =========================================================================



    #======================== traing or eval ============================================================================
    train_harmonyNet(trainingSet_loader=dataset1biased_not2_3_5_balanced_loader,useEval=True)
    eval_for_excel(epoch = 0)
    #======================= training : end =============================================================================