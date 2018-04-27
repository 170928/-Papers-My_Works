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
def test_model(currentEpoch):
    correct = 0
    total = 0

    number_of_correct_for_category = [0 for k in range(10)]
    number_of_category = [0 for k in range(10)]
    mylist = [0 for k in range(10)]


    for image, label in dataset2_loader:
        x = Variable(image, volatile=True).cuda()
        y_ = Variable(label).cuda()

        output = model.forward(x)
        _, output_index = torch.max(output, 1)

        for k in range(batch_size):
            number_of_category[y_.data[k]] += 1
            if (output_index.data[k] == y_.data[k]):
                number_of_correct_for_category[y_.data[k]] += 1

        total += label.size(0)
        correct += (output_index == y_).sum().float()


    for k in range(10):   mylist[k] = round(number_of_correct_for_category[k] / number_of_category[k], 3) if number_of_category[k] != 0 else 0.000

    print("epoch" + str(currentEpoch) + "\t", end='')
    print(round((100 * correct / total).data[0], 3), "\t", end='')
    for k in range(10): print(mylist[k], "\t", end='')
    print("")
# ======================== test_model : end ============================================================================



# ======================== test_model2 ==================================================================================
def test_conductor(currentEpoch):
    correct = 0
    total = 0

    number_of_correct_for_category = [0 for k in range(datasetMaker.getNumOfClasses())]
    number_of_category = [0 for k in range(datasetMaker.getNumOfClasses())]
    mylist = [0 for k in range(datasetMaker.getNumOfClasses())]

    number_of_correct_for_division = [0 for k in range(len(divisedList))]
    number_of_division = [0 for k in range(len(divisedList))]
    mylist2 = [0 for k in range(len(divisedList))]

    for image, label in testset_loader:
        x = Variable(image, volatile=True).cuda()
        y_ = Variable(label).cuda()

        output = model.forward(x)
        _, output_index = torch.max(output, 1)

        for k in range(batch_size):
            number_of_category[y_.data[k]] += 1
            total += 1

            for d in range(num_of_division):
                if y_.data[k] in divisedList[d]:
                    break

            number_of_division[d] += 1

            if output_index.data[k] in divisedList[d] :
                number_of_correct_for_category[y_.data[k]] += 1
                number_of_correct_for_division[d] += 1
                correct += 1

    for k in range(datasetMaker.getNumOfClasses()):   mylist[k] = round(
        number_of_correct_for_category[k] / number_of_category[k], 3) if number_of_category[k] != 0 else 0.000
    for k in range(len(divisedList)):   mylist2[k] = round(number_of_correct_for_division[k] / number_of_division[k],
                                                           3) if number_of_division[k] != 0 else 0.000

    print("epoch" + str(currentEpoch) + "\t", end='')
    print(round((100 * correct / total), 3), "\t", end='')
    for k in range(datasetMaker.getNumOfClasses()): print(mylist[k], "\t", end='')
    print("\t\t", end='')
    for k in range(len(divisedList)): print(mylist2[k], "\t", end='')
    print("")
# ======================== test_model2 : end ============================================================================




# ======================== save_model ==================================================================================
def save_model(currentEpoch):
    if not os.path.isdir(os.path.join(modelpath, MODELNAME)):
        os.makedirs(os.path.join(modelpath, MODELNAME), exist_ok=True)

    torch.save(model, os.path.join(modelpath, MODELNAME, 'cifar_model.pkl'))
    torch.save(model, os.path.join(modelpath, MODELNAME, 'cifar_model_epo' + str(currentEpoch) + '.pkl'))
    print("save : ", currentEpoch)
# ======================== save_model : end ============================================================================






# ======================== train_normal ================================================================================
def train_normal(trainingSet_loader, useEval = False) :

    for i in range(epochOffset + 1, num_epoch):
        for j, [image, label] in enumerate(trainingSet_loader):  # set1biased1_loader    epoch by epoch finrtuning, biased set 5000

            x = Variable(image).cuda()
            y_ = Variable(label).cuda()

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            if j % 10 == 0: print(loss.data, "epoch : " + str(i), j, "model name :" + MODELNAME)


        if useEval == True :
            test_model(currentEpoch=i)

        if (i % 1 == 0):
            save_model(currentEpoch=i)

# ======================== train_normal : end===========================================================================







# ========================= train_EBE_Biased    ========================================================================
def train_EBE_Biased(baseSet_loader,biasedSet,biasedSetSize,EBE_Period =1,  useEval = False):

    for i in range(epochOffset + 1, num_epoch):

        random.shuffle(biasedSet)
        set1biased1_loader = torch.utils.data.DataLoader(biasedSet[0:biasedSetSize], batch_size=batch_size,
                                                         shuffle=True, num_workers=2, drop_last=True)


        for j, [image, label] in enumerate(
                baseSet_loader if i % (EBE_Period+1) == (EBE_Period) else set1biased1_loader):  # set1biased1_loader    epoch by epoch finrtuning, biased set 5000

            x = Variable(image).cuda()
            y_ = Variable(label).cuda()

            optimizer.zero_grad()
            output = model.forward(x)

            loss = loss_func(output, y_)

            loss.backward()
            optimizer.step()

            if j % 100 == 0:
                print(loss.data, "epoch : " + str(i), j, "model name :" + MODELNAME)


        if useEval == True:
            test_model(currentEpoch=i)

        if (i % 1 == 0):
            save_model(currentEpoch=i)


# ========================= train_EBE_Biased :end=======================================================================






#=========================  eval_for_excel =============================================================================
def eval_for_excel(epoch = 0) :
    global model
    if epoch == 0:
        for i in range(epoch+1,num_epoch):
            model = model_Loader.loadModel(i,info=False)
            test_model(i)
#=========================  eval_for_excel : end========================================================================

#=========================  eval_for_excel =============================================================================
def eval_conductor_for_excel(epoch = 0) :
    global model
    if epoch == 0:
        for i in range(epoch+1,num_epoch):
            model = model_Loader.loadModel(i,info=False)
            test_conductor(i)
#=========================  eval_for_excel : end========================================================================




if __name__ == "__main__" :
    #============================ set hyper parameters      ================================================================
    batch_size = 128
    learning_rate = 0.0002
    num_epoch = 150
    modelpath = os.path.join('.', 'train_logs' )
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
        dataset1biased_not2_3_5 = datasetMaker.get_biasedSet(dataset1, [i for i in range(datasetMaker.getNumOfClasses()) if i not in [2, 3, 5]], 0.1)
        dataset1biased_not2_3_5_balanced = datasetMaker.get_balancedSet(dataset1biased_not2_3_5)
        dataset1biased_not2_3_5_balanced_biased235 = datasetMaker.get_biasedSet(dataset1biased_not2_3_5_balanced, [2,3,5], 0)


    #load dataset
    dataset1_loader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    dataset2_loader = torch.utils.data.DataLoader(dataset2,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)
    testset_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)
    dataset1biased2_3_5_loader = torch.utils.data.DataLoader(dataset1biased2_3_5 ,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    dataset1biased_not2_3_5_balanced_loader = torch.utils.data.DataLoader(dataset1biased_not2_3_5_balanced ,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    #========================== prepare dataset (CIFAR10)  : end  ==========================================================





    #========================= model selector ==============================================================================
    weakList = [2, 3, 5]
    divisedList = [[i for i in range(datasetMaker.getNumOfClasses()) if i not in weakList], weakList]
    num_of_division = 2

    #MODELNAME = "CNN_BASIC"
    MODELNAME = "GOOGLENET_normal"  #target model
    #MODELNAME = "GOOGLENET_inbalanced_235"
    #MODELNAME = "GG_inbalanced_235_EBE_biased" #googlenet epoch by epoch finetuning
    BASEMODEL = 'googLeNet'
    epochOffset = 0

    model_Loader = Model_Loader.Model_Loader( num_of_classes=datasetMaker.getNumOfClasses(),
                                            basemodel=BASEMODEL,
                                            modelName=MODELNAME,
                                            epochOffset=epochOffset,
                                            path=modelpath
                                        )

    model = model_Loader.loadModel()
    #========================= model selector : end ========================================================================





    #=========================  set optimizer ==============================================================================
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #========================= set optimizer : end =========================================================================




    #======================== traing and eval ==============================================================================
    train_normal(trainingSet_loader=dataset1_loader,useEval=True)

    #====biased training
    #baseSet_loader= dataset1biased_not2_3_5_balanced_loader
    #biasedSet = dataset1biased_not2_3_5_balanced_biased235
    baseSet_loader = dataset1_loader
    biasedSet = dataset1biased2_3_5_loader

    biasedSetSize = int(len(baseSet_loader)*batch_size*2/10)
    #train_EBE_Biased(baseSet_loader=baseSet_loader,biasedSet=biasedSet, biasedSetSize=biasedSetSize,EBE_Period= 1,useEval=True)
    #eval_for_excel()
    #eval_conductor_for_excel()
    #======================= training : end =============================================================================