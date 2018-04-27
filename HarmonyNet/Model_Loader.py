import torchvision.datasets as dset
import torchvision.transforms as transforms
import copy
import random
import os
from models import *
import torch

class Model_Loader :
    def __init__(self, num_of_classes, basemodel = 'googLeNet', modelName = 'noname', epochOffset = '0' ,  path = os.path.join('.', 'train_logs' )) :
        self.basemodel = basemodel
        self.modelName = modelName
        self.epochOffset = epochOffset
        self.path = path
        self.num_of_classes = num_of_classes

    def loadModel(self, epochOffset = None, info = True):
        if epochOffset == None : epochOffset = self.epochOffset
        model = None
        if(self.basemodel == 'googLeNet') :
            #========== make new model ==========
            model =  googlenet.GoogLeNet(self.num_of_classes).cuda()
        elif(self.basemodel == 'middLeNet') :
            model = middlenet_googlenet2.Middlenet(self.num_of_classes).cuda()

        #========= load saved model. epoch 0 means last model
        try:
            epochOffset_path = ('_epo' + str(epochOffset)) if epochOffset != 0 else ''
            model = torch.load(os.path.join(self.path, self.modelName, 'cifar_model' + epochOffset_path + '.pkl'))
            if info == True : print("model restored")

        except:
            if info == True: print("model not restored")

        return model



    def get_path(self):
        return self.path


