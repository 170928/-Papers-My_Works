import torchvision.datasets as dset
import torchvision.transforms as transforms
import copy
import random

class DatasetMaker_cifar :
    def __init__(self, datasetName ,path = "./"):
        self.datasetName = datasetName
        self.path = path

        self.trainset = []
        self.testset = []
        self.num_of_trainset = []
        self.num_of_testset = []
        self.num_Of_classes = 0

        if datasetName == 'cifar10' :
            self.trainset = dset.CIFAR10("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
            self.testset = dset.CIFAR10("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
            self.num_of_trainset = 50000
            self.num_of_testset = 10000
            self.num_of_classes = 10

        else :
            print("no dataset error")


    def getDatasetName(self):
        return self.datasetName

    def getTrainset(self):
        return copy.deepcopy(self.trainset)


    def getTestset(self):
        return copy.deepcopy(self.testset)

    def getNumOfClasses(self):
        return self.num_of_classes


    def get_slicedSet(self,originalSet, start,end):
        slicedSet = [originalSet[i] for i in range(start,end)]
        return slicedSet


    def get_biasedSet(self,originalSet, biasList = [], pureRate = 0):
        biasedSet = []
        for i in range(0, len(originalSet)):
            if (originalSet[i][1] not in biasList):
                if not int(random.random() * 1.0 / pureRate)   if pureRate != 0 else  0 :
                    biasedSet.append(originalSet[i])

            else :
                biasedSet.append(originalSet[i])

        return biasedSet




    def get_balancedSet(self,originalSet):
        balancedSet = []

        data_per_categories = [ [] for i in range(self.num_of_classes)]
        for i in range (len(originalSet)):
            data_per_categories[originalSet[i][1]].append(originalSet[i])
        print("<<get_balanced_dataset>>", "source set len: ", len(originalSet))
        num_per_categories = [len(i) for i in copy.deepcopy(data_per_categories)]
        print( "before : ", num_per_categories)
        maxNum = max(num_per_categories)
        maxIndex = num_per_categories.index(maxNum)

        for i in range(self.num_of_classes) :
            while len(data_per_categories[i]) < len(data_per_categories[maxIndex]) :
                data_per_categories[i].append( random.choice(data_per_categories[i]))

        for i in range(self.num_of_classes) : balancedSet += data_per_categories[i]
        random.shuffle(balancedSet)



        data_per_categories = [ [] for i in range(self.num_of_classes)]
        for i in range(len(balancedSet)):
            data_per_categories[balancedSet[i][1]].append(balancedSet[i])
        num_per_categories = [len(i) for i in copy.deepcopy(data_per_categories)]
        print( "after",  num_per_categories)

        return balancedSet