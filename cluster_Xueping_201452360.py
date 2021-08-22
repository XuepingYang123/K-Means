import numpy as np
import matplotlib.pyplot as plt

def load_data(fname, label):
    features = []

    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            p.append(label)
            features.append(np.array(p[1:], float))
    return np.array(features)

class Cluster(object):

    def __init__(self, dataset, k, maxIter):
        self.k = k
        self.maxIter = maxIter
        self.YArray = []
        # Generate indices for initial means
        YIndex = np.random.choice(len(dataset), self.k, replace=False)
        for index in YIndex:
            self.YArray.append(dataset[index])

    #caculate Eulidean distance of K-means
    def EuclideanDistance(self,X,Y):
        return np.sqrt(np.sum(np.square(X-Y)))

    # caculate Manhattan distance of K-medians
    def ManhattanDistance(self, X, Y):
        return np.sum(np.abs(X - Y))

    #Assignment phase
    #Assign all objects in the dataset to the closest representative
    def assignToY(self, dataset, isMean):
        # Generate k clusters to store points(X)
        self.clusters = []
        for i in range(len(dataset)):
            X = dataset[i]
            # find the closest mean
            meanIndOfX = -1
            distanceToClosestMean = np.Inf
            for centroid in self.YArray:
                dist= np.Inf
                if isMean:
                    dist = self.EuclideanDistance(X, centroid)
                else:
                    dist = self.ManhattanDistance(X, centroid)
                if dist < distanceToClosestMean:
                    distanceToClosestMean = dist
                    pointOfX = centroid
            # assign X to its closest mean
            self.clusters.append(pointOfX)

    #Optimisation phase
    #Compute the new representatives as the means of the current clusters
    def updateY(self, dataset, allGroups, isMean):
        allGroups = []
        for mean in self.YArray:
            oneGroup = []
            for i in range(len(self.clusters)):
                if (self.clusters[i] == mean).all():
                    oneGroup.append(dataset[i])
            allGroups.append(oneGroup)
        newYArray = []
        for item in allGroups:
            if isMean:
                newY = np.mean(item)
            else:
                newY = np.median(item)

            newYArray.append(newY)
        self.YArray = newYArray
        return oneGroup, allGroups

    #give each object an label and caculate the sum  of each labels in each cluster
    def putLabels(self):
        allGroups = []
        label1CountArray = []
        label2CountArray = []
        label3CountArray = []
        label4CountArray = []
        for representitive in self.YArray:
            oneGroup = []
            label_1 = 0
            label_2 = 0
            label_3 = 0
            label_4 = 0
            for i in range(len(self.clusters)):
                if (self.clusters[i] == representitive).all():
                    oneGroup.append(dataset[i][:-1])
                    if dataset[i][-1] == 1:
                        label_1 += 1
                    elif dataset[i][-1] == 2:
                        label_2 += 1
                    elif dataset[i][-1] == 3:
                        label_3 += 1
                    elif dataset[i][-1] == 4:
                        label_4 += 1
            label1CountArray.append(label_1)
            label2CountArray.append(label_2)
            label3CountArray.append(label_3)
            label4CountArray.append(label_4)
            allGroups.append(oneGroup)
        return allGroups, label1CountArray,label2CountArray,label3CountArray,label4CountArray

    #caculate B-CUBED precision, recall, fscore
    def bcubed(self, dataset,allGroups, label1CountArray,label2CountArray,label3CountArray,label4CountArray):
        precision = recall = fscore = 0
        for i in range(len(allGroups)):
            if len(allGroups[i]) != 0:
                label1Precision=label1CountArray[i]/len(allGroups[i])
                label2Precision=label2CountArray[i]/len(allGroups[i])
                label3Precision=label3CountArray[i]/len(allGroups[i])
                label4Precision=label4CountArray[i]/len(allGroups[i])

                label1Recall = label1CountArray[i] / sum(label1CountArray)
                label2Recall = label2CountArray[i] / sum(label2CountArray)
                label3Recall = label3CountArray[i] / sum(label3CountArray)
                label4Recall = label4CountArray[i] / sum(label4CountArray)
            label1F_score=0
            if (label1Precision + label1Recall)!=0:
                label1F_score = 2 * label1Precision * label1Recall / (label1Precision + label1Recall)
            label2F_score=0
            if (label2Precision + label2Recall) != 0:
                label2F_score = 2 * label2Precision * label2Recall / (label2Precision + label2Recall)
            label3F_score=0
            if (label3Precision + label3Recall) != 0:
                label3F_score = 2 * label3Precision * label3Recall / (label3Precision + label3Recall)
            label4F_score=0
            if (label4Precision + label4Recall) != 0:
                label4F_score = 2 * label4Precision * label4Recall / (label4Precision + label4Recall)

            precision += (label1Precision*label1CountArray[i])/len(dataset)+\
                               (label2Precision*label2CountArray[i])/len(dataset)+\
                               (label3Precision*label3CountArray[i])/len(dataset)+\
                               (label4Precision*label4CountArray[i])/len(dataset)

            recall += (label1Recall * label1CountArray[i]) / len(dataset) + \
                         (label2Recall * label2CountArray[i]) / len(dataset) + \
                         (label3Recall * label3CountArray[i]) / len(dataset) + \
                         (label4Recall * label4CountArray[i]) / len(dataset)

            fscore += (label1F_score * label1CountArray[i]) / len(dataset) + \
                         (label2F_score * label2CountArray[i]) / len(dataset) + \
                         (label3F_score * label3CountArray[i]) / len(dataset) + \
                         (label4F_score * label4CountArray[i]) / len(dataset)
        return precision, recall, fscore

    #plot feagure of the Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall,
    # and F-score for each set of clusters.
    def plot(self,precisionArray, recallArray, F_scoreArray, isMean):
        x = np.arange(1, len(precisionArray) + 1, 1)
        y1 = precisionArray
        y2 = recallArray
        y3 = F_scoreArray
        if isMean:
            plt.title('kMeans figure')
        else:
            plt.title('kMedians figure')

        plt.xlabel('k value:1-9')
        plt.ylabel('B-CUBED')
        l1, = plt.plot(x, y1, color='red', linewidth=1, marker='o',label='precision')
        l2, = plt.plot(x, y2, color='blue', linewidth=1, marker='o',label='recall')
        l3, = plt.plot(x, y3, color='green',  linewidth=1, marker='o',label='F-score')
        plt.legend(loc='best')
        plt.show()

# normalise dataset
def l2_normal(dataset):
    newDataset = []
    for object in dataset:
        aDataset = []
        label = object[-1]
        object = object[:-1]
        for feature in object:
            aDataset.append(feature / np.linalg.norm(object))
        aDataset.append(label)
        newDataset.append(aDataset)
    return np.array(newDataset)

# re-run the k-means and K-medians clustering algorithm
def reRun(dataset, isMean):
    precisionArray = []
    recallArray = []
    F_scoreArray = []
    for k in range(9):
        test = Cluster(dataset, k + 1, 30)
        i = 0
        precision = None
        recall = None
        F_score = None
        while i < test.maxIter:
            test.assignToY(dataset, isMean)
            allGroups, label1CountArray, label2CountArray, label3CountArray, label4CountArray = test.putLabels()
            test.updateY(dataset, allGroups, isMean)
            precision, recall, fscore = test.bcubed(dataset, allGroups, label1CountArray, label2CountArray,
                                                    label3CountArray, label4CountArray)
            i += 1
        precisionArray.append(precision)
        recallArray.append(recall)
        F_scoreArray.append(fscore)
    test.plot(precisionArray, recallArray, F_scoreArray, isMean)

if __name__ == '__main__':
    clust1 = load_data("animals",'1')
    clust2 = load_data("countries",'2')
    clust3 = load_data("fruits",'3')
    clust4 = load_data("veggies", '4')
    dataset = np.concatenate((clust1, clust2, clust3, clust4))
    reRun(dataset, True)
    reRun(dataset, False)
    reRun(l2_normal(dataset), True)
    reRun(l2_normal(dataset), False)

