# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# January 2021
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import argparse
import random
import pickle
import numpy as np
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RandomHingeForest import RandomHingeForest, RandomHingeFern
from deterministic import set_deterministic
import datasets

class Net(nn.Module):
    def __init__(self, forestType, numTrees, depth):
        super(Net, self).__init__()

        numFeatures=100
    
        self.features = nn.Linear(in_features=21, out_features=numFeatures, bias=False)
        self.bn = nn.BatchNorm1d(num_features=numFeatures, affine=False)
        self.forest= forestType(in_channels=numFeatures, out_channels=numTrees, depth=depth, init_type="random")
        self.agg = nn.Linear(in_features=numTrees, out_features=3)

    def forward(self, x):
        x = self.features(x)
        x = self.forest(self.bn(x))
        x = self.agg(x)
        return x
       
def seed(seedStr):
    seed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(seed)
    np.random.seed(seed) # Bad way to do this!
    #torch.random.manual_seed(seed)
    #torch.cuda.random.manual_seed(seed)
    torch.manual_seed(seed)

# NOTE: Different balanced_shuffle
def balanced_shuffle(data, target, percTrain):
    if percTrain < 0.0 or percTrain > 1.0:
        return None

    #if target.dtype != np.integer:
    if not np.issubdtype(target.dtype, np.integer):
        return None

    minLabel = np.min(target)
    maxLabel = np.max(target)
    
    if minLabel == maxLabel:
        return None # What?

    numTrain = 0
    numTest = 0

    # Compute numTrain and numTest
    for label in range(minLabel, maxLabel+1):
        indices = np.where(target == label)[0]

        trainSize = int(percTrain*indices.size + 0.5)
        testSize = indices.size - trainSize

        numTrain += trainSize
        numTest += testSize

    # Now assign the shuffled examples
    indexTrain = np.zeros(numTrain, dtype=np.int64)
    indexTest = np.zeros(numTest, dtype=np.int64)

    trainPos = 0
    testPos = 0
    
    for label in range(minLabel, maxLabel+1):
        indices = np.where(target == label)[0]
        np.random.shuffle(indices)
        
        trainSize = int(percTrain*indices.size + 0.5)
        testSize = indices.size - trainSize
        
        trainBegin = trainPos
        trainEnd = trainBegin + trainSize
        
        testBegin = testPos
        testEnd = testBegin + testSize
        
        indexTrain[trainBegin:trainEnd] = indices[:trainSize]
        indexTest[testBegin:testEnd] = indices[trainSize:]
        
        trainPos = trainEnd
        testPos = testEnd

    np.random.shuffle(indexTrain)
    np.random.shuffle(indexTest)
    
    return data[indexTrain, :], target[indexTrain], data[indexTest, :], target[indexTest]
    
# Change to torch tensors owing to poker's massive test set size
def batches(x, y, batchSize):
    numBatches = int(x.shape[0] / batchSize)
    
    for b in range(numBatches):
        begin = b*batchSize
        end = begin + batchSize
        yield x[begin:end, :], y[begin:end]
        
    # Wrap around
    if batchSize*numBatches < x.shape[0]:
        begin = batchSize*numBatches
        end = x.shape[0]
        length = end - begin
        
        xpart1 = x[begin:end, :]
        ypart1 = y[begin:end]
        
        missing = batchSize - length
        
        xpart2 = x[:missing, :]
        ypart2 = y[:missing]
        
        #yield np.concatenate((xpart1, xpart2), axis=0), np.concatenate((ypart1, ypart2))
        yield torch.cat((xpart1, xpart2), dim=0), torch.cat((ypart1, ypart2))

def train(snapshotroot, device, forestType, numTrees, depth):
    xtrain, ytrain, xtest, ytest = datasets.load_ann_thyroid()
    
    # Labels
    ytrain = ytrain.astype(np.int32)
    ytest = ytest.astype(np.int32)

    xtrain, ytrain, xval, yval = balanced_shuffle(xtrain, ytrain, 0.75)
    
    net = Net(forestType, numTrees, depth).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Transfer this data to the device
    xtrain = torch.from_numpy(xtrain).type(torch.float32).to(device)
    ytrain = torch.from_numpy(ytrain).type(torch.long).to(device)
    xval = torch.from_numpy(xval).type(torch.float32).to(device)
    yval = torch.from_numpy(yval).type(torch.long).to(device)
    xtest = torch.from_numpy(xtest).type(torch.float32).to(device)
    ytest = torch.from_numpy(ytest).type(torch.long).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    
    numEpochs=1000
    batchSize=50
    
    indices = [ i for i in range(xtrain.shape[0]) ]
    
    bestEpoch=numEpochs-1
    bestLoss=1000.0
    
    valLosses = np.zeros([numEpochs])
    
    for epoch in range(numEpochs):
        random.shuffle(indices)
        
        xtrain = xtrain[indices, :]
        ytrain = ytrain[indices]
        
        runningLoss = 0.0
        count = 0
        for xbatch, ybatch in batches(xtrain, ytrain, batchSize):
            optimizer.zero_grad()
            
            outputs = net(xbatch)
            loss = criterion(outputs, ybatch)
            
            loss.backward()
            
            optimizer.step()
            
            runningLoss += loss
            count += 1         

        meanLoss = runningLoss/count
        
        snapshotFile = os.path.join(snapshotroot, f"epoch_{epoch}")
        torch.save(net.state_dict(), snapshotFile)
        
        runningLoss = 0.0
        count = 0

        with torch.no_grad():
            net.train(False)
            #for xbatch, ybatch in batches(xval, yval, batchSize):
            for xbatch, ybatch in zip([xval], [yval]):
                outputs = net(xbatch)
                loss = criterion(outputs, ybatch)
                
                runningLoss += loss
                count += 1
                
            net.train(True)
            
        valLoss = runningLoss / count
        
        if valLoss < bestLoss:
            bestLoss = valLoss
            bestEpoch = epoch
        
        valLosses[epoch] = valLoss
        
        print(f"Info: epoch = {epoch}, loss = {meanLoss}, validation loss = {valLoss}", flush=True)

    snapshotFile = os.path.join(snapshotroot, f"epoch_{bestEpoch}")
    
    net = Net(forestType, numTrees, depth)
    net.load_state_dict(torch.load(snapshotFile, map_location="cpu"))
    net = net.to(device)
    
    totalCorrect = 0
    count = 0
    
    with torch.no_grad():
        net.train(False)
        for xbatch, ybatch in zip([xtest], [ytest]):
            outputs = net(xbatch)
            outputs = torch.argmax(outputs, dim=1)
            
            tmpCorrect = torch.sum(outputs == ybatch)
            
            totalCorrect += tmpCorrect
            count += xbatch.shape[0]

    accuracy = float(totalCorrect) / float(count)
    print(f"Info: Best epoch = {bestEpoch}, test accuracy = {accuracy}, misclassification rate = {1.0 - accuracy}", flush=True)
    
    return accuracy, valLosses

def main(device, **kwargs):
    snapshotroot = "ann_thyroid"
    
    if not os.path.exists(snapshotroot):
        os.mkdir(snapshotroot)

    set_deterministic(True)
    
    numExperiments = 100
    
    for forestType in [ RandomHingeForest, RandomHingeFern ]:
    #for forestType in [ RandomHingeFern ]:
        forestTypeName = "RandomHingeForest"
        
        if forestType == RandomHingeFern:
            forestTypeName = "RandomHingeFern"
    
        for numTrees in [ 1, 10, 50, 100 ]:
        #for numTrees in [ 100 ]:
            for depth in [ 1, 3, 5, 7, 10 ]:
            #for depth in [ 7 ]:
                print(f"Info: Running {forestTypeName}, numTrees = {numTrees}, depth = {depth} ...", flush=True)
                
                pickleFileName=os.path.join(snapshotroot, f"{forestTypeName}_{numTrees}_{depth}.pkl")
 
                allTestAcc = np.zeros(numExperiments, dtype=np.float32)
                allValLosses = []
 
                for i in range(numExperiments):
                    #snapshotdir = os.path.join(snapshotroot, forestTypeName, str(numTrees), str(depth), str(i))
                    snapshotdir = snapshotroot
                    
                    if not os.path.exists(snapshotdir):
                        os.makedirs(snapshotdir)
                        
                    seed(f"ann_thyroid{i}")
                    
                    print(f"Training {snapshotdir} ...", flush=True)
                    
                    testAcc, valLosses = train(snapshotroot=snapshotdir, device=device, forestType=forestType, numTrees=numTrees, depth=depth)
                    allTestAcc[i] = testAcc
                    allValLosses.append(valLosses)
                    
                print(f"Info: Mean test accuracy = {allTestAcc.mean()}, test misclassification = {1.0 - allTestAcc.mean()}, std = {allTestAcc.std()}", flush=True)
                
                print(f"Info: Saving results to {pickleFileName} ...", flush=True)
                
                with open(pickleFileName, "wb") as f:
                    pickle.dump({"allTestAcc": allTestAcc, "allValLosses": allValLosses, "depth": depth, "numTrees": numTrees, "forestTypeName": forestTypeName}, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ann_thyroid experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name to train ann_thyroid (e.g. cuda:1)")

    args = parser.parse_args()
    
    main(**vars(args))
    
