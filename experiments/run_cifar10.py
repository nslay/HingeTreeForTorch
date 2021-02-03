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
import time
import argparse
import random
import pickle
import numpy as np
import cv2
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RandomHingeForest import RandomHingeForest, RandomHingeFern
import datasets

class Net(nn.Module):
    def __init__(self, forestType, numTrees, depth):
        super(Net, self).__init__()

        self.bn11 = nn.BatchNorm2d(40, affine=False)
        self.conv11 = nn.Conv2d(3, 40, 3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(40, affine=False)
        self.conv12 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(40, affine=False)
        self.conv13 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
    
        self.pool = nn.MaxPool2d(2, 2)
    
        self.conv21 = nn.Conv2d(40, 80, 3, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(80, affine=False)
        self.conv22 = nn.Conv2d(80, 80, 3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(80, affine=False)
        self.conv23 = nn.Conv2d(80, 80, 3, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(80, affine=False)
    
        self.conv31 = nn.Conv2d(80, 160, 3, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(160, affine=False)
        self.conv32 = nn.Conv2d(160, 160, 3, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(160, affine=False)
        self.conv33 = nn.Conv2d(160, 160, 3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(160, affine=False)
    
        self.conv41 = nn.Conv2d(160, 320, 3, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(320, affine=False)
        self.conv42 = nn.Conv2d(320, 320, 3, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(320, affine=False)
        self.conv43 = nn.Conv2d(320, 320, 3, padding=1, bias=False)
        self.bn43 = nn.BatchNorm2d(320, affine=False)

        self.conv51 = nn.Conv2d(320, 640, 3, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(640, affine=False)
        self.conv52 = nn.Conv2d(640, 640, 3, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(640, affine=False)
        self.conv53 = nn.Conv2d(640, 640, 3, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(640, affine=False)
  
        self.forestbn1 = nn.BatchNorm1d(640, affine=False)
        self.forest1 = forestType(in_channels=640, out_channels=numTrees, extra_outputs=10, depth=depth)

    def forward(self, x):
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = self.pool(F.relu(self.bn13(self.conv13(x))))
        
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.pool(F.relu(self.bn23(self.conv23(x))))
        
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = self.pool(F.relu(self.bn33(self.conv33(x))))
        
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = self.pool(F.relu(self.bn43(self.conv43(x))))

        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = self.pool(F.relu(self.bn53(self.conv53(x))))
       
        x = x.view(-1, 640)
        x = self.forest1(self.forestbn1(x))
        x = x.sum(dim=1)
        
        return x
        
def seed(seedStr):
    seed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(seed)
    np.random.seed(seed) # Bad way to do this!
    #torch.random.manual_seed(seed)
    #torch.cuda.random.manual_seed(seed)
    torch.manual_seed(seed)
    
def set_deterministic():
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def augment(data, target):
    rotPos = np.empty_like(data)
    rotNeg = np.empty_like(data)

    Rpos = cv2.getRotationMatrix2D((16, 16), 4, 1)
    Rneg = cv2.getRotationMatrix2D((16, 16), -4, 1)

    for i in range(data.shape[0]):
        tmp = np.transpose(data[i,:,:,:], (1, 2, 0))

        tmpPos = cv2.warpAffine(tmp, Rpos, (32,32))
        tmpNeg = cv2.warpAffine(tmp, Rneg, (32,32))

        #cv2.imwrite("tmpPos.png", tmpPos)
        #cv2.imwrite("tmpNeg.png", tmpNeg)
        #exit(1)

        rotPos[i,:,:,:] = np.transpose(tmpPos, (2, 0, 1))
        rotNeg[i,:,:,:] = np.transpose(tmpNeg, (2, 0, 1))

    augData = np.concatenate((data, rotPos, rotNeg))
    augTarget = np.tile(target, 3)
    #augTarget = np.concatenate((target, target, target))

    # N x 3 x H x W
    flippedData = np.flip(augData, axis=3)

    #cv2.imwrite("flipped.png", np.transpose(flippedData[0,:,:,:], (1,2,0)))
    #exit(1)

    augData = np.concatenate((augData, flippedData))
    #augTarget = np.concatenate((augTarget, augTarget))
    augTarget = np.tile(augTarget, 2)

    return augData, augTarget


# See also:
# https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/4
def add_weight_decay(model, weightDecay):
    decay = []
    noDecay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".thresholds") or name.endswith(".ordinals"):
            noDecay.append(param)
        else:
            decay.append(param)

    return [{ "params": noDecay, "weight_decay": 0.0 }, { "params": decay, "weight_decay": weightDecay }]

def balanced_shuffle(data, target, numTrain):
    if target.size <= 0 or numTrain < 0 or numTrain > target.size:
        return None
       
    if not np.issubdtype(target.dtype, np.integer):
        return None
        
    numTest = target.size - numTrain
    percTrain = float(numTrain) / float(target.size)
 
    minLabel = np.min(target)
    maxLabel = np.max(target)
    
    if minLabel == maxLabel:
        return None # What?
    
    indexTrain = np.zeros(numTrain, dtype=np.int64)
    indexTest = np.zeros(numTest, dtype=np.int64)
    
    trainPos = 0
    testPos = 0
    
    for label in range(minLabel, maxLabel):
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
        
    # Deal with last label separately
    indices = np.where(target == maxLabel)[0]
    np.random.shuffle(indices)
    
    trainSize = numTrain - trainPos
    testSize = numTest - testPos
    
    trainBegin = trainPos
    trainEnd = trainBegin + trainSize
    
    testBegin = testPos
    testEnd = testBegin + testSize
    
    indexTrain[trainBegin:trainEnd] = indices[:trainSize]
    indexTest[testBegin:testEnd] = indices[trainSize:]
    
    np.random.shuffle(indexTrain)
    np.random.shuffle(indexTest)
    
    return data[indexTrain, :], target[indexTrain], data[indexTest, :], target[indexTest]
    
def batches(x, y, batchSize):
    numBatches = int(x.shape[0] / batchSize)
    
    for b in range(numBatches):
        begin = b*batchSize
        end = begin + batchSize
        yield x[begin:end, :].contiguous(), y[begin:end].contiguous()
        
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
        yield torch.cat((xpart1, xpart2), dim=0).contiguous(), torch.cat((ypart1, ypart2)).contiguous()

def train(snapshotroot, device, forestType, numTrees, depth):
    xtrain, ytrain, xtest, ytest = datasets.load_cifar10()

    ytrain = ytrain.astype(np.int32) # For shuffle

    # This is the usual way...
    #xtrain, ytrain, xval, yval = balanced_shuffle(xtrain, ytrain, 40000)
    #xtrain, ytrain = augment(xtrain, ytrain)

    # This works better (for test set)...
    xtrain, ytrain = augment(xtrain, ytrain)
    xtrain, ytrain, xval, yval = balanced_shuffle(xtrain, ytrain, 240000)

    # Transfer this data to the device
    xtrain = torch.from_numpy(xtrain).type(torch.float32).to(device)
    ytrain = torch.from_numpy(ytrain).type(torch.long).to(device)
    xval = torch.from_numpy(xval).type(torch.float32).to(device)
    yval = torch.from_numpy(yval).type(torch.long).to(device)
    xtest = torch.from_numpy(xtest).type(torch.float32).to(device)
    ytest = torch.from_numpy(ytest).type(torch.long).to(device)
    
    net = Net(forestType, numTrees, depth).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    #optimizer = optim.Adam(add_weight_decay(net,5e-3), lr = 0.001)

    # Count parameters
    numParams = sum(params.numel() for params in net.parameters())
    numTrainable = sum(params.numel() for params in net.parameters() if params.requires_grad)
    print(f"There are {numParams} parameters total in this model ({numTrainable} are trainable)")

    numEpochs=300
    #batchSize=256
    batchSize=200
    
    indices = [ i for i in range(xtrain.shape[0]) ]
    
    bestEpoch=numEpochs-1
    bestAccuracy=0.0
    bestLoss=1000.0
    
    valLosses = np.zeros([numEpochs])
    
    for epoch in range(numEpochs):
        #t = time.time()
        random.shuffle(indices)

        xtrain = xtrain[indices, :]
        ytrain = ytrain[indices]

        runningLoss = 0.0
        count = 0
        for xbatch, ybatch in batches(xtrain, ytrain, batchSize):
            #t = time.time()
            optimizer.zero_grad()
            
            outputs = net(xbatch)
            loss = criterion(outputs, ybatch)
            
            loss.backward()
            
            optimizer.step()

            runningLoss += loss
            count += 1         
            #print(f"elapsed = {time.time() - t}, count = {count}")

        meanLoss = runningLoss/count
        
        snapshotFile = os.path.join(snapshotroot, f"epoch_{epoch}")
        torch.save(net.state_dict(), snapshotFile)

        runningLoss = 0.0
        count = 0
       
        with torch.no_grad():
            net.train(False)

            for xbatch, ybatch in batches(xval, yval, 200):
            #for xbatch, ybatch in zip([xval], [yval]):
                outputs = net(xbatch)
                lossTmp = criterion(outputs, ybatch)
                
                runningLoss += lossTmp
                count += 1
                
            net.train(True)
                
        valLoss = runningLoss / count
       
        if valLoss < bestLoss:
            bestLoss = valLoss
            bestEpoch = epoch
        
        print(f"Info: Epoch = {epoch}, loss = {meanLoss}, validation loss = {valLoss}", flush=True)
        valLosses[epoch] = valLoss
        #print(f"elapsed = {time.time() - t}")
        
    snapshotFile = os.path.join(snapshotroot, f"epoch_{bestEpoch}")
    
    net = Net(forestType, numTrees, depth)
    net.load_state_dict(torch.load(snapshotFile, map_location="cpu"))
    net = net.to(device)

    totalCorrect = 0
    count = 0

    with torch.no_grad():
        net.train(False)

        for xbatch, ybatch in batches(xtest, ytest, 200):
        #for xbatch, ybatch in zip([xtest], [ytest]):
            outputs = net(xbatch)
            outputs = torch.argmax(outputs, dim=1)
            
            tmpCorrect = torch.sum(outputs == ybatch)
            
            totalCorrect += tmpCorrect
            count += xbatch.shape[0]

        accuracy = float(totalCorrect) / float(count)
        print(f"Info: Best epoch = {bestEpoch}, test accuracy = {accuracy}, misclassification rate = {1.0 - accuracy}", flush=True)
    
    return accuracy, valLosses

def main(device, **kwargs):
    snapshotroot = "cifar10"
    
    if not os.path.exists(snapshotroot):
        os.mkdir(snapshotroot)
        
    set_deterministic()

    numExperiments = 10
    
    for forestType in [ RandomHingeForest, RandomHingeFern ]:
    #for forestType in [ RandomHingeFern ]:
        forestTypeName = "RandomHingeForest"
        
        if forestType == RandomHingeFern:
            forestTypeName = "RandomHingeFern"
    
        #for numTrees in [ 1, 10, 50, 100 ]:
        for numTrees in [ 100 ]:
            #for depth in [ 1, 3, 5, 7, 10 ]:
            for depth in [ 10 ]:
                print(f"Info: Running {forestTypeName}, numTrees = {numTrees}, depth = {depth} ...", flush=True)
                
                pickleFileName=os.path.join(snapshotroot, f"{forestTypeName}_{numTrees}_{depth}.pkl")
 
                allTestAcc = np.zeros(numExperiments, dtype=np.float32)
                allValLosses = []
 
                for i in range(numExperiments):
                    #snapshotdir = os.path.join(snapshotroot, forestTypeName, str(numTrees), str(depth), str(i))
                    snapshotdir = snapshotroot
                    
                    if not os.path.exists(snapshotdir):
                        os.makedirs(snapshotdir)
                        
                    seed(f"cifar10{i}")
                    
                    print(f"Training {snapshotdir} ...", flush=True)
                    
                    testAcc, valLosses = train(snapshotroot=snapshotdir, device=device, forestType=forestType, numTrees=numTrees, depth=depth)
                    allTestAcc[i] = testAcc
                    allValLosses.append(valLosses)
                    
                print(f"Info: Mean test accuracy = {allTestAcc.mean()}, test misclassification = {1.0 - allTestAcc.mean()}, std = {allTestAcc.std()}", flush=True)
                
                print(f"Info: Saving results to {pickleFileName} ...", flush=True)
                
                with open(pickleFileName, "wb") as f:
                    pickle.dump({"allTestAcc": allTestAcc, "allValLosses": allValLosses, "depth": depth, "numTrees": numTrees, "forestTypeName": forestTypeName}, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cifar10 experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name to train/test (e.g. cuda:1)")

    args = parser.parse_args()
    
    main(**vars(args))
    
