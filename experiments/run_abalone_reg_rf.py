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
import datasets
from sklearn.ensemble import RandomForestRegressor

def seed(seedStr):
    seed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(seed)
    np.random.seed(seed) # Bad way to do this!
    return seed

def set_deterministic():
    pass
    
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

def train(snapshotroot, ensembleType, numTrees, depth, seed=0):
    xtrain, ytrain, xtest, ytest = datasets.load_abalone_reg()
    
    clf = ensembleType(random_state=seed, n_estimators=numTrees, max_features="sqrt", max_depth=depth)
    clf.fit(xtrain, ytrain)

    ypred = clf.predict(xtest)

    testLoss = (ypred - ytest).var()
    testR2 = 1.0 - testLoss / ytest.var()

    return testLoss, testR2

def main(device, **kwargs):
    snapshotroot = "abalone_reg_rf"
    
    if not os.path.exists(snapshotroot):
        os.mkdir(snapshotroot)

    set_deterministic()
        
    numExperiments = 100
    
    for ensembleType in [ RandomForestRegressor ]:
        forestTypeName = "RandomForestRegressor"

        for numTrees in [ 1, 10, 50, 100 ]:
            for depth in [ 1, 3, 5, 7, 10 ]:
                print(f"Info: Running {forestTypeName}, numTrees = {numTrees}, depth = {depth} ...")
                
                pickleFileName=os.path.join(snapshotroot, f"{forestTypeName}_{numTrees}_{depth}.pkl")
 
                allTestLosses = np.zeros(numExperiments, dtype=np.float32)
                allTestR2 = np.zeros(numExperiments, dtype=np.float32)
 
                for i in range(numExperiments):
                    #snapshotdir = os.path.join(snapshotroot, forestTypeName, str(numTrees), str(depth), str(i))
                    snapshotdir = snapshotroot
                    
                    if not os.path.exists(snapshotdir):
                        os.makedirs(snapshotdir)
                        
                    seedVal = seed(f"abalone{i}")
                    
                    print(f"Training {snapshotdir} ...")
                    
                    testLoss, testR2 = train(snapshotroot=snapshotdir, ensembleType=ensembleType, numTrees=numTrees, depth=depth, seed=seedVal)
                    allTestLosses[i] = testLoss
                    allTestR2[i] = testR2

                print(f"Info: Mean test loss = {allTestLosses.mean()}, std = {allTestLosses.std()}, mean test R2 = {allTestR2.mean()}, std = {allTestR2.std()}", flush=True)
                
                print(f"Info: Saving results to {pickleFileName} ...")

                with open(pickleFileName, "wb") as f:
                    pickle.dump({"allTestLosses": allTestLosses, "allTestR2": allTestR2, "depth": depth, "numTrees": numTrees, "forestTypeName": forestTypeName}, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abalone experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name to train/test (e.g. cuda:1)")

    args = parser.parse_args()
    
    main(**vars(args))
    
