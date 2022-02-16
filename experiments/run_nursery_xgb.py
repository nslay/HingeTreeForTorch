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
import xgboost as xgb

def seed(seedStr):
    seed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(seed)
    np.random.seed(seed) # Bad way to do this!
    return seed

def set_deterministic(mode):
    pass
    
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

def train(snapshotroot, ensembleType, numTrees, depth, seed=0):
    xdata, ydata = datasets.load_nursery()
    
    # Labels
    ydata = ydata.astype(np.int32)

    xtrain, ytrain, xtest, ytest = balanced_shuffle(xdata, ydata, 0.7) # 70% for training, 30% for testing
    xtrain, ytrain, xval, yval = balanced_shuffle(xtrain, ytrain, 0.75)

    metric="mlogloss"

    earlyStop=max(1,int(0.1*numTrees))

    clf = ensembleType(max_depth=depth, use_label_encoder=False, tree_method="exact", n_estimators=numTrees, random_state=seed)
    clf.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xval, yval)], eval_metric=metric, verbose=False, early_stopping_rounds=earlyStop)

    print(f"best iteration = {clf.best_iteration}, best_score = {clf.best_score}, best_ntree_limit = {clf.best_ntree_limit}")

    results = clf.evals_result()
    ypred = clf.predict(xtest)

    acc = (ypred == ytest).mean()

    return acc, np.array(results["validation_1"][metric])

def main(device, **kwargs):
    snapshotroot = "nursery_xgb"
    
    if not os.path.exists(snapshotroot):
        os.mkdir(snapshotroot)

    set_deterministic(True)
        
    numExperiments = 100
    
    for ensembleType in [ xgb.XGBClassifier, xgb.XGBRFClassifier ]:
        forestTypeName = "XGBClassifier"

        if ensembleType == xgb.XGBRFClassifier:
            forestTypeName = "XGBRFClassifier"

        for numTrees in [ 1, 10, 50, 100 ]:
            for depth in [ 1, 3, 5, 7, 10 ]:
                print(f"Info: Running {forestTypeName}, numTrees = {numTrees}, depth = {depth} ...")
                
                pickleFileName=os.path.join(snapshotroot, f"{forestTypeName}_{numTrees}_{depth}.pkl")
 
                allTestAcc = np.zeros(numExperiments, dtype=np.float32)
                allValLosses = []
 
                for i in range(numExperiments):
                    #snapshotdir = os.path.join(snapshotroot, forestTypeName, str(numTrees), str(depth), str(i))
                    snapshotdir = snapshotroot
                    
                    if not os.path.exists(snapshotdir):
                        os.makedirs(snapshotdir)
                        
                    seedVal = seed(f"nursery{i}")
                    
                    print(f"Training {snapshotdir} ...")
                    
                    testAcc, valLosses = train(snapshotroot=snapshotdir, ensembleType=ensembleType, numTrees=numTrees, depth=depth, seed=seedVal)
                    allTestAcc[i] = testAcc
                    allValLosses.append(valLosses)

                print(f"Info: Mean test accuracy = {allTestAcc.mean()}, test misclassification = {1.0 - allTestAcc.mean()}, std = {allTestAcc.std()}")
                
                print(f"Info: Saving results to {pickleFileName} ...")
                
                with open(pickleFileName, "wb") as f:
                    pickle.dump({"allTestAcc": allTestAcc, "allValLosses": allValLosses, "depth": depth, "numTrees": numTrees, "forestTypeName": forestTypeName}, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nursery experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name to train/test (e.g. cuda:1)")

    args = parser.parse_args()
    
    main(**vars(args))
    
