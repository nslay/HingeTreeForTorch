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

def set_deterministic():
    pass
    
def shuffle(data, target, numTrain):
    if target.size <= 0 or numTrain < 0 or numTrain > target.size:
        return None
       
    indices = np.arange(target.size)
    np.random.shuffle(indices)

    indexTrain = indices[:numTrain]
    indexTest = indices[numTrain:]
    
    return data[indexTrain, :], target[indexTrain], data[indexTest, :], target[indexTest]

def train(snapshotroot, ensembleType, numTrees, depth, seed=0):
    xtrain, ytrain, xtest, ytest = datasets.load_abalone_reg()
    
    xtrain, ytrain, xval, yval = shuffle(xtrain, ytrain, 2089)

    metric="rmse"

    earlyStop=max(1,int(0.1*numTrees))

    clf = ensembleType(max_depth=depth, use_label_encoder=False, tree_method="exact", n_estimators=numTrees, random_state=seed)
    clf.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xval, yval)], eval_metric=metric, verbose=False, early_stopping_rounds=earlyStop)

    print(f"best iteration = {clf.best_iteration}, best_score = {clf.best_score}, best_ntree_limit = {clf.best_ntree_limit}")

    results = clf.evals_result()
    ypred = clf.predict(xtest)

    testLoss = (ypred - ytest).var()
    testR2 = 1.0 - testLoss / ytest.var()

    return testLoss, testR2, np.array(results["validation_1"][metric])

def main(device, **kwargs):
    snapshotroot = "abalone_reg_xgb"
    
    if not os.path.exists(snapshotroot):
        os.mkdir(snapshotroot)

    set_deterministic()
        
    numExperiments = 100
    
    for ensembleType in [ xgb.XGBRegressor, xgb.XGBRFRegressor ]:
        forestTypeName = "XGBRegressor"

        if ensembleType == xgb.XGBRFRegressor:
            forestTypeName = "XGBRFRegressor"

        for numTrees in [ 1, 10, 50, 100 ]:
            for depth in [ 1, 3, 5, 7, 10 ]:
                print(f"Info: Running XGBClassifier, numTrees = {numTrees}, depth = {depth} ...")
                
                pickleFileName=os.path.join(snapshotroot, f"{forestTypeName}_{numTrees}_{depth}.pkl")
 
                allTestLosses = np.zeros(numExperiments, dtype=np.float32)
                allTestR2 = np.zeros(numExperiments, dtype=np.float32)
                allValLosses = []
 
                for i in range(numExperiments):
                    #snapshotdir = os.path.join(snapshotroot, forestTypeName, str(numTrees), str(depth), str(i))
                    snapshotdir = snapshotroot
                    
                    if not os.path.exists(snapshotdir):
                        os.makedirs(snapshotdir)
                        
                    seedVal = seed(f"abalone_reg{i}")
                    
                    print(f"Training {snapshotdir} ...")
                    
                    testLoss, testR2, valLosses = train(snapshotroot=snapshotdir, ensembleType=ensembleType, numTrees=numTrees, depth=depth, seed=seedVal)
                    allTestLosses[i] = testLoss
                    allTestR2[i] = testR2
                    allValLosses.append(valLosses)

                print(f"Info: Mean test loss = {allTestLosses.mean()}, std = {allTestLosses.std()}, mean test R2 = {allTestR2.mean()}, std = {allTestR2.std()}", flush=True)
                
                print(f"Info: Saving results to {pickleFileName} ...", flush=True)
                
                with open(pickleFileName, "wb") as f:
                    pickle.dump({"allTestLosses": allTestLosses, "allTestR2": allTestR2, "allValLosses": allValLosses, "depth": depth, "numTrees": numTrees, "forestTypeName": forestTypeName}, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abalone_reg experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name to train/test (e.g. cuda:1)")

    args = parser.parse_args()
    
    main(**vars(args))
    
