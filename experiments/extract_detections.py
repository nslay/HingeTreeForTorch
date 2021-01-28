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

import sys
import os
import glob
import pickle
import numpy as np

def main(folders):
    for path in folders:
        perfKey = "allTestAcc"
        fernBest = 0.0
        fernStd = 0.0
        forestBest = 0.0
        forestStd = 0.0
        numFerns = 0
        fernDepth = 0
        numTrees = 0
        treeDepth = 0
        xgbBest = 0.0
        xgbStd = 0.0
        xgbRfBest = 0.0
        xgbRfStd = 0.0
        xgbNumTrees = 0
        xgbDepth = 0
        xgbRfNumTrees = 0
        xgbRfDepth = 0
        rfBest = 0.0
        rfStd = 0.0
        rfNumTrees = 0
        rfDepth = 0
        lgbNumTrees=0
        lgbDepth = 0
        lgbBest = 0.0
        lgbStd = 0.0

        if os.path.isfile(path):
            with open(path, "rb") as f:
                perf = pickle.load(f)

            if "allTestR2" in perf.keys():
                perfKey = "allTestR2"

            score = perf[perfKey].mean()
            std = perf[perfKey].std()

            if perf["forestTypeName"] == "RandomHingeForest":
                if score > forestBest:
                    forestBest = score
                    forestStd = std
                    numTrees = perf["numTrees"]
                    treeDepth = perf["depth"]
            elif perf["forestTypeName"] == "RandomHingeFern":
                if score > fernBest:
                    fernBest = score
                    fernStd = std
                    numFerns = perf["numTrees"]
                    fernDepth = perf["depth"]
            elif perf["forestTypeName"] == "XGBClassifier" or perf["forestTypeName"] == "XGBRegressor":
                if score > xgbBest:
                    xgbBest = score
                    xgbStd = std
                    xgbNumTrees = perf["numTrees"]
                    xgbDepth = perf["depth"]
            elif perf["forestTypeName"] == "XGBRFClassifier" or perf["forestTypeName"] == "XGBRFRegressor":
                if score > xgbBest:
                    xgbRfBest = score
                    xgbRfStd = std
                    xgbRfNumTrees = perf["numTrees"]
                    xgbRfDepth = perf["depth"]
            elif perf["forestTypeName"] == "RandomForestClassifier" or perf["forestTypeName"] == "RandomForestRegressor":
                if score > rfBest:
                    rfBest = score
                    rfStd = std
                    rfNumTrees = perf["numTrees"]
                    rfDepth= perf["depth"]
            elif perf["forestTypeName"] == "LGBMClassifier" or perf["forestTypeName"] == "LGBMRegressor":
                if score > lgbBest:
                    lgbBest = score
                    lgbStd = std
                    lgbNumTrees = perf["numTrees"]
                    lgbDepth = perf["depth"]

        else:
            for pklFile in glob.glob(f"{path}/*.pkl"):
                with open(pklFile, "rb") as f:
                    perf = pickle.load(f)

                if "allTestR2" in perf.keys():
                    perfKey = "allTestR2"

                score = perf[perfKey].mean()
                std = perf[perfKey].std()

                if perf["forestTypeName"] == "RandomHingeForest":
                    if score > forestBest:
                        forestBest = score
                        forestStd = std
                        numTrees = perf["numTrees"]
                        treeDepth = perf["depth"]
                elif perf["forestTypeName"] == "RandomHingeFern":
                    if score > fernBest:
                        fernBest = score
                        fernStd = std
                        numFerns = perf["numTrees"]
                        fernDepth = perf["depth"]
                elif perf["forestTypeName"] == "XGBClassifier" or perf["forestTypeName"] == "XGBRegressor":
                    if score > xgbBest:
                        xgbBest = score
                        xgbStd = std
                        xgbNumTrees = perf["numTrees"]
                        xgbDepth = perf["depth"]
                elif perf["forestTypeName"] == "XGBRFClassifier" or perf["forestTypeName"] == "XGBRFRegressor":
                    if score > xgbRfBest:
                        xgbRfBest = score
                        xgbRfStd = std
                        xgbRfNumTrees = perf["numTrees"]
                        xgbRfDepth = perf["depth"]
                elif perf["forestTypeName"] == "RandomForestClassifier" or perf["forestTypeName"] == "RandomForestRegressor":
                    if score > rfBest:
                        rfBest = score
                        rfStd = std
                        rfNumTrees = perf["numTrees"]
                        rfDepth= perf["depth"]
                elif perf["forestTypeName"] == "LGBMClassifier" or perf["forestTypeName"] == "LGBMRegressor":
                    if score > lgbBest:
                        lgbBest = score
                        lgbStd = std
                        lgbNumTrees = perf["numTrees"]
                        lgbDepth = perf["depth"]


        if perfKey == "allTestAcc":
            forestBest = 1.0 - forestBest
            fernBest = 1.0 - fernBest
            xgbBest = 1.0 - xgbBest
            xgbRfBest = 1.0 - xgbRfBest
            rfBest = 1.0 - rfBest
            lgbBest = 1.0 - lgbBest
                  
        print(f"{path} RandomHingeForest: numTrees = {numTrees}, depth = {treeDepth}, {perfKey} = {forestBest} +/- {forestStd}")
        print(f"{path} RandomHingeFern: numTrees = {numFerns}, depth = {fernDepth}, {perfKey} = {fernBest} +/- {fernStd}")
        print(f"{path} XGBClassifier: numTrees = {xgbNumTrees}, depth = {xgbDepth}, {perfKey} = {xgbBest} +/- {xgbStd}")
        print(f"{path} XGBRFClassifier: numTrees = {xgbRfNumTrees}, depth = {xgbRfDepth}, {perfKey} = {xgbRfBest} +/- {xgbRfStd}")
        print(f"{path} RandomForestClassifier: numTrees = {rfNumTrees}, depth = {rfDepth}, {perfKey} = {rfBest} +/- {rfStd}")
        print(f"{path} LGBMClassifier: numTrees = {lgbNumTrees}, depth = {lgbDepth}, {perfKey} = {lgbBest} +/- {lgbStd}")

if __name__ == "__main__":
    main(sys.argv[1:])

