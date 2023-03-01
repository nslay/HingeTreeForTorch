# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# March 2023
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
import sys
import glob
import numpy as np
import SimpleITK as sitk
from rcc_common import LoadImage, LoadMask

def Gini(tpl):
    counts = tpl[2]
    return 4*(1-((counts/counts.sum())**2).sum())/3

def main():
    # Determined from ROC... 20% FPR
    diceThreshold=0.2
    dataRoot="/data/AIR/RCC/NiftiNew"
    probMapRoot="/data/AIR/layns/ProbMaps_unweighted_hingeforest_depth7_vggblock3_randomSplit1_epoch_549"
    leafMapRoot="/data/AIR/layns/LeafMaps_unweighted_hingeforest_depth7_vggblock3_randomSplit1_epoch_549"
    #leafMapRoot="/data/AIR/layns/LeafMaps_unweighted_hingeforest_depth7_vggblock3_randomSplit1_epoch_549_training"
    #testList=os.path.join(dataRoot, "all.txt")
    testList=os.path.join(dataRoot, "test_isbi2022_easyhard_randomSplit1.txt")
    #testList=os.path.join(dataRoot, "train_isbi2022_easyhard_randomSplit1.txt")
    numTrees=100

    with open(testList, mode="rt", newline='') as f:
        cases = [ case.strip() for case in f if len(case.strip()) > 0 ]

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    treeDict = dict()

    allCounts = None

    for case in cases:
        #labelMapFile=os.path.join(probMapRoot, case.replace("/","_") + ".nii.gz")
        gtFile=os.path.join(dataRoot, "Masks", case, "mask_aligned.nii.gz")

        #labelMap = LoadImage(labelMapFile)

        #if labelMap is None:
        #    print(f"Error: Could not load label map '{labelMapFile}'.", file=sys.stderr)

        gtMask = LoadMask(gtFile)

        if gtMask is None:
            print(f"Error: Could not load label map '{gtFile}'.", file=sys.stderr)

        #print(ccMask.GetSize())

        #print(f"{case}: object count = {ccFilter.GetObjectCount()}")

        npGtMask = sitk.GetArrayFromImage(gtMask)
        halfXDim = npGtMask.shape[2] // 2

        npRightGtMask = npGtMask[..., :halfXDim]
        npLeftGtMask = npGtMask[..., halfXDim:]

        if allCounts is None:
            allCounts = np.bincount(npGtMask[npGtMask < 4], minlength=4)
        else:
            allCounts += np.bincount(npGtMask[npGtMask < 4], minlength=4)

        for tree in range(numTrees):
            leafMapFile=os.path.join(leafMapRoot, f"{case.replace('/','_')}_leafmap{tree+1}.nii.gz")

            leafMap = LoadImage(leafMapFile)

            if leafMap is None:
                print(f"Error: Could not load leaf map '{leafMapFile}'.", file=sys.stderr)
                continue

            npLeafMap = sitk.GetArrayFromImage(leafMap)
            npLeafMap[npGtMask <= 0] = -1

            assert npLeafMap.shape == npGtMask.shape

            if tree not in treeDict:
                treeDict[tree] = dict()

            npRightLeafMap = npLeafMap[..., :halfXDim]
            npLeftLeafMap = npLeafMap[..., halfXDim:]

            for label in range(1,4):
                for npTmpLeafMap, npTmpGtMask in zip([npRightLeafMap, npLeftLeafMap], [npRightGtMask, npLeftGtMask]):
                    for leaf in np.unique(npTmpLeafMap[npTmpGtMask == label]):
                        if leaf < 0:
                            continue

                        AintB = np.logical_and(npTmpLeafMap == leaf, npTmpGtMask == label).sum()

                        #print(AintB)

                        if AintB == 0:
                            continue

                        A = (npTmpLeafMap == leaf).sum()
                        B = (npTmpGtMask == label).sum()

                        dice = 2*AintB/(A + B)


                        if dice <= diceThreshold:
                            continue

                        #print(dice)

                        #labels = np.unique(npGtMask[npLeafMap == leaf])

                        labels = npTmpGtMask[npTmpLeafMap == leaf]
                        labelCounts = np.bincount(labels[labels < 4], minlength=4)

                        if leaf not in treeDict[tree]:
                            treeDict[tree][leaf] = labelCounts
                        else:
                            #treeDict[tree][maxLeaf] = np.unique(np.concatenate((treeDict[tree][maxLeaf], gleasonScores),axis=0))
                            #treeDict[tree][leaf] = np.concatenate((treeDict[tree][leaf], labels),axis=0)
                            #print(labelCounts.shape)
                            #print(treeDict[tree][leaf].shape)
                            treeDict[tree][leaf] += labelCounts


    allLeaves = list()
    for tree in treeDict:
        for leaf in treeDict[tree]:
            allLeaves.append((tree, leaf, treeDict[tree][leaf]))

    #print(allLeaves)
    allLeaves.sort(key=Gini)

    #print(allLeaves[:10])
    for leaf in allLeaves:
        print(f"tree = {leaf[0]}, leaf = {leaf[1]}, counts = {leaf[2]}, gini = {Gini(leaf)}, majority = {leaf[2].argmax()}, count = {leaf[2].sum()}")

    print(f"\nallCounts = {allCounts}, mean = {allCounts/len(cases)}")

    """
    for tree in treeDict:
        for leaf in treeDict[tree]:
            labelCounts = treeDict[tree][leaf]

            #if gleasonScores.max() > 0:
            #    gleasonScores = gleasonScores[gleasonScores > 0]

            #uniqueLabels = np.unique(labels)
            #labelCounts = np.bincount(labels)[uniqueLabels]
            print(f"Info: tree = {tree}, leaf = {leaf}:")
            #print(f"Gleason scores: {uniqueLabels}")
            print(f"Counts: {labelCounts}\n")
    """

if __name__ == "__main__":
    main()

