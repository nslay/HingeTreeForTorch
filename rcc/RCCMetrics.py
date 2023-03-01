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

import SimpleITK as sitk
import numpy as np
import functools
import operator
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff
#from sklearn.metrics import roc_curve

# NOTE: See reference
# Heimann, Tobias, et al. "Comparison and evaluation of methods for liver segmentation from CT datasets." IEEE transactions on medical imaging 28.8 (2009): 1251-1265.

def _arg_median(u):
    idx = np.argsort(u)[u.size // 2]
    return u[idx], idx

def _hausdorff_distance(u,v):
    return max(directed_hausdorff(u,v)[0], directed_hausdorff(v,u)[0])

def _average_symmetric_surface_distance(u,v):
    treeU = KDTree(u)
    treeV = KDTree(v)

    dUToV = treeU.query(v)[0]
    dVToU = treeV.query(u)[0]

    return np.concatenate((dVToU, dUToV), axis=0).mean()

def _dice_similarity_coefficient(u,v):
    UintV = np.logical_and(u,v).sum()
    sumUV = u.sum() + v.sum()

    return 1.0 if sumUV == 0.0 else 2.0*UintV/sumUV

def _jaccard_index(u,v):
    UintV = np.logical_and(u,v).sum()
    sumUV = u.sum() + v.sum()

    return 1.0 if sumUV == 0.0 else UintV/(sumUV - UintV)

# NOTE: U is GT
def _overlap_ratio(u,v):
    UintV = np.logical_and(u,v).sum()
    sumU = u.sum()

    return 1.0 if sumU == 0.0 else UintV/sumU

def _connected_components(npMask):
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    ccMask = ccFilter.Execute(sitk.GetImageFromArray(npMask))
    objCount = ccFilter.GetObjectCount()

    return sitk.GetArrayFromImage(ccMask), objCount

def _remove_small_connected_components(npMask, small=15):
    npCcMask, objectCount = _connected_components(npMask)

    for label in range(1, objectCount+1):
        if (npCcMask == label).sum() < small:
            npMask[npCcMask == label] = 0

    return npMask

def _extract_surface_points(mask):
    erode = sitk.BinaryErodeImageFilter()
    erode.SetForegroundValue(1)
    erode.SetBackgroundValue(0)
    erode.SetKernelType(sitk.sitkBall)
    erode.SetKernelRadius(1)

    R = np.reshape(mask.GetDirection(), [3,3])
    S = np.array(mask.GetSpacing())
    T = np.array(mask.GetOrigin())

    erodedMask = erode.Execute(mask)

    npMask = sitk.GetArrayViewFromImage(mask)
    npErodedMask = sitk.GetArrayViewFromImage(erodedMask)

    # NOTE: Make it X, Y, Z instead of Z, Y, X
    points = np.argwhere(npMask - npErodedMask)[:, ::-1]
    #print((points*S).shape)
    points = np.inner(points*S, R.T) + T

    return points

def evaluate(gtMask, segMask, probMap, overlap_threshold=0.5, use_half_logic=True):
    npGtMask = sitk.GetArrayFromImage(gtMask) 
    npSegMask = sitk.GetArrayFromImage(segMask)
    npProbMap = sitk.GetArrayViewFromImage(probMap)

    voxelVolume = functools.reduce(operator.mul, gtMask.GetSpacing())

    if use_half_logic:
        halfX = npGtMask.shape[-1] // 2

        npRightGtMask, npLeftGtMask = npGtMask[..., :halfX], npGtMask[..., halfX:]
        npRightSegMask, npLeftSegMask = npSegMask[..., :halfX], npSegMask[..., halfX:]
        npRightProbMap, npLeftProbMap = npProbMap[..., :halfX, :], npProbMap[..., halfX:, :]

        rightCount, leftCount = (npRightGtMask > 0).sum(), (npLeftGtMask > 0).sum()

        small=5**3

        assert rightCount > small or leftCount > small, "Nothing annotated?"

        if rightCount > small and leftCount > small:
            print("Both")
            return evaluate(gtMask, segMask, probMap, overlap_threshold=overlap_threshold, use_half_logic=False)
        elif rightCount > small:
            print("Right")
            rightGtMask = sitk.GetImageFromArray(npRightGtMask)
            rightSegMask = sitk.GetImageFromArray(npRightSegMask)
            rightProbMap = sitk.GetImageFromArray(npRightProbMap, isVector=True)

            rightGtMask.SetSpacing(gtMask.GetSpacing())
            rightSegMask.SetSpacing(gtMask.GetSpacing())
            rightProbMap.SetSpacing(gtMask.GetSpacing())

            return evaluate(rightGtMask, rightSegMask, rightProbMap, overlap_threshold=overlap_threshold, use_half_logic=False)
        elif leftCount > small:
            print("Left")
            leftGtMask = sitk.GetImageFromArray(npLeftGtMask)
            leftSegMask = sitk.GetImageFromArray(npLeftSegMask)
            leftProbMap = sitk.GetImageFromArray(npLeftProbMap, isVector=True)

            leftGtMask.SetSpacing(gtMask.GetSpacing())
            leftSegMask.SetSpacing(gtMask.GetSpacing())
            leftProbMap.SetSpacing(gtMask.GetSpacing())

            return evaluate(leftGtMask, leftSegMask, leftProbMap, overlap_threshold=overlap_threshold, use_half_logic=False)

        return None

    metricsByLabel = [None]*4

    # Mask out unknowns
    npSegMask[npGtMask == 255] = 0
    npGtMask[npGtMask == 255] = 0

    backgroundDetections = []

    npProbMapForTumor = npProbMap[..., 3]

    for label in [1,2,3]:
        npGtMaskForLabel = (npGtMask == label).astype(np.uint8)
        npSegMaskForLabel = (npSegMask == label).astype(np.uint8)
        npProbMapForLabel = npProbMap[..., label]

        npGtMaskForLabel = _remove_small_connected_components(npGtMaskForLabel)
        npSegMaskForLabel = _remove_small_connected_components(npSegMaskForLabel)

        npCcGtMaskForLabel, gtObjectCountForLabel = _connected_components(npGtMaskForLabel)
        npCcSegMaskForLabel, segObjectCountForLabel = _connected_components(npSegMaskForLabel)

        #if label == 3:
        #    ccGtMaskForLabel = sitk.GetImageFromArray(npCcGtMaskForLabel)
        #    ccGtMaskForLabel.CopyInformation(gtMask)
        #    sitk.WriteImage(ccGtMaskForLabel, "ccGtMaskForTumor.nii.gz", useCompression=True)

        remainingSegCcLabels = set(range(1,segObjectCountForLabel+1))
        missedGtCcLabels = []

        metrics = { "label": label }

        maskDsc = _dice_similarity_coefficient(npGtMaskForLabel, npSegMaskForLabel)
        metrics['mask_dsc'] = maskDsc

        detections = []

        for gtCcLabel in range(1, gtObjectCountForLabel+1):
            incidentSegCcLabels = np.unique(npCcSegMaskForLabel[npCcGtMaskForLabel == gtCcLabel])
            incidentSegCcLabels = incidentSegCcLabels[incidentSegCcLabels > 0]

            volume = (npCcGtMaskForLabel == gtCcLabel).sum()*voxelVolume
            metrics.setdefault('gt_volume', []).append(volume)

            if incidentSegCcLabels.size == 0:
                missedGtCcLabels.append(gtCcLabel)
                #score, idx = _arg_median(npProbMapForLabel[npCcGtMaskForLabel == gtCcLabel])
                #tumorScore = npProbMapForTumor[npCcGtMaskForLabel == gtCcLabel][idx]
                tumorScore = np.median(npProbMapForTumor[npCcGtMaskForLabel == gtCcLabel])
                #detections.append((0.0, 1))
                detections.append((tumorScore, 1))
                continue

            npTmpSegMask = np.zeros_like(npSegMaskForLabel)

            segLabelsToDiscard = set()

            for segCcLabel in incidentSegCcLabels:
                segLabelsToDiscard.add(segCcLabel)
                npTmpSegMask = np.logical_or(npTmpSegMask, (npCcSegMaskForLabel == segCcLabel))

            overlap = _overlap_ratio((npCcGtMaskForLabel == gtCcLabel), npTmpSegMask)

            if overlap < overlap_threshold:
                missedGtCcLabels.append(gtCcLabel)
                #score, idx = _arg_median(npProbMapForLabel[npCcGtMaskForLabel == gtCcLabel])
                #tumorScore = npProbMapForTumor[npCcGtMaskForLabel == gtCcLabel][idx]
                tumorScore = np.median(npProbMapForTumor[npCcGtMaskForLabel == gtCcLabel])
                #detections.append((0.0, 1))
                detections.append((tumorScore, 1))
                continue

            dsc = _dice_similarity_coefficient((npCcGtMaskForLabel == gtCcLabel), npTmpSegMask)

            remainingSegCcLabels -= segLabelsToDiscard

            #score = np.median(npProbMapForLabel[np.logical_and(npCcGtMaskForLabel == gtCcLabel, npTmpSegMask)])
            #score, idx = _arg_median(npProbMapForLabel[np.logical_and(npCcGtMaskForLabel == gtCcLabel, npTmpSegMask)])
            #tumorScore = npProbMapForTumor[np.logical_and(npCcGtMaskForLabel == gtCcLabel, npTmpSegMask)][idx]
            #tumorScore = np.median(npProbMapForTumor[np.logical_and(npCcGtMaskForLabel == gtCcLabel, npTmpSegMask)])
            tumorScore = np.median(npProbMapForTumor[npCcGtMaskForLabel == gtCcLabel])
            #score = np.percentile(npProbMapForLabel[np.logical_and(npCcGtMaskForLabel == gtCcLabel, npTmpSegMask)], 90)
            
            # We're only interested in tumor scores and we need to know if these are tumors or benign structures
            detections.append((tumorScore, 1))

            metrics.setdefault('overlap', []).append(overlap)
            metrics.setdefault('dsc', []).append(dsc)

            tmpGtMask = sitk.GetImageFromArray((npCcGtMaskForLabel == gtCcLabel).astype(np.uint8))
            tmpGtMask.CopyInformation(gtMask)
            
            tmpSegMask = sitk.GetImageFromArray(npTmpSegMask.astype(np.uint8))
            tmpSegMask.CopyInformation(tmpGtMask)

            npGtPoints = _extract_surface_points(tmpGtMask)
            npSegPoints = _extract_surface_points(tmpSegMask)

            assd = _average_symmetric_surface_distance(npGtPoints, npSegPoints)
            hd = _hausdorff_distance(npGtPoints, npSegPoints)

            metrics.setdefault('assd', []).append(assd)
            metrics.setdefault('hd', []).append(hd)
            
        for segCcLabel in remainingSegCcLabels:
            #score = np.median(npProbMapForLabel[npCcSegMaskForLabel == segCcLabel])
            #score, idx = _arg_median(npProbMapForLabel[npCcSegMaskForLabel == segCcLabel])
            #tumorScore = npProbMapForTumor[npCcSegMaskForLabel == segCcLabel][idx]
            tumorScore = np.median(npProbMapForTumor[npCcSegMaskForLabel == segCcLabel])
            #score = np.percentile(npProbMapForLabel[npCcSegMaskForLabel == segCcLabel], 90)
            detections.append((tumorScore, 0))

        metrics["gt_object_count"] = gtObjectCountForLabel
        metrics["seg_object_count"] = segObjectCountForLabel
        metrics["gt_missed_count"] = len(missedGtCcLabels)
        metrics["seg_extra_count"] = len(remainingSegCcLabels)

        if label == 3:
            metrics["tumor_detections"] = detections + backgroundDetections
        else: 
            backgroundDetections += [ (pair[0], 0) for pair in detections if pair[1] == 1 ] # Only non-tumor benign structures count!

        metricsByLabel[label] = metrics

    return metricsByLabel

if __name__ == "__main__":
    segRoot="/data/AMPrj/AllImages/Output/ProbMaps3D_combined_nondeterministic_hingeforest_depth7_vggblock3_3d_adamw_perChannelShift_moreData2_randomSplit4_epoch_799"
    gtRoot="/data/AMPrj/AllImages/NiftiEverything"
    case="0040-Subject-01332/0040-24972"

    patientId, accNumber = case.split("/")

    import os
    from rcc_common import LoadMask

    print(f"Processing {case} ...")

    gtMaskPath = os.path.join(gtRoot, "Masks", patientId, accNumber, "mask_aligned.nii.gz")
    segMaskPath = os.path.join(segRoot, f"{patientId}_{accNumber}_label.nii.gz")
    probMapPath = os.path.join(segRoot, f"{patientId}_{accNumber}.nii.gz")

    gtMask = LoadMask(gtMaskPath)
    segMask = sitk.ReadImage(segMaskPath)
    probMap = sitk.ReadImage(probMapPath)

    metricsByLabel = evaluate(gtMask, segMask, probMap)
    print(metricsByLabel)

    #x = np.array([[1,2,3], [2,5,4]])
    #x = np.array([[2,5,4]])
    #y = x.copy()
    #y = np.array([[1,2,3]])
    #print(_hausdorff_distance(x, y))
    #print(_average_symmetric_surface_distance(x, y))

    

