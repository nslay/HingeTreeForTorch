/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * November 2020
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

#include "torch/extension.h"
#include "HingeTreeCommon.cuh"

#include <cuda.h>

// From: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// And from: https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

//#if __CUDA_ARCH__ < 600
#else
static inline __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace {

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ForwardKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    RealType * const d_out = d_outData + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l)
      d_out[l] = d_leafWeights[l] * margin;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ReachabilityKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, RealType *d_outCounts, 
    int iTreeDepth, int iThresholdStride, int iWeightsStride, int iNumTrees, int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < iNumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;
    RealType * const d_counts = d_outCounts + j*iWeightsStride;

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

        // leaf key, margin, ordinal index
        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

        const KeyType key = keyMarginTuple.leafKey;
        d_counts[key] += RealType(1);
      }
    }
  }
}

// Execute each example on one tree per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < iNumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*iThresholdStride;

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

        const KeyType key = keyMarginTuple.leafKey;
        const RealType signedMargin = keyMarginTuple.signedMargin;
        const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

        const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

        const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
        const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

        RealType tmpSum = RealType(0);
        for (int l = 0; l < iInnerWeightsNum; ++l)
          tmpSum += d_leafWeights[l] * d_outGradient[l];

        tmpSum *= -sign;

        d_thresholdsGradient[thresholdIndex] += tmpSum; // Do this just once
      }
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    RealType tmpSum = RealType(0);
    for (int l = 0; l < iInnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= -sign;

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once

    //for (int l = 0; l < iInnerWeightsNum; ++l)
      //d_thresholdsGradient[thresholdIndex] += -sign * d_leafWeights[l] * d_outGradient[l];
  }
}

// Execute each example on one tree per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < iNumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    for (int i = 0; i < iOuterNum; ++i) {
      for (int k = 0; k < iInnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

        // leaf key, margin, ordinal index
        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

        const KeyType key = keyMarginTuple.leafKey;
        const RealType signedMargin = keyMarginTuple.signedMargin;
        const RealType margin = std::abs(signedMargin);

        const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;
        RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*iWeightsStride + key)*iInnerWeightsNum;

        for (int l = 0; l < iInnerWeightsNum; ++l) {
          d_leafWeightsGradient[l] += margin * d_outGradient[l]; 
        }
      }
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*iWeightsStride + key)*iInnerWeightsNum;

    for (int l = 0; l < iInnerWeightsNum; ++l) {
      atomicAdd(d_leafWeightsGradient + l, margin * d_outGradient[l]); // Really bad!
      //d_leafWeightsGradient[l] += margin * d_outGradient[l];
    }
  }
}
// Execute all trees on one example per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Batch and inner indices
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < iOuterNum && k < iInnerDataNum) {
    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    for (int j = 0; j < iNumTrees; ++j) {
      const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
      const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

      // leaf key, margin, ordinal index
      const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

      const KeyType key = keyMarginTuple.leafKey;
      const RealType signedMargin = keyMarginTuple.signedMargin;
      const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
      const int iInputIndex = (int)d_ordinals[thresholdIndex];

      const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
      const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

      const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
      RealType tmpSum = RealType(0);

      for (int l = 0; l < iInnerWeightsNum; ++l)
        tmpSum += d_leafWeights[l] * d_outGradient[l];

      tmpSum *= sign;

      d_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += tmpSum; 
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const RealType *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int iTreeDepth, int iThresholdStride, int iWeightsStride, int iInnerWeightsNum, int iNumTrees, 
    int iOuterNum, int iNumChannels, int iInnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < iOuterNum && j < iNumTrees && k < iInnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*iThresholdStride;
    const RealType * const d_ordinals = d_inOrdinals + j*iThresholdStride;

    const RealType * const d_row = d_inData + ((i*iNumChannels + 0)*iInnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, iTreeDepth, iInnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
    const int iInputIndex = (int)d_ordinals[thresholdIndex];

    const RealType * const d_leafWeights = d_inWeights + (j*iWeightsStride + key)*iInnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
    RealType tmpSum = RealType(0);

    for (int l = 0; l < iInnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= sign;

    atomicAdd(d_inDataGradient + ((i*iNumChannels + iInputIndex)*iInnerDataNum + k), tmpSum); // Do this just once

    //d_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += tmpSum; // Do this just once
  }
}

} // end anonymous namespace

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return torch::Tensor();
  
  const int iNumTrees = inWeights.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);
  
  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return torch::Tensor();

  const int iBatchSize = inData.sizes()[0];
  const int iNumChannels = inData.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<RealType>() < RealType(0) || inOrdinals.max().to(torch::kCPU).item<RealType>() >= RealType(iNumChannels))
    return torch::Tensor();
 
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
  }

  if (inWeights.sizes().size() > 2) {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }
  
  torch::Tensor outData = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);
  
  RealType * const p_outData = outData.data_ptr<RealType>();
  
  int iInnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    iInnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int iInnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    iInnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const dim3 threadsPerBlock(8,16,4);
  const dim3 numBlocks((iBatchSize + threadsPerBlock.x-1)/threadsPerBlock.x, (iNumTrees + threadsPerBlock.y-1)/threadsPerBlock.y, (iInnerDataNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outData, 
    iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) // Not differentiable, ever!
    return std::vector<torch::Tensor>();
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<torch::Tensor>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<torch::Tensor>();
  
  const int iNumTrees = inWeights.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);
  
  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return std::vector<torch::Tensor>();
  
  const int iBatchSize = inData.sizes()[0];
  const int iNumChannels = inData.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<RealType>() < RealType(0) || inOrdinals.max().to(torch::kCPU).item<RealType>() >= RealType(iNumChannels))
    return std::vector<torch::Tensor>();

  int iInnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    iInnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int iInnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    iInnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  std::vector<torch::Tensor> vGradTensors(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    const dim3 threadsPerBlock(32,16);
    const dim3 numBlocks((iBatchSize + threadsPerBlock.x-1)/threadsPerBlock.x, (iInnerDataNum + threadsPerBlock.y-1)/threadsPerBlock.y);

    DeterministicBackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inDataGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    const dim3 threadsPerBlock(512);
    const dim3 numBlocks((iNumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

    DeterministicBackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inThresholdsGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    const dim3 threadsPerBlock(512);
    const dim3 numBlocks((iNumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

    DeterministicBackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outDataGrad, p_inWeightsGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) // Not differentiable, ever!
    return std::vector<torch::Tensor>();
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<torch::Tensor>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<torch::Tensor>();
  
  const int iNumTrees = inWeights.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);
  
  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return std::vector<torch::Tensor>();
  
  const int iBatchSize = inData.sizes()[0];
  const int iNumChannels = inData.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<RealType>() < RealType(0) || inOrdinals.max().to(torch::kCPU).item<RealType>() >= RealType(iNumChannels))
    return std::vector<torch::Tensor>();
 
  int iInnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    iInnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int iInnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    iInnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(8,16,4);
  const dim3 numBlocks((iBatchSize + threadsPerBlock.x-1)/threadsPerBlock.x, (iNumTrees + threadsPerBlock.y-1)/threadsPerBlock.y, (iInnerDataNum + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();
    
    BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inDataGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inThresholdsGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outDataGrad, p_inWeightsGrad, 
      iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iInnerWeightsNum, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return torch::Tensor();
  
  const int iNumTrees = inWeights.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);
  
  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return torch::Tensor();

  const int iBatchSize = inData.sizes()[0];
  const int iNumChannels = inData.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<RealType>() < RealType(0) || inOrdinals.max().to(torch::kCPU).item<RealType>() >= RealType(iNumChannels))
    return torch::Tensor();
 
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  torch::Tensor outCounts = torch::zeros(inWeights.sizes().slice(0,2), clOptions);
  
  RealType * const p_outCounts = outCounts.data_ptr<RealType>();
  
  int iInnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    iInnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), 1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const dim3 threadsPerBlock(512);
  const dim3 numBlocks((iNumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

  ReachabilityKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outCounts,
    iTreeDepth, iNumDecisionsPerTree, iNumLeavesPerTree, iNumTrees, iBatchSize, iNumChannels, iInnerDataNum);

  return outCounts;
}

template torch::Tensor hingetree_gpu_forward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_forward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_forward<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_forward<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template std::vector<torch::Tensor> hingetree_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template torch::Tensor hingetree_gpu_reachability<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_reachability<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_reachability<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_reachability<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

