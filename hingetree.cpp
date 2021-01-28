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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <functional>

#include "torch/extension.h"
#include "caffe2/core/timer.h"
#include "HingeTreeCommon.h"

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_cpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
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

  if (inOrdinals.min().item<RealType>() < RealType(0) || inOrdinals.max().item<RealType>() >= RealType(iNumChannels))
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

  for (int i = 0; i < iBatchSize; ++i) {
    for (int j = 0; j < iNumTrees; ++j) {
      for (int k = 0; k < iInnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k),
          p_inThresholds + (j*iNumDecisionsPerTree + 0), p_inOrdinals + (j*iNumDecisionsPerTree + 0), iTreeDepth, iInnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::get<1>(clKeyMarginTuple);
		
        for (int m = 0; m < iInnerWeightsNum; ++m)
          p_outData[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m] = std::abs(margin) * p_inWeights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m];
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
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

  if (inOrdinals.min().item<RealType>() < RealType(0) || inOrdinals.max().item<RealType>() >= RealType(iNumChannels))
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

    for (int i = 0; i < iBatchSize; ++i) {
      for (int j = 0; j < iNumTrees; ++j) {
        for (int k = 0; k < iInnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
            p_inThresholds + (j*iNumDecisionsPerTree + 0), p_inOrdinals + (j*iNumDecisionsPerTree + 0), iTreeDepth, iInnerDataNum);
          
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
          
          const int iInputIndex = (int)p_inOrdinals[j*iNumDecisionsPerTree + treeIndex];
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

          for (int m = 0; m < iInnerWeightsNum; ++m) {
            p_inDataGrad[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += sign * p_inWeights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] * p_outDataGrad[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    for (int i = 0; i < iBatchSize; ++i) {
      for (int j = 0; j < iNumTrees; ++j) {
        for (int k = 0; k < iInnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
            p_inThresholds + (j*iNumDecisionsPerTree + 0), p_inOrdinals + (j*iNumDecisionsPerTree + 0), iTreeDepth, iInnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
  
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));
  
          for (int m = 0; m < iInnerWeightsNum; ++m) {
            p_inThresholdsGrad[j*iNumDecisionsPerTree + treeIndex] += -sign * p_inWeights[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] * p_outDataGrad[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    for (int i = 0; i < iBatchSize; ++i) {
      for (int j = 0; j < iNumTrees; ++j) {
        for (int k = 0; k < iInnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k), 
            p_inThresholds + (j*iNumDecisionsPerTree + 0), p_inOrdinals + (j*iNumDecisionsPerTree + 0), iTreeDepth, iInnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
  
          for (int m = 0; m < iInnerWeightsNum; ++m) {
            p_inWeightsGrad[(j*iNumLeavesPerTree + leafKey)*iInnerWeightsNum + m] += std::abs(margin) * p_outDataGrad[((i*iNumTrees + j)*iInnerDataNum + k)*iInnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return torch::Tensor(); }

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) { return std::vector<torch::Tensor>(); }

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) { return std::vector<torch::Tensor>(); }

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return torch::Tensor(); }
#endif // !WITH_CUDA

template<typename RealType, typename TreeTraitsType>
std::vector<bool> hingetree_cpu_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<bool>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<bool>();

  const int iNumTrees = inThresholds.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);

  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return std::vector<bool>();
  
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();

  std::vector<bool> vGood(iNumTrees);

  for (int j = 0; j < iNumTrees; ++j)
   vGood[j] = TreeTraitsType::CheckThresholds(p_inThresholds + j*iNumDecisionsPerTree, p_inOrdinals + j*iNumDecisionsPerTree, iTreeDepth);

  return vGood;
}

template<typename RealType, typename TreeTraitsType>
std::vector<bool> hingetree_cpu_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<bool>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<bool>();

  const int iNumTrees = inThresholds.sizes()[0];
  const int iNumLeavesPerTree = inWeights.sizes()[1];
  const int iNumDecisionsPerTree = inThresholds.sizes()[1];
  const int iTreeDepth = TreeTraitsType::ComputeDepth(iNumLeavesPerTree);

  if (inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(iTreeDepth))
    return std::vector<bool>();
  
  RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();

  std::vector<bool> vChangesMade(iNumTrees);

  for (int j = 0; j < iNumTrees; ++j)
   vChangesMade[j] = TreeTraitsType::FixThresholds(p_inThresholds + j*iNumDecisionsPerTree, p_inOrdinals + j*iNumDecisionsPerTree, iTreeDepth);

  return vChangesMade;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_cpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;

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
 
  if (inOrdinals.min().item<RealType>() < RealType(0) || inOrdinals.max().item<RealType>() >= RealType(iNumChannels))
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
  
  for (int i = 0; i < iBatchSize; ++i) {
    for (int j = 0; j < iNumTrees; ++j) {
      for (int k = 0; k < iInnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*iNumChannels + 0)*iInnerDataNum + k),
          p_inThresholds + (j*iNumDecisionsPerTree + 0), p_inOrdinals + (j*iNumDecisionsPerTree + 0), iTreeDepth, iInnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        p_outCounts[j*iNumLeavesPerTree + leafKey] += RealType(1);
      }
    }
  }
  
  return outCounts;
}

torch::Tensor hingetree_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> hingetree_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> hingefern_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<bool> hingetree_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dtype() != inOrdinals.dtype() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      return hingetree_cpu_check_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      return hingetree_cpu_check_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingefern_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dtype() != inOrdinals.dtype() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      return hingetree_cpu_check_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      return hingetree_cpu_check_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingetree_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dtype() != inOrdinals.dtype() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingefern_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dtype() != inOrdinals.dtype() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

torch::Tensor hingetree_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

torch::Tensor hingefern_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

// inData is the size of minibatch to test... and a hint of which device to test!
torch::Tensor hingetree_speedtest(torch::Tensor inData, bool bDeterministic) {
  constexpr int iNumBatches = 100;
  constexpr int iNumTreeSteps = 10;
  constexpr int iMaxDepth = 12;
  typedef bleak::HingeTreeCommon<float> TreeTraitsType; // Type doesn't matter... just used to deduce dimensions of tensors

  if (inData.dim() < 2)
    return torch::Tensor();

  torch::Tensor clTimings = torch::zeros({4, iNumTreeSteps*iMaxDepth}, torch::TensorOptions().dtype(torch::kFloat32)); // numTrees, depth, fotward timing, backward timing

  float * const p_fNumTrees = clTimings.data_ptr<float>();
  float * const p_fDepths = p_fNumTrees + (iNumTreeSteps*iMaxDepth);
  float * const p_fForwardTimings = p_fDepths + (iNumTreeSteps*iMaxDepth);
  float * const p_fBackwardTimings = p_fForwardTimings + (iNumTreeSteps*iMaxDepth);

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  int t = 0; // Timings index
  for (int s = 0; s < iNumTreeSteps; ++s) {
    const int iNumTrees = (1 << s);
    
    for (int iDepth = 1; iDepth <= 12; ++iDepth, ++t) {
      torch::Tensor inThresholds = torch::rand( { iNumTrees, TreeTraitsType::GetThresholdCount(iDepth) }, clOptions);
      inThresholds *= 6.0f;
      inThresholds -= 3.0f;

      torch::Tensor inWeights = torch::randn( { iNumTrees, TreeTraitsType::GetLeafCount(iDepth) }, clOptions);
      torch::Tensor inOrdinals = torch::randint(0, inData.sizes()[1], { iNumTrees, TreeTraitsType::GetThresholdCount(iDepth) }, clOptions);

      torch::Tensor outData;

      p_fNumTrees[t] = (float)iNumTrees;
      p_fDepths[t] = (float)iDepth;

      {
        caffe2::Timer clForwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          outData = hingetree_forward(inData, inThresholds, inOrdinals, inWeights);

        const float fAverage = clForwardTimer.MilliSeconds() / iNumBatches;
        p_fForwardTimings[t] = fAverage;

        std::cout << "hingetree_forward: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }

      torch::Tensor outDataGrad = torch::ones_like(outData);

      if (bDeterministic) {
        caffe2::Timer clBackwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          hingetree_backward_deterministic(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / iNumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingetree_backward_deterministic: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }
      else {
        caffe2::Timer clBackwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          hingetree_backward(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / iNumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingetree_backward: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }
    }
  }

  return clTimings;
}

torch::Tensor hingefern_speedtest(torch::Tensor inData, bool bDeterministic) {
  constexpr int iNumBatches = 100;
  constexpr int iNumTreeSteps = 10;
  constexpr int iMaxDepth = 12;
  typedef bleak::HingeFernCommon<float> TreeTraitsType; // Type doesn't matter... just used to deduce dimensions of tensors

  if (inData.dim() < 2)
    return torch::Tensor();

  torch::Tensor clTimings = torch::zeros({4, iNumTreeSteps*iMaxDepth}, torch::TensorOptions().dtype(torch::kFloat32)); // numTrees, depth, fotward timing, backward timing

  float * const p_fNumTrees = clTimings.data_ptr<float>();
  float * const p_fDepths = p_fNumTrees + (iNumTreeSteps*iMaxDepth);
  float * const p_fForwardTimings = p_fDepths + (iNumTreeSteps*iMaxDepth);
  float * const p_fBackwardTimings = p_fForwardTimings + (iNumTreeSteps*iMaxDepth);

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  int t = 0; // Timings index
  for (int s = 0; s < iNumTreeSteps; ++s) {
    const int iNumTrees = (1 << s);
    
    for (int iDepth = 1; iDepth <= 12; ++iDepth, ++t) {
      torch::Tensor inThresholds = torch::rand( { iNumTrees, TreeTraitsType::GetThresholdCount(iDepth) }, clOptions);
      inThresholds *= 6.0f;
      inThresholds -= 3.0f;

      torch::Tensor inWeights = torch::randn( { iNumTrees, TreeTraitsType::GetLeafCount(iDepth) }, clOptions);
      torch::Tensor inOrdinals = torch::randint(0, inData.sizes()[1], { iNumTrees, TreeTraitsType::GetThresholdCount(iDepth) }, clOptions);

      torch::Tensor outData;

      p_fNumTrees[t] = (float)iNumTrees;
      p_fDepths[t] = (float)iDepth;

      {
        caffe2::Timer clForwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          outData = hingefern_forward(inData, inThresholds, inOrdinals, inWeights);

        const float fAverage = clForwardTimer.MilliSeconds() / iNumBatches;
        p_fForwardTimings[t] = fAverage;

        std::cout << "hingefern_forward: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }

      torch::Tensor outDataGrad = torch::ones_like(outData);

      if (bDeterministic) {
        caffe2::Timer clBackwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          hingefern_backward_deterministic(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / iNumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingefern_backward_deterministic: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }
      else {
        caffe2::Timer clBackwardTimer;

        for (int iBatch = 0; iBatch < iNumBatches; ++iBatch)
          hingefern_backward(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / iNumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingefern_backward: numBatches = " << iNumBatches << ", numTrees = " << iNumTrees << ", depth = " << iDepth << ": " << fAverage << " ms per batch." << std::endl;
      }
    }
  }

  return clTimings;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tree_forward", &hingetree_forward, "Hinge tree forward");
  m.def("tree_backward", &hingetree_backward, "Hinge tree backward");
  m.def("tree_backward_deterministic", &hingetree_backward_deterministic, "Deterministic hinge tree backward");
  m.def("tree_check_thresholds", &hingetree_check_thresholds, "Check logical consistency of thresholds and ordinals in trees. Returns true if logically consistent.");
  m.def("tree_fix_thresholds", &hingetree_fix_thresholds, "Fix logical consistency of thresholds and ordinals in trees. Returns true if changes were made.");
  m.def("tree_reachability", &hingetree_reachability, "Compute leaf visit counts. Returns per-leaf visit counts.");
  m.def("tree_speedtest", &hingetree_speedtest, "Compute forward/backward timings on a given mini-batch. Returns 2D tensor with rows: numTrees, depth, forward timings, backward timings.");
  
  m.def("fern_forward", &hingefern_forward, "Hinge fern forward");
  m.def("fern_backward", &hingefern_backward, "Hinge fern backward");
  m.def("fern_backward_deterministic", &hingefern_backward_deterministic, "Deterministic hinge fern backward");
  m.def("fern_check_thresholds", &hingefern_check_thresholds, "Check logical consistency of thresholds and ordinals in ferns. Returns true if logically consistent.");
  m.def("fern_fix_thresholds", &hingefern_fix_thresholds, "Fix logical consistency of thresholds and ordinals in ferns. Returns true if changes were made.");
  m.def("fern_reachability", &hingefern_reachability, "Compute leaf visit counts. Returns per-leaf visit counts.");
  m.def("fern_speedtest", &hingefern_speedtest, "Compute forward/backward timings on a given mini-batch. Returns 2D tensor with rows: numTrees, depth, forward timings, backward timings.");
}

