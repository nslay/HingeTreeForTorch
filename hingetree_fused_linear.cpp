/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * March 2022
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
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <functional>

#include "torch/extension.h"
#include "caffe2/core/timer.h"
#include "HingeTreeCommon.h"
#include "MedianInit.h"
#include "GreedyInit.h"

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) {
  return torch::Tensor();
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) {
  return std::vector<torch::Tensor>();
}
#endif // !WITH_CUDA

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_cpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0])
    return torch::Tensor();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<RealType>() < RealType(0) || inOrdinals.max().item<RealType>() >= RealType(i64NumChannels))
    return torch::Tensor();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs
  
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

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  for (int64_t o = 0; o < i64OutChannels; ++o) {
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
          p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] = p_inLinearBias[o];
        }
      }
    }
  }

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::abs(std::get<1>(clKeyMarginTuple));
		
        for (int64_t o = 0; o < i64OutChannels; ++o) {
          const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

          for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
            p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m];
        }
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) // Not differentiable, ever!
    return std::vector<torch::Tensor>();
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2)
    return std::vector<torch::Tensor>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0])
    return std::vector<torch::Tensor>();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<torch::Tensor>();
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<RealType>() < RealType(0) || inOrdinals.max().item<RealType>() >= RealType(i64NumChannels))
    return std::vector<torch::Tensor>();

  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
  }
  
  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }

  // Sanity check on outDataGrad
  if (outDataGrad.sizes() != IntArrayRef(vSizes.data(), vSizes.size()))
    return std::vector<torch::Tensor>();
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  //const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(6);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
          
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
          
          const int64_t i64InputIndex = (int64_t)p_inOrdinals[j*i64NumDecisionsPerTree + treeIndex];
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = sign * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inDataGrad[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
  
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = -sign * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += scale * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[3] = inWeightsGrad;
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearWeightsGrad[o*i64NumTrees + j] += margin * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[4] = inLinearWeightsGrad;
  }

  if (bInLinearBiasGrad) {
    std::vector<IntArrayRef::value_type> vSumOver(vSizes.size()-1);
    vSumOver[0] = 0;
    std::iota(vSumOver.begin()+1, vSumOver.end(), 2);

    torch::Tensor inLinearBiasGrad = outDataGrad.sum(IntArrayRef(vSumOver.data(), vSumOver.size()));

#if 0
    torch::Tensor inLinearBiasGrad = torch::zeros_like(inLinearBias);
    RealType * const p_inLinearBiasGrad = inLinearBiasGrad.data_ptr<RealType>();

    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearBiasGrad[o] += p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeighytsNum + m]
            }
          }
        }
      }
    }
#endif

    vGradTensors[5] = inLinearBiasGrad;
  }

  return vGradTensors;
}

torch::Tensor hingetree_fused_linear_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || inData.dtype() != inOrdinals.dtype() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

