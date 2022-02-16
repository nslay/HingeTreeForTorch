/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * April 2021
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

#pragma once

#ifndef GREEDYINIT_H
#define GREEDYINIT_H

#include <iostream>
#include <algorithm>
#include <type_traits>
#include <numeric>
#include <iterator>
#include <functional>
#include <limits>
#include <unordered_set>
#include <random>
#include "MedianInit.h"

namespace {
// TODO: Add seed flag to python functions
std::mt19937_64 clGenerator;
} // end anonymous namespace

template<typename RealType>
RealType Gini(const std::vector<size_t> &vCounts) {
  if (vCounts.size() == 1)
    return RealType(-1);

  const size_t total = std::accumulate(vCounts.begin(), vCounts.end(), (size_t)0);

  if (total == 0)
    return RealType(-1);

  RealType gini = RealType();

  for (size_t count : vCounts) {
    const RealType p = RealType(count) / RealType(total);
    gini += p*p;
  }

  return RealType(vCounts.size())*(RealType(1) - gini)/RealType(vCounts.size()-1); // Normalized to take on a maximum value of 1
}

// i64NumChannels NOT including the extra label
template<typename RealType>
RealType OptimalSplitClassification(Vertex<RealType> &stVertex, int64_t i64NumChannels, int64_t i64NumClasses, int64_t i64Stride) {
  if (!stVertex.p_begin)
    return RealType(-1);
  
  const size_t sampleSize = std::distance(stVertex.p_begin, stVertex.p_end);

  if (sampleSize < 2 || i64NumClasses < 2)
    return RealType(-1);

  std::vector<size_t> vAllCounts(i64NumClasses, 0);

  for (auto itr = stVertex.p_begin; itr != stVertex.p_end; ++itr) {
    const int64_t i64Label = (int64_t)((*itr)[i64NumChannels*i64Stride]);

    if (i64Label < 0 || i64Label >= i64NumClasses)
      return RealType(-1);

    ++vAllCounts[i64Label];
  }

  {
    size_t nonZeroCount = 0;
    for (size_t count : vAllCounts) {
      if (count > 0)
        ++nonZeroCount;
    }

    if (nonZeroCount < 2)
      return RealType(-1);
  }

  std::shuffle(stVertex.p_begin, stVertex.p_end, clGenerator);
  const size_t reservoirSize = std::min((size_t)1000, sampleSize);

  const int64_t i64NumFeatures = (int64_t)std::max(1.0, std::sqrt((double)i64NumChannels));
  std::vector<RealType> vThresholds;
  std::vector<std::pair<RealType, int64_t>> vFeaturesAndLabels(reservoirSize);

  std::vector<int64_t> vOrdinals(i64NumChannels);
  std::iota(vOrdinals.begin(), vOrdinals.end(), (int64_t)0);

  std::shuffle(vOrdinals.begin(), vOrdinals.end(), clGenerator);
  vOrdinals.resize(i64NumFeatures);

  vThresholds.reserve(reservoirSize);
  std::vector<std::vector<size_t>> vBinCounts;

  stVertex.i64Ordinal = vOrdinals[0];
  RealType maxGain = RealType(-1);

  const RealType allPurity = Gini<RealType>(vAllCounts);

  for (int64_t c : vOrdinals) {
    //std::transform(stVertex.p_begin, stVertex.p_end, vFeaturesAndLabels.begin(),
    std::transform(stVertex.p_begin, stVertex.p_begin + reservoirSize, vFeaturesAndLabels.begin(),
     [&c, &i64Stride, &i64NumChannels](const RealType *a) -> std::pair<RealType, int64_t> {
       return std::make_pair(a[c*i64Stride], (int64_t)a[i64NumChannels*i64Stride]);
     });

    std::sort(vFeaturesAndLabels.begin(), vFeaturesAndLabels.end(),
      [](const std::pair<RealType, int64_t> &a, const std::pair<RealType, int64_t> &b) -> bool {
        return a.first < b.first;
      });

    vThresholds.clear();
    for (size_t i = 1; i < vFeaturesAndLabels.size(); ++i) {
      if (vFeaturesAndLabels[i].first - vFeaturesAndLabels[i-1].first > 1e-5)
        vThresholds.push_back((vFeaturesAndLabels[i].first + vFeaturesAndLabels[i-1].first)/2);
    }

    vBinCounts.resize(vThresholds.size());
    for (std::vector<size_t> &vCounts : vBinCounts) {
      vCounts.resize(i64NumClasses);
      std::fill(vCounts.begin(), vCounts.end(), 0);
    }

    for (const auto &stPair : vFeaturesAndLabels) {
      const RealType value = stPair.first;
      const int64_t i64Label = stPair.second;

      for (size_t i = 0; i < vThresholds.size() && value > vThresholds[i]; ++i)
        ++vBinCounts[i][i64Label];
    }

    for (size_t i = 0; i < vBinCounts.size(); ++i) {
      const size_t rightSampleSize = std::accumulate(vBinCounts[i].begin(), vBinCounts[i].end(), (size_t)0);

      if (rightSampleSize == 0 || rightSampleSize == reservoirSize)
        continue;

      RealType gain = allPurity;

      const RealType rightPurity = Gini<RealType>(vBinCounts[i]);
      gain -= RealType(rightSampleSize) * rightPurity / RealType(reservoirSize);

      std::transform(vAllCounts.begin(), vAllCounts.end(), vBinCounts[i].begin(), vBinCounts[i].begin(), std::minus<size_t>());

      const RealType leftPurity = Gini<RealType>(vBinCounts[i]);
      const size_t leftSampleSize = reservoirSize - rightSampleSize;

      gain -= RealType(leftSampleSize) * leftPurity / RealType(reservoirSize);
      //std::cout << "gain = " << gain << std::endl;
      gain = std::max(RealType(0), gain);

      if (gain > maxGain) {
        stVertex.i64Ordinal = c;
        stVertex.threshold = vThresholds[i];
        maxGain = gain;
      }
    }
  }
 
  return maxGain;
}

// NOTE: kInt64 should match torch.long
template<typename RealType, typename TreeTraitsType>
bool InitGreedySplitsClassification(std::vector<std::vector<Vertex<RealType>>> &vTrees, torch::Tensor inData, torch::Tensor inLabels, torch::Tensor inWeights) {
  typedef c10::IntArrayRef IntArrayRef;
  typedef Vertex<RealType> VertexType;

  static_assert(std::is_same<bleak::HingeTreeCommon<RealType>, TreeTraitsType>::value, "Only HingeTreeCommon is supported.");

  if (inData.dim() < 2 || inWeights.dim() < 2 || inLabels.dim() < 1 || inLabels.scalar_type() != torch::kInt64) 
    return false;

  if (inWeights.sizes()[0] != (int)vTrees.size() || inLabels.sizes()[0] != inData.sizes()[0] || inLabels.sizes().size()+1 != inData.sizes().size())
    return false;

  if (inLabels.min().item<int64_t>() < 0)
    return false;

  for (size_t i = 1; i < inLabels.sizes().size(); ++i) {
    if (inLabels.sizes()[i] != inData.sizes()[i+1])
      return false;
  }

  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = TreeTraitsType::GetThresholdCount(i64TreeDepth);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || (int64_t)vTrees[0].size() != i64NumDecisionsPerTree)
    return false;

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const int64_t * const p_inLabels = inLabels.data_ptr<int64_t>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  // Sanity check ordinals
  for (int64_t i = 0; i < i64NumTrees; ++i) {
    const std::vector<VertexType> &vVertices = vTrees[i];

    for (int64_t j = 0; j < i64NumDecisionsPerTree; ++j) {
      const VertexType &stVertex = vVertices[j];

      if (stVertex.i64Ordinal < 0 || stVertex.i64Ordinal >= i64NumChannels)
        return false;
    }
  }

  std::vector<RealType> vDataWithLabels(i64BatchSize*(i64NumChannels+1)*i64InnerDataNum);

  if (i64InnerDataNum > 1) {
    for (int64_t b = 0; b < i64BatchSize; ++b) {
      for (int64_t c = 0; c < i64NumChannels; ++c) {
        std::copy_n(p_inData + (b*i64NumChannels + c)*i64InnerDataNum, i64InnerDataNum, vDataWithLabels.data() + (b*(i64NumChannels+1) + c)*i64InnerDataNum);
      }

      std::copy_n(p_inLabels + b*i64InnerDataNum, i64InnerDataNum, vDataWithLabels.data() + (b*(i64NumChannels+1) + i64NumChannels)*i64InnerDataNum);
    }
  }
  else {
    for (int64_t b = 0; b < i64BatchSize; ++b) {
      std::copy_n(p_inData + b*i64NumChannels, i64NumChannels, vDataWithLabels.data() + b*(i64NumChannels+1));
      vDataWithLabels[b*(i64NumChannels+1) + i64NumChannels] = p_inLabels[b];
    }
  }

  // Make row pointers
  std::vector<const RealType *> vRows;
  vRows.reserve(i64BatchSize*i64InnerDataNum);

  for (int64_t b = 0; b < i64BatchSize; ++b) {
    for (int64_t k = 0; k < i64InnerDataNum; ++k) {
      vRows.push_back(vDataWithLabels.data() + ((b*(i64NumChannels+1) + 0)*i64InnerDataNum + k));
    }
  }

  // Setup root verices
  for (int64_t i = 0; i < i64NumTrees; ++i) {
    std::vector<VertexType> &vVertices = vTrees[i];
    vVertices[0].p_begin = vRows.data();
    vVertices[0].p_end = vVertices[0].p_begin + vRows.size();
  }

  const int64_t i64NumClasses = inLabels.max().item<int64_t>()+1;

  if (i64NumClasses < 2)
    return false;

  for (auto &vVertices : vTrees) {
    //std::cout << "Processing tree " << (&vVertices - &vTrees[0]) << std::endl;

    for (size_t i = 0; i < vVertices.size(); ++i) {
      auto &stVertex = vVertices[i];

      const RealType gain = OptimalSplitClassification<RealType>(stVertex, i64NumChannels, i64NumClasses, i64InnerDataNum);

      //std::cout << "optimal gain = " << gain << std::endl;

      if (gain < 0) {
        if (i == 0)
          return false;

        auto &stParent = vVertices[(i-1)/2];

        stVertex.i64Ordinal = stParent.i64Ordinal;
        stVertex.threshold = std::numeric_limits<RealType>::max();

        continue;
      }

      if (2*i + 2 < vVertices.size()) {
        auto midItr = std::partition(stVertex.p_begin, stVertex.p_end,
          [&stVertex, &i64InnerDataNum](const RealType *a) {
            return a[stVertex.i64Ordinal*i64InnerDataNum] > stVertex.threshold;
          });

        auto &stLeftVertex = vVertices[2*i+1];
        auto &stRightVertex = vVertices[2*i+2];

        stLeftVertex.p_begin = midItr;
        stLeftVertex.p_end = stVertex.p_end;

        stRightVertex.p_begin = stVertex.p_begin;
        stRightVertex.p_end = midItr;
      }
    }
  }

  return true;
}

#endif // !GREEDYINIT_H

