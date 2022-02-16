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

#ifndef MEDIANINIT_H
#define MEDIANINIT_H

#include <iostream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <type_traits>
#include "torch/extension.h"
#include "HingeTreeCommon.h"
#include "HingeTrieCommon.h"

template<typename RealType>
struct Vertex {
  int64_t i64Ordinal = -1;
  RealType threshold = RealType();

  // Partition
  const RealType **p_begin = nullptr;
  const RealType **p_end = nullptr;
};

// Weights only used to deduce properties of trees
template<typename RealType, typename TreeTraitsType>
std::vector<std::vector<Vertex<RealType>>> FromPyTorch(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
bool ToPyTorch(const std::vector<std::vector<Vertex<RealType>>> &vTrees, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
bool InitMedianSplits(std::vector<std::vector<Vertex<RealType>>> &vTrees, torch::Tensor inData, torch::Tensor inWeights); 

template<typename RealType, typename TreeTraitsType>
std::vector<std::vector<Vertex<RealType>>> FromPyTorch(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef Vertex<RealType> VertexType;

  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<std::vector<VertexType>>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<std::vector<VertexType>>();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<std::vector<VertexType>>();

  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();

  std::vector<std::vector<VertexType>> vTrees;
  vTrees.reserve(i64NumTrees);

  std::vector<VertexType> vVertices;

  for (int64_t i = 0; i < i64NumTrees; ++i) {
    vVertices.clear(); 
    vVertices.reserve(i64NumDecisionsPerTree);

    for (int64_t j = 0; j < i64NumDecisionsPerTree; ++j) {
      VertexType stVertex;
      stVertex.i64Ordinal = (int64_t)p_inOrdinals[i*i64NumDecisionsPerTree + j];
      stVertex.threshold = p_inThresholds[i*i64NumDecisionsPerTree + j];
      vVertices.push_back(stVertex);
    }

    vTrees.emplace_back(std::move(vVertices));
  }

  return vTrees;
}

template<typename RealType, typename TreeTraitsType>
bool ToPyTorch(const std::vector<std::vector<Vertex<RealType>>> &vTrees, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  //typedef Vertex<RealType> VertexType;

  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return false;

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return false;

  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return false;

  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (i64NumTrees != (int64_t)vTrees.size() || (int64_t)vTrees[0].size() != i64NumDecisionsPerTree)
    return false;

  RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  RealType * const p_inOrdinals = inOrdinals.data_ptr<RealType>();

  for (int64_t i = 0; i < i64NumTrees; ++i) {
    for (int64_t j = 0; j < i64NumDecisionsPerTree; ++j) {
      p_inThresholds[i*i64NumDecisionsPerTree + j] = vTrees[i][j].threshold;
      p_inOrdinals[i*i64NumDecisionsPerTree + j] = RealType(vTrees[i][j].i64Ordinal); // Shouldn't need to assign this, but it won't hurt
    }
  }

  return true;
}

template<typename RealType, typename TreeTraitsType>
bool InitMedianSplits(std::vector<std::vector<Vertex<RealType>>> &vTrees, torch::Tensor inData, torch::Tensor inWeights) {
  typedef c10::IntArrayRef IntArrayRef;
  typedef Vertex<RealType> VertexType;

  if (inData.dim() < 2 || inWeights.dim() < 2)
    return false;

  if (inWeights.sizes()[0] != (int)vTrees.size())
    return false;

  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = TreeTraitsType::GetThresholdCount(i64TreeDepth);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || (int64_t)vTrees[0].size() != i64NumDecisionsPerTree)
    return false;

  const RealType * const p_inData = inData.data_ptr<RealType>();

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

  // Make row pointers
  std::vector<const RealType *> vRows;
  vRows.reserve(i64BatchSize*i64InnerDataNum);

  for (int64_t b = 0; b < i64BatchSize; ++b) {
    for (int64_t k = 0; k < i64InnerDataNum; ++k) {
      vRows.push_back(p_inData + ((b*i64NumChannels + 0)*i64InnerDataNum + k));
    }
  }

  // Setup root verices
  for (int64_t i = 0; i < i64NumTrees; ++i) {
    std::vector<VertexType> &vVertices = vTrees[i];
    vVertices[0].p_begin = vRows.data();
    vVertices[0].p_end = vVertices[0].p_begin + vRows.size();
  }

  if (std::is_same<TreeTraitsType, bleak::HingeFernCommon<RealType>>::value) {
    for (int64_t i = 0; i < i64NumTrees; ++i) {
      std::vector<VertexType> &vVertices = vTrees[i];

      for (int64_t j = 0; j < i64TreeDepth; ++j) {
        VertexType &stVertex = vVertices[j];
        stVertex.p_begin = vVertices[0].p_begin;
        stVertex.p_end = vVertices[0].p_end;

        const RealType **p_split = stVertex.p_begin + (stVertex.p_end - stVertex.p_begin)/2;
        std::nth_element(stVertex.p_begin, p_split, stVertex.p_end,
          [&stVertex, i64InnerDataNum](const RealType *a, const RealType *b) -> bool {
            return a[stVertex.i64Ordinal * i64InnerDataNum] > b[stVertex.i64Ordinal * i64InnerDataNum];
          });

        stVertex.threshold = (*p_split)[stVertex.i64Ordinal * i64InnerDataNum];
      }
    }
  }
  else if (std::is_same<TreeTraitsType, bleak::HingeTreeCommon<RealType>>::value || std::is_same<TreeTraitsType, bleak::HingeTrieCommon<RealType>>::value) {
    for (int64_t i = 0; i < i64NumTrees; ++i) {
      std::vector<VertexType> &vVertices = vTrees[i];

      for (int64_t j = 0; j < i64NumDecisionsPerTree; ++j) {
        VertexType &stVertex = vVertices[j];

        if (stVertex.p_begin == stVertex.p_end) {
          std::cerr << "Warning: Empty sample in vertex? Skipping..." << std::endl;
          continue; // Should not happen?
        }

        const RealType **p_split = stVertex.p_begin + (stVertex.p_end - stVertex.p_begin)/2;
        std::nth_element(stVertex.p_begin, p_split, stVertex.p_end,
          [&stVertex, i64InnerDataNum](const RealType *a, const RealType *b) -> bool {
            return a[stVertex.i64Ordinal * i64InnerDataNum] > b[stVertex.i64Ordinal * i64InnerDataNum];
          });

        stVertex.threshold = (*p_split)[stVertex.i64Ordinal * i64InnerDataNum];

        const int64_t i64LeftChild = 2*j + 1;

        if (i64LeftChild+1 < (int)vVertices.size()) {
          // Partition so that larger comes before smaller
          p_split = std::partition(stVertex.p_begin, stVertex.p_end,
            [&stVertex, i64InnerDataNum](const RealType *row) -> bool {
              return row[stVertex.i64Ordinal * i64InnerDataNum] > stVertex.threshold;
            });

          // Smaller partition to left child
          vVertices[i64LeftChild].p_begin = p_split;
          vVertices[i64LeftChild].p_end = stVertex.p_end;

          // Larger partition to right child
          vVertices[i64LeftChild+1].p_begin = stVertex.p_begin;
          vVertices[i64LeftChild+1].p_end = p_split;
        }
      }
    }
  }
  else { // Not supported?
    return false;
  }

  return true;
}

#endif // !MEDIANINIT_H

