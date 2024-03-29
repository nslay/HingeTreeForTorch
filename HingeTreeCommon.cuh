/*-
 * Copyright (c) 2020 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
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

#ifndef HINGETREECOMMON_CUH
#define HINGETREECOMMON_CUH

#include "HingeTreeCommon.h"

namespace bleak {

template<typename TreeTraitsType>
class HingeTreeCommonGPU { };

// A lame way to extend these classes to get __device__ functions!
template<typename RealTypeT, typename KeyTypeT>
class HingeTreeCommonGPU<HingeFernCommon<RealTypeT, KeyTypeT>> {
public:
  typedef HingeFernCommon<RealTypeT, KeyTypeT> TreeTraitsType;
  typedef typename TreeTraitsType::RealType RealType;
  typedef typename TreeTraitsType::KeyType KeyType;

  struct KeyMarginTuple {
    KeyType leafKey;
    RealType signedMargin;
    KeyType thresholdIndex;
    __device__ KeyMarginTuple(const KeyType &leafKey_, const RealType &signedMargin_, const KeyType &thresholdIndex_)
    : leafKey(leafKey_), signedMargin(signedMargin_), thresholdIndex(thresholdIndex_) { }
  };

  typedef KeyMarginTuple KeyMarginTupleType;

  __device__ static int64_t GetThresholdCount(int64_t i64TreeDepth) { return i64TreeDepth; } // Internal tree vertices
  __device__ static int64_t GetLeafCount(int64_t i64TreeDepth) { return ((int64_t)1) << i64TreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  __device__ static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const int64_t *p_ordinals, int64_t i64TreeDepth, int64_t i64Stride = 1) {
    KeyType leafKey = KeyType();
    RealType minMargin = p_data[i64Stride*p_ordinals[0]] - p_thresholds[0];
    KeyType minFernIndex = 0;

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = p_ordinals[i];
      const RealType margin = p_data[i64Stride*j] - p_thresholds[i];
      const KeyType bit = (margin > RealType(0));

      leafKey |= (bit << i);

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minFernIndex = KeyType(i);
      }
    }

    return KeyMarginTupleType(leafKey, minMargin, minFernIndex);
  }

  __device__ static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_img, const RealType *p_vec, const RealType *p_thresholds, const int64_t *p_ordinals, int64_t i64TreeDepth, int64_t i64ImgChannels, int64_t i64Stride = 1) {
    auto GetFeature = [&](int64_t j) -> RealType {
      if (j < i64ImgChannels)
          return p_img[i64Stride*j];

      return p_vec[j-i64ImgChannels];
    };

    KeyType leafKey = KeyType();
    RealType minMargin = GetFeature(p_ordinals[0]) - p_thresholds[0];
    KeyType minFernIndex = 0;

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = p_ordinals[i];
      const RealType margin = GetFeature(j) - p_thresholds[i];
      const KeyType bit = (margin > RealType(0));

      leafKey |= (bit << i);

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minFernIndex = KeyType(i);
      }
    }

    return KeyMarginTupleType(leafKey, minMargin, minFernIndex);
  }

};

// A lame way to extend these classes to get __device__ functions!
template<typename RealTypeT, typename KeyTypeT>
class HingeTreeCommonGPU<HingeTreeCommon<RealTypeT, KeyTypeT>> {
public:
  typedef HingeTreeCommon<RealTypeT, KeyTypeT> TreeTraitsType;
  typedef typename TreeTraitsType::RealType RealType;
  typedef typename TreeTraitsType::KeyType KeyType;

  struct KeyMarginTuple {
    KeyType leafKey;
    RealType signedMargin;
    KeyType thresholdIndex;
    __device__ KeyMarginTuple(const KeyType &leafKey_, const RealType &signedMargin_, const KeyType &thresholdIndex_)
    : leafKey(leafKey_), signedMargin(signedMargin_), thresholdIndex(thresholdIndex_) { }
  };

  typedef KeyMarginTuple KeyMarginTupleType;

  __device__ static int64_t GetThresholdCount(int64_t i64TreeDepth) { return (((int64_t)1) << i64TreeDepth) - 1; } // Internal tree vertices
  __device__ static int64_t GetLeafCount(int64_t i64TreeDepth) { return ((int64_t)1) << i64TreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  __device__ static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const int64_t *p_ordinals, int64_t i64TreeDepth, int64_t i64Stride = 1) {
    KeyType leafKey = KeyType();
    KeyType treeIndex = KeyType();
    RealType minMargin = p_data[i64Stride * p_ordinals[0]] - p_thresholds[0];
    KeyType minTreeIndex = KeyType();

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = p_ordinals[treeIndex];
      const RealType margin = p_data[j*i64Stride] - p_thresholds[treeIndex];
      const KeyType bit = (margin > RealType(0));

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minTreeIndex = treeIndex;
      }

      leafKey |= (bit << i);
      treeIndex = 2*treeIndex + 1 + bit;
    }

    return KeyMarginTuple(leafKey, minMargin, minTreeIndex);
  }

  __device__ static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_img, const RealType *p_vec, const RealType *p_thresholds, const int64_t *p_ordinals, int64_t i64TreeDepth, int64_t i64ImgChannels, int64_t i64Stride = 1) {
    auto GetFeature = [&](int64_t j) -> RealType {
      if (j < i64ImgChannels)
          return p_img[i64Stride*j];

      return p_vec[j-i64ImgChannels];
    };

    KeyType leafKey = KeyType();
    KeyType treeIndex = KeyType();
    RealType minMargin = GetFeature(p_ordinals[0]) - p_thresholds[0];
    KeyType minTreeIndex = KeyType();

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = p_ordinals[treeIndex];
      const RealType margin = GetFeature(j) - p_thresholds[treeIndex];
      const KeyType bit = (margin > RealType(0));

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minTreeIndex = treeIndex;
      }

      leafKey |= (bit << i);
      treeIndex = 2*treeIndex + 1 + bit;
    }

    return KeyMarginTuple(leafKey, minMargin, minTreeIndex);
  }

};

} // end namespace bleak

#endif // HINGETREECOMMON_CUH
