/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_HINGETREECOMMON_H
#define BLEAK_HINGETREECOMMON_H

#include <cmath>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <limits>
#include <utility>
#include <tuple>
#include <type_traits>

namespace bleak {

template<typename RealTypeT, typename KeyTypeT = uint32_t>
class HingeFernCommon {
public:
  typedef RealTypeT RealType;
  typedef KeyTypeT KeyType;
  typedef std::tuple<KeyType, RealType, KeyType> KeyMarginTupleType; // leaf key, margin, and threshold/ordinal index

  static_assert(std::is_integral<KeyType>::value && std::is_unsigned<KeyType>::value, "Unsigned integral type is required for leaf keys.");

  static constexpr int64_t GetMaxDepth() { return CHAR_BIT*sizeof(KeyType) - 1; }

  // -1 if an error (not power of 2)
  // This can be determined from threshold/ordinals size though!
  static int64_t ComputeDepth(int64_t i64LeafCount) {
    if (i64LeafCount <= 0 || (i64LeafCount & (i64LeafCount-1)) != 0)
      return -1; // Not a power of 2

    int64_t i64TreeDepth = 0;

    for ( ; i64LeafCount != 0; i64LeafCount >>= 1)
      ++i64TreeDepth;

    return i64TreeDepth-1;
  }

  // For sanity checks!
  static int64_t GetThresholdCount(int64_t i64TreeDepth) { return i64TreeDepth; } // Internal tree vertices
  static int64_t GetLeafCount(int64_t i64TreeDepth) { return ((int64_t)1) << i64TreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const RealType *p_ordinals, int64_t i64TreeDepth, int64_t i64Stride = 1) {
    KeyType leafKey = KeyType();
    RealType minMargin = p_data[i64Stride*(int64_t)p_ordinals[0]] - p_thresholds[0];
    KeyType minFernIndex = 0;

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = (int64_t)p_ordinals[i];
      const RealType margin = p_data[i64Stride*j] - p_thresholds[i];
      const KeyType bit = (margin > RealType(0));

      leafKey |= (bit << i);

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minFernIndex = KeyType(i);
      }
    }

    return std::make_tuple(leafKey, minMargin, minFernIndex);
  }

  // Check if thresholds are logically consistent
  static bool CheckThresholds(const RealType * /*p_thresholds*/, const RealType * /*p_ordinals*/, int64_t /*i64TreeDepth*/) { return true; } // Nothing to do since order of decisions does not matter

  // Checks and fixes logical consistency of thresholds... returns true if changes were made
  static bool FixThresholds(RealType * /*p_thresholds*/, const RealType * /*p_ordinals*/, int64_t /*i64TreeDepth*/) { return false; } // Nothing to do since order of decisions does not matter
};

template<typename RealTypeT, typename KeyTypeT = uint32_t>
class HingeTreeCommon {
public:
  typedef RealTypeT RealType;
  typedef KeyTypeT KeyType;
  typedef std::tuple<KeyType, RealType, KeyType> KeyMarginTupleType; // leaf key, margin, and threshold/ordinal index

  static_assert(std::is_integral<KeyType>::value && std::is_unsigned<KeyType>::value, "Unsigned integral type is required for leaf keys.");

  static constexpr int64_t GetMaxDepth() { return CHAR_BIT*sizeof(KeyType) - 1; }

  // -1 if an error (not power of 2)
  static int64_t ComputeDepth(int64_t i64LeafCount) {
    if (i64LeafCount <= 0 || (i64LeafCount & (i64LeafCount-1)) != 0)
      return -1; // Not a power of 2

    int64_t i64TreeDepth = 0;

    for ( ; i64LeafCount != 0; i64LeafCount >>= 1)
      ++i64TreeDepth;

    return i64TreeDepth-1;
  }

  static int64_t GetThresholdCount(int64_t i64TreeDepth) { return (((int64_t)1) << i64TreeDepth) - 1; } // Internal tree vertices
  static int64_t GetLeafCount(int64_t i64TreeDepth) { return ((int64_t)1) << i64TreeDepth; }

  // Returns leaf key, signed margin and threshold/ordinal index
  static KeyMarginTupleType ComputeKeyAndSignedMargin(const RealType *p_data, const RealType *p_thresholds, const RealType *p_ordinals, int64_t i64TreeDepth, int64_t i64Stride = 1) {
    KeyType leafKey = KeyType();
    KeyType treeIndex = KeyType();
    RealType minMargin = p_data[i64Stride * (int64_t)p_ordinals[0]] - p_thresholds[0];
    KeyType minTreeIndex = KeyType();

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = (int64_t)p_ordinals[treeIndex];
      const RealType margin = p_data[j*i64Stride] - p_thresholds[treeIndex];
      const KeyType bit = (margin > RealType(0));

      if (std::abs(margin) < std::abs(minMargin)) {
        minMargin = margin;
        minTreeIndex = treeIndex;
      }

      leafKey |= (bit << i);
      treeIndex = 2*treeIndex + 1 + bit;
    }

    return std::make_tuple(leafKey, minMargin, minTreeIndex);
  }

  static bool CheckThresholds(const RealType *p_thresholds, const RealType *p_ordinals, int64_t i64TreeDepth) {
    const int64_t i64ThresholdCount = GetThresholdCount(i64TreeDepth);

    for (int64_t i = i64ThresholdCount-1; i > 0; --i) {
      const int64_t i64Ordinal = (int64_t)p_ordinals[i];
      int64_t i64Node = i;

      while (i64Node > 0) {
        const int64_t i64Parent = (i64Node-1)/2;
        const bool bCameFromRight = (2*i64Parent+2 == i64Node);
        const int64_t i64ParentOrdinal = (int64_t)p_ordinals[i64Parent];

        if (i64Ordinal == i64ParentOrdinal) {
          if (bCameFromRight) {
            // Node i's threshold shall be larger than this parent's threshold
            if (p_thresholds[i] < p_thresholds[i64Parent])
              return false;
          }
          else {
            // Node i's threshold shall be smaller than this parent's threshold
            if (p_thresholds[i] > p_thresholds[i64Parent])
              return false;
          }
        }

        i64Node = i64Parent;
      }
    }

    return true;
  }

  // Checks and fixes logical consistency of thresholds
  static bool FixThresholds(RealType *p_thresholds, const RealType *p_ordinals, int64_t i64TreeDepth) {
    //constexpr RealType small = RealType(1e-1);
    const int64_t i64ThresholdCount = GetThresholdCount(i64TreeDepth);

    bool bChangesMade = false;

    for (int64_t i = 1; i < i64ThresholdCount; ++i) {
      const int64_t i64Ordinal = (int64_t)p_ordinals[i];
      int64_t i64Node = i;

      RealType minThreshold = -std::numeric_limits<RealType>::infinity();
      RealType maxThreshold = std::numeric_limits<RealType>::infinity();

      while (i64Node > 0) {
        const int64_t i64Parent = (i64Node-1)/2;
        const bool bCameFromRight = (2*i64Parent+2 == i64Node);
        const int64_t i64ParentOrdinal = (int64_t)p_ordinals[i64Parent];

        if (i64Ordinal == i64ParentOrdinal) {
          if (bCameFromRight)
            minThreshold = std::max(minThreshold, p_thresholds[i64Parent]);
          else
            maxThreshold = std::min(maxThreshold, p_thresholds[i64Parent]);
        }

        i64Node = i64Parent;
      }

      if (p_thresholds[i] < minThreshold || p_thresholds[i] > maxThreshold) {
        // At least one of them is finite if either of these conditions are true!

        if (!std::isfinite(minThreshold))
          p_thresholds[i] = maxThreshold - RealType(1);
        else if (!std::isfinite(maxThreshold))
          p_thresholds[i] = minThreshold + RealType(1);
        else
          p_thresholds[i] = RealType(0.5)*(minThreshold + maxThreshold);

        bChangesMade = true;
      }
    }

    return bChangesMade;
  }
};

} // end namespace bleak

#endif // !BLEAK_HINGETREECOMMON_H
