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

#ifndef HINGETRIECOMMON_H
#define HINGETRIECOMMON_H

#include <climits>
#include <cstdint>
#include <vector>
#include <type_traits>
#include <tuple>

namespace bleak {

// A Trie-like Fern would just be a linear combination of the same absolute value terms... makes no sense!

template<typename RealTypeT, typename KeyTypeT = uint32_t>
class HingeTrieCommon {
public:
  typedef RealTypeT RealType;
  typedef KeyTypeT KeyType;
  typedef std::tuple<RealType, KeyType> KeyMarginTupleType; // margin, and threshold/ordinal index ... No leaf key for now

  static_assert(std::is_integral<KeyType>::value && std::is_unsigned<KeyType>::value, "Unsigned integral type is required for vertex indices.");

  static constexpr int64_t GetMaxDepth() { return CHAR_BIT*sizeof(KeyType) - 1; }

  // -1 if an error (not power of 2)
  static int64_t ComputeDepth(int64_t i64ThresholdCount) {
    int i64LeafCount = i64ThresholdCount+1;
    if (i64LeafCount <= 0 || (i64LeafCount & (i64LeafCount-1)) != 0)
      return -1; // Not a power of 2

    int64_t i64TreeDepth = 0;

    for ( ; i64LeafCount != 0; i64LeafCount >>= 1)
      ++i64TreeDepth;

    return i64TreeDepth-1;
  }

  // For sanity checks!
  static int64_t GetThresholdCount(int64_t i64TreeDepth) { return (1 << i64TreeDepth) - 1; } // Internal tree vertices
  //static int GetLeafCount(int iTreeDepth) { return 1 << iTreeDepth; } // Not used yet! Maybe never...

  // Returns leaf key, signed margin and threshold/ordinal index
  static void ComputeKeyAndSignedMargin(KeyMarginTupleType a_tplPath[GetMaxDepth()], const RealType *p_data, const RealType *p_thresholds, const int64_t *p_ordinals, int64_t i64TreeDepth, int64_t i64Stride = 1) {
    KeyType treeIndex = KeyType();

    //vPath.resize(i64TreeDepth);

    for (int64_t i = 0; i < i64TreeDepth; ++i) {
      const int64_t j = p_ordinals[treeIndex];
      const RealType margin = p_data[j*i64Stride] - p_thresholds[treeIndex];
      const KeyType bit = (margin > RealType(0));

      a_tplPath[i] = std::make_tuple(margin, treeIndex);

      treeIndex = 2*treeIndex + 1 + bit;
    }
  }
};

} // end namespace bleak

#endif // !HINGETRIECOMMON_H
