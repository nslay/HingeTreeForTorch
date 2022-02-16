/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_IMAGETOMATRIX_H
#define BLEAK_IMAGETOMATRIX_H

#include <cstdint>
#include <array>
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#define bleakNewImageToMatrix(className, superClass) \
  typedef className SelfType; \
  typedef superClass SuperType; \
  typedef typename SuperType::RasterType RasterType; \
  typedef typename SuperType::SizeType SizeType; \
  typedef typename SuperType::CoordType CoordType; \
  using SuperType::kernelSize; \
  using SuperType::padding; \
  using SuperType::stride; \
  using SuperType::dilate; \
  using SuperType::padValue; \
  using SuperType::GetDimension; \
  using SuperType::ComputeMatrixDimensions; \
  using SuperType::ComputeOutputSize; \
  using SuperType::ComputeOutputCount; \
  using SuperType::ComputeKernelCount; \
  using SuperType::ComputeWindowSize; \
  using SuperType::ExtractMatrix; \
  using SuperType::Good

namespace bleak {

template<unsigned int Dimension>
class RasterCurve {
public:
  typedef std::array<int64_t, Dimension> SizeType;
  typedef SizeType CoordType;

  explicit RasterCurve(const SizeType &clSize)
  : m_clSize(clSize) { }

  explicit RasterCurve(const int64_t a_i64Size[Dimension]) {
    std::copy_n(a_i64Size, Dimension, m_clSize.begin());
  }

  int64_t Count() const { return std::accumulate(m_clSize.begin(), m_clSize.end(), (int64_t)1, std::multiplies<int64_t>()); }

  const SizeType & GetSize() const { return m_clSize; }

  int64_t Index(const CoordType &clCoord) const {
    int64_t index = clCoord[0];

    for (unsigned int d = 1; d < Dimension; ++d)
      index = m_clSize[d] * index + clCoord[d];

    return index;
  }

  int64_t IndexChecked(const CoordType &clCoord) const {
    if (clCoord[0] < 0 || clCoord[0] >= m_clSize[0])
      return -1;

    int64_t index = clCoord[0];

    for (unsigned int d = 1; d < Dimension; ++d) {
      if (clCoord[d] < 0 || clCoord[d] >= m_clSize[d])
        return -1;

      index = m_clSize[d] * index + clCoord[d];
    }

    return index;
  }

  CoordType Coordinate(int64_t index) const {
    CoordType clCoord;

    for (unsigned int d = Dimension-1; d > 0; --d) {
      const int64_t q = index / m_clSize[d];
      const int64_t r = index - q * m_clSize[d];
      clCoord[d] = r;
      index = q;
    }

    clCoord[0] = index;

    return clCoord;
  }

private:
  SizeType m_clSize;
  //std::array<float, Dimension> m_clInvSize;
};

// Don't refer to this base class by reference!
template<typename RealType, unsigned int Dimension>
class ImageToMatrixBase {
public:
  static_assert(Dimension > 0, "Dimension must be larger than 0");

  typedef RasterCurve<Dimension> RasterType;
  typedef typename RasterType::SizeType SizeType;
  typedef typename RasterType::CoordType CoordType;

  // Z x Y x X
  SizeType kernelSize;
  SizeType stride;
  SizeType padding;
  SizeType dilate;

  RealType padValue = RealType();

  static constexpr unsigned int GetDimension() {
    return Dimension;
  }

  ImageToMatrixBase() {
    kernelSize.fill(0);
    padding.fill(0);
    stride.fill(1);
    dilate.fill(1);

    padValue = RealType();
  }

  // Convenience functions...
  void SetKernelSize(const int64_t a_i64KernelSize[Dimension]) { std::copy_n(a_i64KernelSize, Dimension, kernelSize.begin()); }
  void SetPadding(const int64_t a_i64Padding[Dimension]) { std::copy_n(a_i64Padding, Dimension, padding.begin()); }
  void SetStride(const int64_t a_i64Stride[Dimension]) { std::copy_n(a_i64Stride, Dimension, stride.begin()); }
  void SetDilate(const int64_t a_i64Dilate[Dimension]) { std::copy_n(a_i64Dilate, Dimension, dilate.begin()); }

  bool Good() const {
    if (*std::min_element(kernelSize.begin(), kernelSize.end()) <= 0 ||
      *std::min_element(padding.begin(), padding.end()) < 0 ||
      *std::min_element(dilate.begin(), dilate.end()) <= 0 ||
      *std::min_element(stride.begin(), stride.end()) <= 0) {
      return false;
    }

    return true;
  }

  // a_i64ImageSize: C x Z x Y x X x ...
  bool Good(const int64_t a_i64ImageSize[Dimension+1]) const {
    if (a_i64ImageSize[0] < 1 || !Good())
      return false;

    const SizeType winSize = ComputeWindowSize();
    
    for (unsigned int d = 0; d < Dimension; ++d) {
      if (a_i64ImageSize[1+d] < 1 || a_i64ImageSize[1+d] + 2*padding[d] < winSize[d])
        return false;
    }

    return true;
  }

  // Window = dilated kernel
  // Kernel = not dilated
  
  // Dilated kernel size
  SizeType ComputeWindowSize() const {
    SizeType winSize;

    for (unsigned int d = 0; d < Dimension; ++d)
      winSize[d] = kernelSize[d] + (kernelSize[d] - 1)*(dilate[d] - 1);

    return winSize;
  }

  int64_t ComputeKernelCount() const { return std::accumulate(kernelSize.begin(), kernelSize.end(), (int64_t)1, std::multiplies<int64_t>()); }

  // Output size neglecting channels: C x Z x Y x X ...
  SizeType ComputeOutputSize(const int64_t a_i64ImageSize[Dimension+1]) const {
    const SizeType winSize = ComputeWindowSize();
    SizeType outSize;

    for (unsigned int d = 0; d < Dimension; ++d)
      outSize[d] = (a_i64ImageSize[1+d] + 2*padding[d] - winSize[d]) / stride[d] + 1;

    return outSize;
  }

  // Number of windows in image neglecting channels: C x Z x Y x X ...
  int64_t ComputeOutputCount(const int64_t a_i64ImageSize[Dimension+1]) const {
    const SizeType outSize = ComputeOutputSize(a_i64ImageSize);
    return std::accumulate(outSize.begin(), outSize.end(), (int64_t)1, std::multiplies<int64_t>());
  }

  // Row major (C/C++)
  void ComputeMatrixDimensions(int64_t &i64Rows, int64_t &i64Cols, const int64_t a_i64ImageSize[Dimension+1]) const {
    i64Rows = ComputeOutputCount(a_i64ImageSize);
    i64Cols = ComputeKernelCount() * a_i64ImageSize[0];
  }

  void ExtractMatrix(RealType *p_matrix, const RealType *p_image, const int64_t *p_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const {
    int64_t i64Rows = 0;
    int64_t i64Cols = 0;
    ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

    for (int64_t j = 0; j < i64Cols; ++j) {
      for (int64_t i = 0; i < i64Rows; ++i) {
        // This seems less efficient, but it keeps locality better in p_image (or should in most cases)
        const int64_t index = p_indexMatrix[i64Cols*i + j];
        p_matrix[i64Cols*i + j] = (index < 0) ? padValue : p_image[index];
      }
    }
  }

  void MapAndAdd(RealType *p_diff, int64_t i64Stride, const RealType *p_matrix, const int64_t *p_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const {
    int64_t i64Rows = 0;
    int64_t i64Cols = 0;
    ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

    for (int64_t j = 0; j < i64Cols; ++j) {
      for (int64_t i = 0; i < i64Rows; ++i) {
        const int64_t index = p_indexMatrix[i64Cols*i + j];
        if (index >= 0)
          p_diff[index*i64Stride] += p_matrix[i64Cols*i + j];
      }
    }
  }

//#ifdef BLEAK_USE_CUDA
#ifdef WITH_CUDA
  void ExtractMatrixGPU(RealType *d_matrix, const RealType *d_image, const int64_t *d_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const;
  void MapAndAddGPU(RealType *d_diff, int64_t i64Stride, const RealType *d_matrix, const int64_t *d_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const;
#endif // BLEAK_USE_CUDA

};

template<typename RealType, unsigned int Dimension>
class ImageToMatrix : public ImageToMatrixBase<RealType, Dimension> { 
public:
  typedef ImageToMatrixBase<RealType, Dimension> WorkAroundVarArgsType;

  bleakNewImageToMatrix(ImageToMatrix, WorkAroundVarArgsType);

  void ExtractMatrix(RealType *p_matrix, const RealType *p_image, const int64_t a_i64ImageSize[Dimension+1]) const {
    const RasterType outRaster(ComputeOutputSize(a_i64ImageSize));
    const RasterType kernRaster(kernelSize);
    const RasterType imageRaster(a_i64ImageSize+1);

    const int64_t i64Channels = a_i64ImageSize[0];

    int64_t i64Rows = 0;
    int64_t i64Cols = 0;
    ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

    const int64_t i64KernelCount = kernRaster.Count();
    const int64_t i64OutCount = outRaster.Count();
    const int64_t i64InCount = imageRaster.Count();

    for (int64_t c = 0; c < i64Channels; ++c) {
      const int64_t indexOffset = c*i64InCount;
      const int64_t jOffset = c*i64KernelCount;

      // Do loops in weird order to preserve locality in p_image
      for (int64_t j = 0; j < i64KernelCount; ++j) {
        CoordType winCoord = kernRaster.Coordinate(j);

        for (unsigned int d = 0; d < Dimension; ++d)
          winCoord[d] *= dilate[d];

        for (int64_t i = 0; i < i64OutCount; ++i) {
          CoordType coord = outRaster.Coordinate(i);

          for (unsigned int d = 0; d < Dimension; ++d)
            coord[d] = stride[d]*coord[d] + winCoord[d] - padding[d];

          const int64_t index = imageRaster.IndexChecked(coord);
          p_matrix[i64Cols*i + (j + jOffset)] = (index < 0) ? padValue : p_image[index + indexOffset];
        }
      }
    }
  }

  void ExtractIndexMatrix(int64_t *p_matrix, const int64_t a_i64ImageSize[Dimension+1]) const {
    const RasterType outRaster(ComputeOutputSize(a_i64ImageSize));
    const RasterType kernRaster(kernelSize);
    const RasterType imageRaster(a_i64ImageSize+1);

    const int64_t i64Channels = a_i64ImageSize[0];

    int64_t i64Rows = 0;
    int64_t i64Cols = 0;
    ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

    const int64_t i64KernelCount = kernRaster.Count();
    const int64_t i64OutCount = outRaster.Count();
    const int64_t i64InCount = imageRaster.Count();

    for (int64_t i = 0; i < i64OutCount; ++i) {
      CoordType anchorCoord = outRaster.Coordinate(i); 

      for (unsigned int d = 0; d < Dimension; ++d)
        anchorCoord[d] = anchorCoord[d]*stride[d] - padding[d];

      for (int64_t j = 0; j < i64KernelCount; ++j) {
        CoordType coord = kernRaster.Coordinate(j); // kernel coordinate

        for (unsigned int d = 0; d < Dimension; ++d)
          coord[d] = anchorCoord[d] + coord[d]*dilate[d]; // Now coordinate in source image

        p_matrix[i64Cols*i + j] = imageRaster.IndexChecked(coord);
      }
    }

    // Stripe the other channels
    for (int64_t c = 1; c < i64Channels; ++c) {
      const int64_t indexOffset = c*i64InCount;
      const int64_t jOffset = c*i64KernelCount;

      for (int64_t i = 0; i < i64OutCount; ++i) {
        for (int64_t j = 0; j < i64KernelCount; ++j) {
          const int64_t index = p_matrix[i64Cols*i + j];

          p_matrix[i64Cols*i + (j + jOffset)] = (index < 0) ? index : index + indexOffset;
        }
      }
    }

    // Original code
#if 0
    for (int64_t c = 0; c < i64Channels; ++c) {
      const int64_t indexOffset = c*i64InCount;
      const int64_t jOffset = c*i64KernelCount;

      for (int64_t j = 0; j < i64KernelCount; ++j) {
        CoordType winCoord = kernRaster.Coordinate(j);

        for (unsigned int d = 0; d < Dimension; ++d)
          winCoord[d] *= dilate[d];

        for (int64_t i = 0; i < i64OutCount; ++i) {
          CoordType coord = outRaster.Coordinate(i);

          for (unsigned int d = 0; d < Dimension; ++d)
            coord[d] = stride[d]*coord[d] + winCoord[d] - padding[d];

          const int64_t index = imageRaster.IndexChecked(coord);
          p_matrix[i64Cols*i + (j + jOffset)] = (index < 0) ? index : (index + indexOffset);
        }
      }
    }
#endif
  }

#ifdef WITH_CUDA
  void ExtractIndexMatrixGPU(int64_t *d_matrix, const int64_t a_i64ImageSize[Dimension+1]) const;
#endif // WITH_CUDA
};

} // end namespace bleak

#endif // !BLEAK_IMAGETOMATRIX_H

