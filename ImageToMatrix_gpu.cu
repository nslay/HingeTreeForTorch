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

#include "ImageToMatrix.h"

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

namespace bleak {

namespace {

// Can be passed by value between CPU and GPU
template<unsigned int Dimension>
struct Size {
  int64_t data[Dimension];
};

template<unsigned int Dimension>
class RasterCurveGPU {
public:
  typedef Size<Dimension> SizeType;
  typedef SizeType CoordType;

  __device__ RasterCurveGPU(const SizeType &stSize)
  : m_stSize(stSize) { }

  __device__ RasterCurveGPU(const int64_t a_i64Size[Dimension]) {
    for (unsigned int d = 0; d < Dimension; ++d)
      m_stSize.data[d] = a_i64Size[d];
  }

  __device__ int64_t Count() const {
    int64_t count = m_stSize.data[0];

    for (unsigned int d = 1; d < Dimension; ++d)
      count *= m_stSize.data[d];

    return count;
  }

  __device__ const SizeType & GetSize() const { return m_stSize; }

  __device__ int64_t Index(const CoordType &stCoord) const {
    int64_t index = stCoord.data[0];

    for (unsigned int d = 1; d < Dimension; ++d)
      index = m_stSize.data[d] * index + stCoord.data[d];

    return index;
  }

  __device__ int64_t IndexChecked(const CoordType &stCoord) const {
    if (stCoord.data[0] < 0 || stCoord.data[0] >= m_stSize.data[0])
      return -1;

    int64_t index = stCoord.data[0];

    for (unsigned int d = 1; d < Dimension; ++d) {
      if (stCoord.data[d] < 0 || stCoord.data[d] >= m_stSize.data[d])
        return -1;

      index = m_stSize.data[d] * index + stCoord.data[d];
    }

    return index;
  }

  __device__ CoordType Coordinate(int64_t index) const {
    CoordType stCoord;

    for (unsigned int d = Dimension-1; d > 0; --d) {
      const int64_t q = index / m_stSize.data[d];
      const int64_t r = index - q * m_stSize.data[d];
      stCoord.data[d] = r;
      index = q;
    }

    stCoord.data[0] = index;

    return stCoord;
  }

private:
  SizeType m_stSize;
};

template<typename RealType>
__global__ void ExtractMatrixHelper(RealType *d_matrix, const RealType *d_image, const int64_t *d_indexMatrix, int64_t i64Rows, int64_t i64Cols, RealType padValue) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (i < i64Rows && j < i64Cols) {
    const int64_t index = d_indexMatrix[i64Cols*i + j];
    d_matrix[i64Cols*i + j] = (index < 0) ? padValue : d_image[index];
  }
}

template<typename RealType>
__global__ void MapAndAddHelper(RealType *d_diff, int64_t i64Stride, const RealType *d_matrix, const int64_t *d_indexMatrix, int64_t i64Rows, int64_t i64Cols) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (i < i64Rows && j < i64Cols) {
    const int64_t index = d_indexMatrix[i64Cols*i + j];
    if (index >= 0) {
      atomicAdd(d_diff + index*i64Stride, d_matrix[i64Cols*i + j]);
      //d_diff[index*iStride] += d_matrix[iCols*i + j];
    }
  }
}

template<unsigned int Dimension>
__global__ void ExtractIndexMatrixHelper(int64_t *d_matrix, Size<Dimension> stKernelSize, Size<Dimension> stStride, Size<Dimension> stPadding, Size<Dimension> stDilate, Size<Dimension> stOutSize, Size<Dimension+1> stImageSize) {
  typedef RasterCurveGPU<Dimension> RasterType;
  typedef typename RasterType::CoordType CoordType; 

  const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  RasterType clOutRaster(stOutSize);
  RasterType clKernRaster(stKernelSize);
  RasterType clImageRaster(stImageSize.data+1);

  const int64_t i64Channels = stImageSize.data[0];
  const int64_t i64KernelCount = clKernRaster.Count();  
  const int64_t i64OutCount = clOutRaster.Count();

  if (c < i64Channels && j < i64KernelCount && i < i64OutCount) {
    const int64_t i64InCount = clImageRaster.Count();
    const int64_t indexOffset = c*i64InCount;
    const int64_t jOffset = c*i64KernelCount;
    const int64_t i64Cols = i64Channels * i64KernelCount;

    CoordType stWinCoord = clKernRaster.Coordinate(j);
    CoordType stCoord = clOutRaster.Coordinate(i);

    for (unsigned int d = 0; d < Dimension; ++d)
      stCoord.data[d] = stCoord.data[d] * stStride.data[d] + stWinCoord.data[d] * stDilate.data[d] - stPadding.data[d];

    const int64_t index = clImageRaster.IndexChecked(stCoord);
    d_matrix[i64Cols*i + (j + jOffset)] = (index < 0) ? index : index + indexOffset;
  }
}

} // end anonymous namespace

template<typename RealType, unsigned int Dimension>
void ImageToMatrixBase<RealType, Dimension>::ExtractMatrixGPU(RealType *d_matrix, const RealType *d_image, const int64_t *d_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const {
  int64_t i64Rows = 0;
  int64_t i64Cols = 0;
  ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((i64Rows + threadsPerBlock.x-1) / threadsPerBlock.x, (i64Cols + threadsPerBlock.y-1) / threadsPerBlock.y);
  ExtractMatrixHelper<<<numBlocks, threadsPerBlock>>>(d_matrix, d_image, d_indexMatrix, i64Rows, i64Cols, padValue);
}

template<typename RealType, unsigned int Dimension>
void ImageToMatrixBase<RealType, Dimension>::MapAndAddGPU(RealType *d_diff, int64_t i64Stride, const RealType *d_matrix, const int64_t *d_indexMatrix, const int64_t a_i64ImageSize[Dimension+1]) const {
  int64_t i64Rows = 0;
  int64_t i64Cols = 0;
  ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((i64Rows + threadsPerBlock.x-1) / threadsPerBlock.x, (i64Cols + threadsPerBlock.y-1) / threadsPerBlock.y);
  MapAndAddHelper<<<numBlocks, threadsPerBlock>>>(d_diff, i64Stride, d_matrix, d_indexMatrix, i64Rows, i64Cols);
}

template<typename RealType, unsigned int Dimension>
void ImageToMatrix<RealType, Dimension>::ExtractIndexMatrixGPU(int64_t *d_matrix, const int64_t a_i64ImageSize[Dimension+1]) const {
  const int64_t i64Channels = a_i64ImageSize[0];
  const int64_t i64KernelCount = ComputeKernelCount();
  const int64_t i64OutCount = ComputeOutputCount(a_i64ImageSize);

  const dim3 threadsPerBlock(4,16,8);
  const dim3 numBlocks((i64Channels + threadsPerBlock.x-1) / threadsPerBlock.x, (i64KernelCount + threadsPerBlock.y-1) / threadsPerBlock.y, (i64OutCount + threadsPerBlock.z-1) / threadsPerBlock.z);

  Size<Dimension> stKernelSize;
  Size<Dimension> stStride;
  Size<Dimension> stPadding;
  Size<Dimension> stDilate;
  Size<Dimension> stOutSize;
  Size<Dimension+1> stImageSize;

  std::copy_n(kernelSize.data(), Dimension, stKernelSize.data);
  std::copy_n(stride.data(), Dimension, stStride.data);
  std::copy_n(padding.data(), Dimension, stPadding.data);
  std::copy_n(dilate.data(), Dimension, stDilate.data);
  std::copy_n(ComputeOutputSize(a_i64ImageSize).data(), Dimension, stOutSize.data);
  std::copy_n(a_i64ImageSize, Dimension+1, stImageSize.data);

  ExtractIndexMatrixHelper<<<numBlocks, threadsPerBlock>>>(d_matrix, stKernelSize, stStride, stPadding, stDilate, stOutSize, stImageSize);
}

// Instantiate these functions by instantiating duplicate ImageToMatrixBase
template class ImageToMatrixBase<float, 1>;
template class ImageToMatrixBase<float, 2>;
template class ImageToMatrixBase<float, 3>;

template class ImageToMatrixBase<double, 1>;
template class ImageToMatrixBase<double, 2>;
template class ImageToMatrixBase<double, 3>;

// Instantiate these functions by instantiating duplicate ImageToMatrix
template class ImageToMatrix<float, 1>;
template class ImageToMatrix<float, 2>;
template class ImageToMatrix<float, 3>;

template class ImageToMatrix<double, 1>;
template class ImageToMatrix<double, 2>;
template class ImageToMatrix<double, 3>;

} // end namespace bleak

#if 0
// Test code...
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  bleak::ImageToMatrix<float, 2> clIm2Col;

  //
  // Reference 2D access pattern for a 6x5 image with padding
  //
  // -1 -1 -1 -1 -1 -1 -1
  // -1  0  1  2  3  4 -1
  // -1  5  6  7  8  9 -1
  // -1 10 11 12 13 14 -1
  // -1 15 16 17 18 19 -1
  // -1 20 21 22 23 24 -1
  // -1 25 26 27 28 29 -1
  // -1 -1 -1 -1 -1 -1 -1
  //

  clIm2Col.kernelSize[0] = 3;
  clIm2Col.kernelSize[1] = 4;

  clIm2Col.stride[0] = 1;
  clIm2Col.stride[1] = 1;

  clIm2Col.dilate[0] = 1;
  clIm2Col.dilate[1] = 1;

  clIm2Col.padding[0] = 1;
  clIm2Col.padding[1] = 1;

  const int64_t a_i64ImageSize[3] = { 3, 6, 5 }; // 3 channels, 6 rows, 5 columns

  if (!clIm2Col.Good(a_i64ImageSize)) {
    std::cerr << "Error: Bad image size." << std::endl;
    return -1;
  }

  int64_t rows = 0, cols = 0;
  clIm2Col.ComputeMatrixDimensions(rows, cols, a_i64ImageSize);

  std::vector<int64_t> vIndexMatrix(rows*cols, 0);

  clIm2Col.ExtractIndexMatrix(vIndexMatrix.data(), a_i64ImageSize);

  std::cout << "Index matrix: " << std::endl;

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j)
      std::cout << vIndexMatrix[cols*i + j] << ' ';

    std::cout << std::endl;
  }

  int64_t *d_indexMatrix = nullptr;
  if (cudaMalloc((void **)(&d_indexMatrix), vIndexMatrix.size()*sizeof(int64_t)) != cudaSuccess) {
    std::cerr << "Error: cudaMalloc failed." << std::endl;
    return -1;
  }

  clIm2Col.ExtractIndexMatrixGPU(d_indexMatrix, a_i64ImageSize);

  std::vector<int64_t> vIndexMatrix2(vIndexMatrix.size());

  if (cudaMemcpy(vIndexMatrix2.data(), d_indexMatrix, vIndexMatrix2.size()*sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "Error: cudaMemcpy failed." << std::endl;
    return -1;
  }

  std::cout << "GPU index matrix: " << std::endl;

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j)
      std::cout << vIndexMatrix2[cols*i + j] << ' ';

    std::cout << std::endl;
  }

  if (std::equal(vIndexMatrix.begin(), vIndexMatrix.end(), vIndexMatrix2.begin())) {
    std::cout << "Info: Good." << std::endl;
  }
  else {
    std::cerr << "Error: Not equal." << std::endl;
  }

  return 0;
}
#endif 

