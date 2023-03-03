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

typedef c10::IntArrayRef IntArrayRef;

// PyTorch can't fold/unfold 3D volumes. I'll just write my own...

template<typename RealType>
torch::Tensor contract2d_cpu(torch::Tensor inData, const int64_t a_i64Window[2], const int64_t a_i64Padding[2]) {
  if (inData.dim() != 4 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0)
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64Height = inData.sizes()[2];
  const int64_t i64Width = inData.sizes()[3];

  if (a_i64Window[0] < 1 || a_i64Window[1] < 1 || a_i64Window[0] > i64Height + 2*a_i64Padding[0] || a_i64Window[1] > i64Width + 2*a_i64Padding[1])
    return torch::Tensor();

  std::vector<IntArrayRef::value_type> vSizes(6);

  vSizes[0] = inData.sizes()[0];
  vSizes[1] = inData.sizes()[1];
  vSizes[2] = (i64Height + 2*a_i64Padding[0] - a_i64Window[0])/a_i64Window[0] + 1;
  vSizes[3] = (i64Width + 2*a_i64Padding[1] - a_i64Window[1])/a_i64Window[1] + 1;
  vSizes[4] = a_i64Window[0];
  vSizes[5] = a_i64Window[1];

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  const RealType * const p_inData = inData.data_ptr<RealType>();
  RealType * const p_outData = outData.data_ptr<RealType>();

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t c = 0; c < i64NumChannels; ++c) {
      for (int64_t j = 0; j < outData.sizes()[2]; ++j) {
        const int64_t by = j*a_i64Window[0] - a_i64Padding[0];
        const int64_t ey = by + a_i64Window[0];
        const int64_t by2 = std::max(int64_t(0), by);
        const int64_t ey2 = std::min(i64Height, ey);

        for (int64_t k = 0; k < outData.sizes()[3]; ++k) {
          const int64_t bx = k*a_i64Window[1] - a_i64Padding[1];
          const int64_t ex = bx + a_i64Window[1];
          const int64_t bx2 = std::max(int64_t(0), bx);
          const int64_t ex2 = std::min(i64Width, ex);

          for (int64_t y = by2; y < ey2; ++y) {
            for (int64_t x = bx2; x < ex2; ++x) {
              p_outData[((((i*i64NumChannels + c)*outData.sizes()[2] + j)*outData.sizes()[3] + k)*a_i64Window[0] + (y-by))*a_i64Window[1] + (x-bx)] = p_inData[((i*i64NumChannels + c)*i64Height + y)*i64Width + x];
            }
          }
        }
      }
    }
  }

  return outData;
}

template<typename RealType>
torch::Tensor contract3d_cpu(torch::Tensor inData, const int64_t a_i64Window[3], const int64_t a_i64Padding[3]) {
  if (inData.dim() != 5 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0 || a_i64Padding[2] < 0)
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64Depth = inData.sizes()[2];
  const int64_t i64Height = inData.sizes()[3];
  const int64_t i64Width = inData.sizes()[4];

  if (a_i64Window[0] < 1 || a_i64Window[1] < 1 || a_i64Window[2] < 1 || a_i64Window[0] > i64Depth + 2*a_i64Padding[0] || a_i64Window[1] > i64Height + 2*a_i64Padding[1] || a_i64Window[2] > i64Width + 2*a_i64Padding[2])
    return torch::Tensor();

  std::vector<IntArrayRef::value_type> vSizes(8);

  vSizes[0] = inData.sizes()[0];
  vSizes[1] = inData.sizes()[1];
  vSizes[2] = (i64Depth + 2*a_i64Padding[0] - a_i64Window[0])/a_i64Window[0] + 1;
  vSizes[3] = (i64Height + 2*a_i64Padding[1] - a_i64Window[1])/a_i64Window[1] + 1;
  vSizes[4] = (i64Width + 2*a_i64Padding[2] - a_i64Window[2])/a_i64Window[2] + 1;
  vSizes[5] = a_i64Window[0];
  vSizes[6] = a_i64Window[1];
  vSizes[7] = a_i64Window[2];

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  const RealType * const p_inData = inData.data_ptr<RealType>();
  RealType * const p_outData = outData.data_ptr<RealType>();

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t c = 0; c < i64NumChannels; ++c) {
      for (int64_t j = 0; j < outData.sizes()[2]; ++j) {
        const int64_t bz = j*a_i64Window[0] - a_i64Padding[0];
        const int64_t ez = bz + a_i64Window[0];
        const int64_t bz2 = std::max(int64_t(0), bz);
        const int64_t ez2 = std::min(i64Depth, ez);

        for (int64_t k = 0; k < outData.sizes()[3]; ++k) {
          const int64_t by = k*a_i64Window[1] - a_i64Padding[1];
          const int64_t ey = by + a_i64Window[1];
          const int64_t by2 = std::max(int64_t(0), by);
          const int64_t ey2 = std::min(i64Height, ey);

          for (int64_t l = 0; l < outData.sizes()[4]; ++l) {
            const int64_t bx = l*a_i64Window[2] - a_i64Padding[2];
            const int64_t ex = bx + a_i64Window[2];
            const int64_t bx2 = std::max(int64_t(0), bx);
            const int64_t ex2 = std::min(i64Width, ex);

            for (int64_t z = bz2; z < ez2; ++z) {
              for (int64_t y = by2; y < ey2; ++y) {
                for (int64_t x = bx2; x < ex2; ++x) {
                  p_outData[((((((i*i64NumChannels + c)*outData.sizes()[2] + j)*outData.sizes()[3] + k)*outData.sizes()[4] + l)*a_i64Window[0] + (z-bz))*a_i64Window[1] + (y-by))*a_i64Window[2] + (x-bx)] = p_inData[(((i*i64NumChannels + c)*i64Depth + z)*i64Height + y)*i64Width + x];
                }
              }
            }
          }
        }
      }
    }
  }

  return outData;
}

template<typename RealType>
torch::Tensor expand2d_cpu(torch::Tensor inData, const int64_t a_i64Padding[2]) {
  if (inData.dim() != 6 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0)
    return torch::Tensor();


  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t a_i64Window[2] = { inData.sizes()[4], inData.sizes()[5] };
  const int64_t i64Height = inData.sizes()[2]*a_i64Window[0] - ((2*a_i64Padding[0])/a_i64Window[0])*a_i64Window[0];
  const int64_t i64Width = inData.sizes()[3]*a_i64Window[1] - ((2*a_i64Padding[1])/a_i64Window[1])*a_i64Window[1];

  if (i64Height < 1 || i64Width < 1 )
    return torch::Tensor();
 
  std::vector<IntArrayRef::value_type> vSizes(4);

  vSizes[0] = inData.sizes()[0];   
  vSizes[1] = inData.sizes()[1];
  vSizes[2] = i64Height;
  vSizes[3] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  const RealType * const p_inData = inData.data_ptr<RealType>();
  RealType * const p_outData = outData.data_ptr<RealType>();

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t c = 0; c < i64NumChannels; ++c) {
      for (int64_t j = 0; j < inData.sizes()[2]; ++j) {
        const int64_t by = j*a_i64Window[0] - a_i64Padding[0];
        const int64_t ey = by + a_i64Window[0];
        const int64_t by2 = std::max(int64_t(0), by);
        const int64_t ey2 = std::min(i64Height, ey);

        for (int64_t k = 0; k < inData.sizes()[3]; ++k) {
          const int64_t bx = k*a_i64Window[1] - a_i64Padding[1];
          const int64_t ex = bx + a_i64Window[1];
          const int64_t bx2 = std::max(int64_t(0), bx);
          const int64_t ex2 = std::min(i64Width, ex);

          for (int64_t y = by2; y < ey2; ++y) {
            for (int64_t x = bx2; x < ex2; ++x) {
              p_outData[((i*i64NumChannels + c)*i64Height + y)*i64Width + x] = p_inData[((((i*i64NumChannels + c)*inData.sizes()[2] + j)*inData.sizes()[3] + k)*a_i64Window[0] + (y-by))*a_i64Window[1] + (x-bx)];
            }
          }
        }
      }
    }
  }

  return outData;
}

template<typename RealType>
torch::Tensor expand3d_cpu(torch::Tensor inData, const int64_t a_i64Padding[2]) {
  if (inData.dim() != 8 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0 || a_i64Padding[2] < 0)
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t a_i64Window[3] = { inData.sizes()[5], inData.sizes()[6], inData.sizes()[7] };
  const int64_t i64Depth = inData.sizes()[2]*a_i64Window[0] - ((2*a_i64Padding[0])/a_i64Window[0])*a_i64Window[0];
  const int64_t i64Height = inData.sizes()[3]*a_i64Window[1] - ((2*a_i64Padding[1])/a_i64Window[1])*a_i64Window[1];
  const int64_t i64Width = inData.sizes()[4]*a_i64Window[2] - ((2*a_i64Padding[2])/a_i64Window[2])*a_i64Window[2];

  if (i64Depth < 1 || i64Height < 1 || i64Width < 1)
    return torch::Tensor();
 
  std::vector<IntArrayRef::value_type> vSizes(5);

  vSizes[0] = inData.sizes()[0];   
  vSizes[1] = inData.sizes()[1];
  vSizes[2] = i64Depth;
  vSizes[3] = i64Height;
  vSizes[4] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  const RealType * const p_inData = inData.data_ptr<RealType>();
  RealType * const p_outData = outData.data_ptr<RealType>();

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t c = 0; c < i64NumChannels; ++c) {
      for (int64_t j = 0; j < inData.sizes()[2]; ++j) {
        const int64_t bz = j*a_i64Window[0] - a_i64Padding[0];
        const int64_t ez = bz + a_i64Window[0];
        const int64_t bz2 = std::max(int64_t(0), bz);
        const int64_t ez2 = std::min(i64Depth, ez);

        for (int64_t k = 0; k < inData.sizes()[3]; ++k) {
          const int64_t by = k*a_i64Window[1] - a_i64Padding[1];
          const int64_t ey = by + a_i64Window[1];
          const int64_t by2 = std::max(int64_t(0), by);
          const int64_t ey2 = std::min(i64Height, ey);

          for (int64_t l = 0; l < inData.sizes()[4]; ++l) {
            const int64_t bx = l*a_i64Window[2] - a_i64Padding[2];
            const int64_t ex = bx + a_i64Window[2];
            const int64_t bx2 = std::max(int64_t(0), bx);
            const int64_t ex2 = std::min(i64Width, ex);

            for (int64_t z = bz2; z < ez2; ++z) {
              for (int64_t y = by2; y < ey2; ++y) {
                for (int64_t x = bx2; x < ex2; ++x) {
                  p_outData[(((i*i64NumChannels + c)*i64Depth + z)*i64Height + y)*i64Width + x] = p_inData[((((((i*i64NumChannels + c)*inData.sizes()[2] + j)*inData.sizes()[3] + k)*inData.sizes()[4] + l)*a_i64Window[0] + (z-bz))*a_i64Window[1] + (y-by))*a_i64Window[2] + (x-bx)];
                }
              }
            }
          }
        }
      }
    }
  }

  return outData;
}

torch::Tensor contract(torch::Tensor inData, IntArrayRef window, IntArrayRef padding) {
  if (window.empty() || window.size() != padding.size())
    return torch::Tensor();

  if (inData.device() != torch::kCPU)
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kUInt8:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<uint8_t>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<uint8_t>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kInt8:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<int8_t>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<int8_t>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kInt16:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<int16_t>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<int16_t>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kInt32:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<int32_t>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<int32_t>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kInt64:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<int64_t>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<int64_t>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kFloat32:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<float>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<float>(inData, window.data(), padding.data());
      }
    }
    break;
  case torch::kFloat64:
    {
      switch (window.size()) {
      case 2:
        return contract2d_cpu<double>(inData, window.data(), padding.data());
      case 3:
        return contract3d_cpu<double>(inData, window.data(), padding.data());
      }
    }
    break;
  default:
    return torch::Tensor();
  }

  return torch::Tensor(); 
}

torch::Tensor expand(torch::Tensor inData, IntArrayRef padding) {
  if (padding.empty())
    return torch::Tensor();

  if (inData.device() != torch::kCPU)
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kUInt8:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<uint8_t>(inData, padding.data());
      case 3:
        return expand3d_cpu<uint8_t>(inData, padding.data());
      }
    }
    break;
  case torch::kInt8:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<int8_t>(inData, padding.data());
      case 3:
        return expand3d_cpu<int8_t>(inData, padding.data());
      }
    }
    break;
  case torch::kInt16:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<int16_t>(inData, padding.data());
      case 3:
        return expand3d_cpu<int16_t>(inData, padding.data());
      }
    }
    break;
  case torch::kInt32:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<int32_t>(inData, padding.data());
      case 3:
        return expand3d_cpu<int32_t>(inData, padding.data());
      }
    }
    break;
  case torch::kInt64:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<int64_t>(inData, padding.data());
      case 3:
        return expand3d_cpu<int64_t>(inData, padding.data());
      }
    }
    break;
  case torch::kFloat32:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<float>(inData, padding.data());
      case 3:
        return expand3d_cpu<float>(inData, padding.data());
      }
    }
    break;
  case torch::kFloat64:
    {
      switch (padding.size()) {
      case 2:
        return expand2d_cpu<double>(inData, padding.data());
      case 3:
        return expand3d_cpu<double>(inData, padding.data());
      }
    }
    break;
  default:
    return torch::Tensor();
  }

  return torch::Tensor(); 
}

