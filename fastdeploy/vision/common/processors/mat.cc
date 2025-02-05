// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/utils/utils.h"
namespace fastdeploy {
namespace vision {

cv::Mat* Mat::GetCpuMat() {
  return &cpu_mat;
}

void Mat::ShareWithTensor(FDTensor* tensor) {
  tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                          GetCpuMat()->ptr());
  tensor->device = Device::CPU;
  if (layout == Layout::HWC) {
    tensor->shape = {Height(), Width(), Channels()};
  }
}

bool Mat::CopyToTensor(FDTensor* tensor) {
  cv::Mat* im = GetCpuMat();
  int total_bytes = im->total() * im->elemSize();
  if (total_bytes != tensor->Nbytes()) {
    FDERROR << "While copy Mat to Tensor, requires the memory size be same, "
               "but now size of Tensor = "
            << tensor->Nbytes() << ", size of Mat = " << total_bytes << "."
            << std::endl;
    return false;
  }
  memcpy(tensor->MutableData(), im->ptr(), im->total() * im->elemSize());
  return true;
}

void Mat::PrintInfo(const std::string& flag) {
  cv::Mat* im = GetCpuMat();
  cv::Scalar mean = cv::mean(*im);
  std::cout << flag << ": "
            << "Channel=" << Channels() << ", height=" << Height()
            << ", width=" << Width() << ", mean=";
  for (int i = 0; i < Channels(); ++i) {
    std::cout << mean[i] << " ";
  }
  std::cout << std::endl;
}

FDDataType Mat::Type() {
  int type = -1;
  type = cpu_mat.type();
  if (type < 0) {
    FDASSERT(false,
             "While calling Mat::Type(), get negative value, which is not "
             "expected!.");
  }
  type = type % 8;
  if (type == 0) {
    return FDDataType::UINT8;
  } else if (type == 1) {
    return FDDataType::INT8;
  } else if (type == 2) {
    FDASSERT(false,
             "While calling Mat::Type(), get UINT16 type which is not "
             "supported now.");
  } else if (type == 3) {
    return FDDataType::INT16;
  } else if (type == 4) {
    return FDDataType::INT32;
  } else if (type == 5) {
    return FDDataType::FP32;
  } else if (type == 6) {
    return FDDataType::FP64;
  } else {
    FDASSERT(
        false,
        "While calling Mat::Type(), get type = %d, which is not expected!.",
        type);
  }
}

Mat CreateFromTensor(const FDTensor& tensor) {
  int type = tensor.dtype;
  cv::Mat temp_mat;
  FDASSERT(tensor.shape.size() == 3,
           "When create FD Mat from tensor, tensor shape should be 3-Dim, HWC "
           "layout");
  int64_t height = tensor.shape[0];
  int64_t width = tensor.shape[1];
  int64_t channel = tensor.shape[2];
  switch (type) {
    case FDDataType::UINT8:
      temp_mat = cv::Mat(height, width, CV_8UC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT8:
      temp_mat = cv::Mat(height, width, CV_8SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT16:
      temp_mat = cv::Mat(height, width, CV_16SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT32:
      temp_mat = cv::Mat(height, width, CV_32SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::FP32:
      temp_mat = cv::Mat(height, width, CV_32FC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::FP64:
      temp_mat = cv::Mat(height, width, CV_64FC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    default:
      FDASSERT(
          false,
          "Tensor type %d is not supported While calling CreateFromTensor.",
          type);
      break;
  }
  Mat mat = Mat(temp_mat);
  return mat;
}

}  // namespace vision
}  // namespace fastdeploy
