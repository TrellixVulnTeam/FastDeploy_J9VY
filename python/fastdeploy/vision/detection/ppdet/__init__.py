# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import logging
from .... import FastDeployModel, Frontend
from .... import c_lib_wrap as C


class PPYOLOE(FastDeployModel):
    """PaddleDetection PPYOLOE class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g ppyoloe/ppyoloe/model.pdmodel
        :param params_file: (str) Path of parameters file e.g ppyoloe/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLOE model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PPYOLOE(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLOE model initialize failed."

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (np.ndarray) input image data as 3-D np.ndarray dtype, represent a BGR image with HWC data layout
        :return: DetectionResult
        """
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)


class PPYOLO(PPYOLOE):
    """PaddleDetection PPYOLO class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g ppyolo/ppyolo/model.pdmodel
        :param params_file: (str) Path of parameters file e.g ppyolo/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g ppyolo/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLO model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PPYOLO(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLO model initialize failed."


class PPYOLOv2(PPYOLOE):
    """PaddleDetection PPYOLOv2 class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g ppyolov2/model.pdmodel
        :param params_file: (str) Path of parameters file e.g ppyolov2/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g ppyolov2/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLOv2 model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PPYOLOv2(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLOv2 model initialize failed."


class PaddleYOLOX(PPYOLOE):
    """PaddleDetection PaddleYOLOX class
       This class support deploy the YOLOX model implemented in PaddleDetection."
    """

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g paddle_yolox/model.pdmodel
        :param params_file: (str) Path of parameters file e.g paddle_yolox/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g paddle_yolox/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PaddleYOLOX model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PaddleYOLOX(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleYOLOX model initialize failed."


class PicoDet(PPYOLOE):
    """PaddleDetection PicoDet class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g picodet/model.pdmodel
        :param params_file: (str) Path of parameters file e.g picodet/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g picodet/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PicoDet model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PicoDet(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PicoDet model initialize failed."


class FasterRCNN(PPYOLOE):
    """PaddleDetection FasterRCNN class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g faster_rcnn/model.pdmodel
        :param params_file: (str) Path of parameters file e.g faster_rcnn/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g faster_rcnn/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "FasterRCNN model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.FasterRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "FasterRCNN model initialize failed."


class YOLOv3(PPYOLOE):
    """PaddleDetection YOLOv3 class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g yolov3/model.pdmodel
        :param params_file: (str) Path of parameters file e.g yolov3/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g yolov3/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "YOLOv3 model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.YOLOv3(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "YOLOv3 model initialize failed."


class MaskRCNN(PPYOLOE):
    """PaddleDetection MaskRCNN class"""

    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        """
        :param model_file: (str) Path of model file e.g maskrcnn/model.pdmodel
        :param params_file: (str) Path of parameters file e.g maskrcnn/model.pdiparams
        :param config_file: (str) Path of inference configuration file, e.g maskrcnn/infer_cfg.yml
        :param runtime_option: (RuntimeOption) Option to configure how to run the model
        :param model_format: support Frontend.PADDLE/Frontend.ONNX, while in ONNX format, only need to set the model_file, and set the params_file to empty string ''
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "MaskRCNN model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.MaskRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "MaskRCNN model initialize failed."
