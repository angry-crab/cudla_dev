#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

echo "Build YoloXP DLA loadable for fp16 and int8"
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
${TRTEXEC} --onnx=data/model/modified_yolox-sPlus-T4-960x960-pseudo-finetune.onnx --fp16 --useDLACore=0 --buildDLAStandalone --saveEngine=data/loadable/yoloxp.fp16.fp16chwin.fp16chwout.standalone.bin  --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear
${TRTEXEC} --onnx=data/model/modified_yolox-sPlus-T4-960x960-pseudo-finetune.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=data/loadable/yoloxp.int8.int8chwin.fp16chwout.standalone.bin  --inputIOFormats=int8:dla_linear --outputIOFormats=fp16:dla_linear --int8 --fp16 --calib=data/model/yoloXP.cache --precisionConstraints=obey --layerPrecisions="/head/Concat_2":fp16,"/head/Concat_1":fp16,"/head/Concat":fp16
# "/head/reg_preds.2/Conv":fp16,"/head/Sigmoid_4":fp16,"/head/Sigmoid_5":fp16,"/head/reg_preds.1/Conv":fp16,"/head/Sigmoid_2":fp16,"/head/Sigmoid_3":fp16,"/head/reg_preds.0/Conv":fp16,"/head/Sigmoid":fp16,"/head/Sigmoid_1":fp16,
