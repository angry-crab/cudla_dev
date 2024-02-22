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

CUDA_PATH ?= /usr/local/cuda
BUILD_DIR := ./build
SRCDIR := ./src

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

USE_DLA_STANDALONE_MODE := 1

ALL_CCFLAGS += --std=c++17 -Wno-deprecated-declarations -Wall

ifeq ($(USE_DLA_STANDALONE_MODE),1)
    ALL_CCFLAGS += -DUSE_DLA_STANDALONE_MODE
	ifeq ($(USE_DETERMINISTIC_SEMAPHORE),1)
		ALL_CCFLAGS += -DUSE_DETERMINISTIC_SEMAPHORE
	endif
endif

ifeq ($(DEBUG),1)
    ALL_CCFLAGS += -g
else
    ALL_CCFLAGS += -O2
endif

NVCC_FLAGS := -gencode arch=compute_87,code=sm_87

OPENCV_INCLUDE_PATH ?= /usr/include/opencv4/
OPENCV_LIB_PATH ?= /usr/lib/aarch64-linux-gnu/

INCLUDES += -I $(CUDA_PATH)/include \
            -I $(OPENCV_INCLUDE_PATH) \
            -I /usr/include/jsoncpp/ \
			-I /usr/include \
			-I /home/autoware/develop/nvsci_headers
LIBRARIES += -l cudla -L$(CUDA_PATH)/lib64 \
             -l cuda -l cudart -l nvinfer \
             -L $(OPENCV_LIB_PATH) \
	         -l opencv_objdetect -l opencv_highgui -l opencv_imgproc -l opencv_core -l opencv_imgcodecs -l opencv_dnn \
             -l jsoncpp \
			 -L /usr/lib/aarch64-linux-gnu/nvidia/ -lnvscibuf \
			 -L /usr/lib/aarch64-linux-gnu/nvidia/ -lnvscisync

CXXSRCS := $(wildcard $(SRCDIR)/*.cc)
CXXOBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(notdir $(CXXSRCS)))
NVCCSRCS := $(wildcard $(SRCDIR)/*.cu)
NVCCOBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(notdir $(NVCCSRCS)))
all: cudla_yoloxp

$(BUILD_DIR)/%.o: $(SRCDIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(INCLUDES) $(ALL_CCFLAGS) -c -o $@ $<
    @echo "Compiled cxx object file: $@ from $<"

$(BUILD_DIR)/%.o: $(SRCDIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) -c -o $@ $<
    @echo "Compiled nvcc object file: $@"

cudla_yoloxp: $(NVCCOBJS) $(CXXOBJS) | $(BUILD_DIR)
	$(CXX) $(ALL_CCFLAGS) $(INCLUDES) $(ALL_LDFLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES)

run_fp16: cudla_yoloxp
	./$(BUILD_DIR)/cudla_yoloxp --engine ./data/loadable/yoloxp.fp16.fp16chwin.fp16chwout.standalone.bin --image ./data/images/image.jpg --backend cudla_fp16

run_int8: cudla_yoloxp
	./$(BUILD_DIR)/cudla_yoloxp --engine ./data/loadable/yoloxp.int8.int8hwc4in.fp16chw16out.standalone.bin --image ./data/images/image.jpg --backend cudla_int8

validate_cudla_fp16: cudla_yoloxp
	./$(BUILD_DIR)/cudla_yoloxp --engine ./data/loadable/yoloxp.fp16.fp16chw16in.fp16chw16out.standalone.bin --coco_path ./data/coco/ --backend cudla_fp16
	python3 test_coco_map.py --predict predict.json --coco ./data/coco/

validate_cudla_int8: cudla_yoloxp
	./$(BUILD_DIR)/cudla_yoloxp --engine ./data/loadable/yoloxp.int8.int8hwc4in.fp16chw16out.standalone.bin --coco_path ./data/coco/ --backend cudla_int8
	python3 test_coco_map.py --predict predict.json --coco ./data/coco/

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/*

# export LD_LIBRARY_PATH=./src/matx_reformat/build/:$(OPENCV_LIB_PATH):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$LD_LIBRARY_PATH