# cuDLA YoloX


FP16 dla_linear OK


# Env


Orin


Docker image (nvcr.io/nvidia/l4t-jetpack:r36.2.0)


# Dependencies


`sudo apt update`


`sudo apt install libopencv-dev libjsoncpp-dev python3-pip git git-lfs`


# YOLOX ONNX


ONNX file is modified from `yolox-sPlus-T4-960x960-pseudo-finetune.onnx` by removing Reshape and following layers.

![image](figures/Screenshot.png.png "Remove")


# Instructions


### FP16


`bash data/model/build_dla_standalone.sh`

`make run_fp16`