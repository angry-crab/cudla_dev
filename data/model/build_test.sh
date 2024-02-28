echo "Build test loadable for fp16"
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
${TRTEXEC} --onnx=data/model/net.onnx --fp16 --useDLACore=0 --buildDLAStandalone --saveEngine=data/loadable/net_fp16.standalone.bin  --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear
