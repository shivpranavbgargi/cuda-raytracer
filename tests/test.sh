#!/bin/bash

pushd "$WORKDIR" || exit
echo "COMPILING"
echo "---------"
make clean && make
pushd "bin" || exit
echo ""
echo "EXECUTING"
echo "---------"
./"${CPU_BIN}" && ./"${GPU_BIN}"
popd || exit
pushd "results" || exit
# cat "" > gpu_image.ppm
# cat "" > cpu_image.ppm
echo ""
echo "COMPARING IMAGES"
echo "----------------"
if [[ $(sha1sum gpu_image.ppm | cut -d' ' -f1) == $(sha1sum cpu_image.ppm | cut -d' ' -f1) ]]; then
  echo "[PASS]"
else
  echo "[FAIL]"
fi
