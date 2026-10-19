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

echo ""
echo "COMPARING IMAGES"
echo "----------------"
if [[ $(sha1sum gpu_image.ppm | cut -d' ' -f1) == $(sha1sum cpu_image.ppm | cut -d' ' -f1) ]]; then
  echo "[PASS] sha1sums match"
else
  echo "[FAIL] sha1sums don't match"
fi
