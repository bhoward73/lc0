#!/usr/bin/env bash

set -euo pipefail

CDPATH= cd -- "$(dirname -- "$0")"

case "${1:-}" in
  plain|debug|debugoptimized|release|minsize)
    BUILDTYPE="$1"
    shift
    ;;
  *)
    BUILDTYPE="release"
    ;;
esac

BUILDDIR="${BUILDDIR:-build/${BUILDTYPE}}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-10}"
ONNX_INCLUDE="${ONNX_INCLUDE:-/opt/onnxruntime/include}"
ONNX_LIBDIR="${ONNX_LIBDIR:-/opt/onnxruntime/lib}"
NATIVE_CUDA="${NATIVE_CUDA:-false}"
TARGET="${TARGET:-lc0}"

MESON=$(PATH="${PATH}:${HOME}/.local/bin" command -v meson || :)
MESON=${MESON:?"Could not find meson. Is it installed and in PATH?"}

if [ -f "${BUILDDIR}/build.ninja" ]; then
  "${MESON}" setup "${BUILDDIR}" --reconfigure \
    --buildtype "${BUILDTYPE}" \
    --prefix "${INSTALL_PREFIX}" \
    -Dnvcc_ccbin="${NVCC_CCBIN}" \
    -Dnative_cuda="${NATIVE_CUDA}" \
    -Donnx_include="${ONNX_INCLUDE}" \
    -Donnx_libdir="${ONNX_LIBDIR}" \
    "$@"
else
  "${MESON}" setup "${BUILDDIR}" \
    --buildtype "${BUILDTYPE}" \
    --prefix "${INSTALL_PREFIX}" \
    -Dnvcc_ccbin="${NVCC_CCBIN}" \
    -Dnative_cuda="${NATIVE_CUDA}" \
    -Donnx_include="${ONNX_INCLUDE}" \
    -Donnx_libdir="${ONNX_LIBDIR}" \
    "$@"
fi

"${MESON}" compile -C "${BUILDDIR}" "${TARGET}"
