#!/bin/bash

# Copyright 2019-2020 NVIDIA CORPORATION
# Copyright 2024 The Khronos Group, Inc.


set -e

PACKMAN_CMD="$(dirname "${BASH_SOURCE}")/packman"
if [ ! -f "$PACKMAN_CMD" ]; then
    PACKMAN_CMD="${PACKMAN_CMD}.sh"
fi
source "$PACKMAN_CMD" init
export PYTHONPATH="${PM_MODULE_DIR}:${PYTHONPATH}"

if [ -z "${PYTHONNOUSERSITE:-}" ]; then
    export PYTHONNOUSERSITE=1
fi

# For performance, default to unbuffered; however, allow overriding via
# PYTHONUNBUFFERED=0 since PYTHONUNBUFFERED on windows can truncate output
# when printing long strings
if [ -z "${PYTHONUNBUFFERED:-}" ]; then
    export PYTHONUNBUFFERED=1
fi

# workaround for our python not shipping with certs
if [[ -z ${SSL_CERT_DIR:-} ]]; then
    export SSL_CERT_DIR=/etc/ssl/certs/
fi

"${PM_PYTHON}" "$@"
