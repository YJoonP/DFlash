# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
# Licensed under the BSD-style license in the root LICENSE file.

"""Select recurrent matrix precision independently of convolution/KV state."""

import os


def recurrent_state_is_quantized():
    mode = os.environ.get("QNN_RECURRENT_STATE_DTYPE", "inherit")
    if mode not in ("inherit", "float32", "uint16"):
        raise ValueError("QNN_RECURRENT_STATE_DTYPE must be inherit, float32, or uint16")
    if mode == "inherit":
        return os.environ.get("QNN_STATE_IO_16BIT", "0") == "1"
    return mode == "uint16"
