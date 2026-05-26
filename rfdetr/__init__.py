# Copyright 2024 Roboflow Inc.
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

"""RF-DETR: Real-time object detection with DEtection TRansformer.

This package provides a high-performance implementation of RF-DETR,
an object detection model based on the DETR (DEtection TRansformer)
architecture, optimized for real-time inference.

Example usage::

    from rfdetr import RFDETRBase

    model = RFDETRBase()
    detections = model.predict("image.jpg")

    # To predict with a custom confidence threshold:
    detections = model.predict("image.jpg", threshold=0.4)

    # To use the larger model variant for better accuracy:
    from rfdetr import RFDETRLarge
    model = RFDETRLarge()

    # To export the model to ONNX format:
    model.export(format="onnx")

    # To export the model to TensorRT format (requires TensorRT installed):
    model.export(format="tensorrt")

    # To train on a custom dataset (COCO format expected):
    model.train(dataset_dir="/path/to/dataset", epochs=50)

    # Quick way to check the package version at runtime:
    import rfdetr
    print(rfdetr.__version__)
"""

from rfdetr.main import RFDETRBase, RFDETRLarge

__version__ = "1.0.0"
__author__ = "Roboflow Inc."
__all__ = ["RFDETRBase", "RFDETRLarge"]

# Default confidence threshold used across predict() calls.
# Lowering this value (e.g. 0.3) surfaces more low-confidence detections,
# which can be useful during dataset exploration or debugging.
# NOTE: Lowered from 0.5 to 0.35 for my use case — I'm working with a
# dataset that has many small/occluded objects where the model tends to
# under-predict at the default threshold.
DEFAULT_THRESHOLD = 0.35

# Default resolution used when no resolution is specified for inference.
# RF-DETR expects square inputs; 560 is a good balance between speed and
# accuracy for my hardware (RTX 3060). Use 640 for slightly better mAP
# at the cost of ~15% slower inference.
DEFAULT_RESOLUTION = 560
