"""Modal inference runtime. Match the detector versions used in remote/train.py.

Post-processing uses scipy (Hungarian stitching), installed as an ultralytics
dependency. ReID checkpoints are uploaded to the weights volume by infer.py, so
the image needs no network access to BoxMOT's model downloads.
"""

import modal

app = modal.App("nextup-player-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "ultralytics==8.4.155",
        "boxmot==25.0.0",
        "av==15.1.0",
        "opencv-python==4.14.0.94",
        "pydantic==2.13.4",
    )
    .add_local_python_source("tracking")
)

weights_volume = modal.Volume.from_name("nextup-weights", create_if_missing=True)
videos_volume = modal.Volume.from_name("nextup-videos", create_if_missing=True)
outputs_volume = modal.Volume.from_name("nextup-inference", create_if_missing=True)
