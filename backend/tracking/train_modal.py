import modal

app = modal.App("nextup-player-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "ultralytics==8.4.155",
    )
)
datasets = modal.Volume.from_name("nextup-datasets")
checkpoints = modal.Volume.from_name("nextup-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=8 * 60 * 60,
    volumes={
        "/datasets": datasets,
        "/checkpoints": checkpoints,
    },
)
def train(smoke: bool = False):
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")

    model.add_callback(
        "on_model_save",
        lambda trainer: checkpoints.commit(),
    )

    try:
        model.train(
            data="/datasets/players_v1/data.yaml",
            epochs=2 if smoke else 60,
            fraction=0.05 if smoke else 1.0,
            imgsz=640 if smoke else 1280,
            batch=4,
            device=0,
            workers=4,
            patience=15,
            close_mosaic=0 if smoke else 10,
            flipud=0.0,
            seed=42,
            project="/checkpoints",
            name="smoke" if smoke else "players-v1",
        )
        print(f"Best weights: {model.trainer.best}")
    finally:
        checkpoints.commit()
