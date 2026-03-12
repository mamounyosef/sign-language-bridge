from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="train_rgb_front_clips/*",
    local_dir=r"D:\How2Sign\train"
)

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="val_rgb_front_clips/*",
    local_dir=r"D:\How2Sign\val"
)

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="test_rgb_front_clips/*",
    local_dir=r"D:\How2Sign\test"
)