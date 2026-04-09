from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="train_rgb_front_clips/*",
    local_dir=r"C:\My Projects\sign-language-bridge\data\how2sign_raw_clips\train"
)
print("✅ Train clips downloaded successfully.")

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="val_rgb_front_clips/*",
    local_dir=r"C:\My Projects\sign-language-bridge\data\how2sign_raw_clips\val"
)
print("✅ Validation clips downloaded successfully.")

snapshot_download(
    repo_id="aipieces/How2Sign",
    repo_type="dataset",
    allow_patterns="test_rgb_front_clips/*",
    local_dir=r"C:\My Projects\sign-language-bridge\data\how2sign_raw_clips\test"
)
print("✅ Test clips downloaded successfully.")