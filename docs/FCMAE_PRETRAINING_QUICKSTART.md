# FCMAE Pretraining Quickstart

1. Put images somewhere under `data/`, for example:

```bash
mkdir -p data/fcmae_images
# copy .png/.jpg/.jpeg/.webp files into data/fcmae_images/
```

2. Check the Slurm command without submitting:

```bash
./pretrain_fcmae.sh submit --dry-run --time 01:00:00 -- \
  data.image_dir=data/fcmae_images \
  training.max_steps=10 \
  logging.wandb_enabled=false
```

3. Submit a short Slurm smoke run:

```bash
./pretrain_fcmae.sh submit --time 01:00:00 -- \
  data.image_dir=data/fcmae_images \
  training.max_steps=10 \
  logging.wandb_enabled=false
```

By default, `submit` creates a fresh directory under `weights/` using the configured run name plus
a timestamp, matching the main `train.sh` launcher.

4. Watch it:

```bash
./pretrain_fcmae.sh queue
./pretrain_fcmae.sh logs
```

5. Submit the real run:

```bash
./pretrain_fcmae.sh submit --time 48:00:00 -- \
  data.image_dir=data/fcmae_images \
  training.max_steps=200000 \
  logging.wandb_enabled=true
```

For a manifest instead of a folder:

```bash
./pretrain_fcmae.sh submit --time 48:00:00 --mem 64G -- \
  data.image_dir=null \
  data.manifest_path=data/raw/fcmae/real-scans/manifest.txt
```

6. Resume if needed:

```bash
./pretrain_fcmae.sh submit --resume weights/fcmae-real-scans/last.ckpt -- \
  data.image_dir=data/fcmae_images
```

That resumes from the checkpoint but writes into a new timestamped run directory. To continue writing
into the same directory, pass `checkpoint.dirpath=weights/fcmae-real-scans` explicitly.

If you set `export.export_on_train_end=true`, the wrapper defaults `export.output_dir` to
`exported_encoder` inside the generated checkpoint directory.

7. Export the encoder:

```bash
./pretrain_fcmae.sh export \
  weights/fcmae-real-scans/last.ckpt \
  weights/fcmae-real-scans/exported_encoder \
  --validate
```

8. Use the exported encoder in supervised training:

```bash
./train.sh submit -- \
  --model.encoder_model_name_or_path=weights/fcmae-real-scans/exported_encoder \
  --checkpoint.dirpath=weights/smt-fcmae-real-scans \
  --checkpoint.run_name=smt-fcmae-real-scans
```

The exported encoder is loaded through the existing `encoder_provider="transformers"` path.
