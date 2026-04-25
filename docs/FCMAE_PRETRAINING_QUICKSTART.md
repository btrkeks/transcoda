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
  checkpoint.dirpath=weights/fcmae-smoke \
  training.max_steps=10 \
  logging.wandb_enabled=false
```

4. Watch it:

```bash
./pretrain_fcmae.sh queue
./pretrain_fcmae.sh logs
```

5. Submit the real run:

```bash
./pretrain_fcmae.sh submit --time 48:00:00 --mem 64G --cpus-per-task 8 -- \
  data.image_dir=data/fcmae_images \
  checkpoint.dirpath=weights/fcmae-real-scans \
  training.max_steps=200000 \
  logging.wandb_enabled=true \
  logging.run_name=fcmae-real-scans
```

For a manifest instead of a folder:

```bash
./pretrain_fcmae.sh submit --time 48:00:00 --mem 64G -- \
  data.image_dir=null \
  data.manifest_path=data/raw/fcmae/real-scans/manifest.txt \
  checkpoint.dirpath=weights/fcmae-real-scans
```

6. Resume if needed:

```bash
./pretrain_fcmae.sh submit --resume weights/fcmae-real-scans/last.ckpt -- \
  data.image_dir=data/fcmae_images \
  checkpoint.dirpath=weights/fcmae-real-scans
```

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
