# CycloFormer (supplementary)

Anonymous code and checkpoints for the three Table-1 models:


| Name                 | Params | CNN widths           | `d_model` | `d_ff` | heads | checkpoint                            |
| -------------------- | ------ | -------------------- | --------- | ------ | ----- | ------------------------------------- |
| `cycloformer_3p53m`  | 3.53M  | [32, 64, 128, 128]   | 224       | 512    | 8     | `checkpoints/cycloformer_3p53m.ckpt`  |
| `cycloformer_6p80m`  | 6.80M  | [64, 128, 256, 256]  | 256       | 768    | 8     | `checkpoints/cycloformer_6p80m.ckpt`  |
| `cycloformer_56p90m` | 56.90M | [128, 256, 512, 512] | 768       | 3072   | 24    | `checkpoints/cycloformer_56p90m.ckpt` |


All three use the 40x stem: kernels `[11, 5, 5, 3]`, strides `[5, 2, 2, 2]`,
`(N_spat, N_temp) = (3, 3)`, window `T = 10000` (`T' = 250`), dropout 0.15,
circular RoPE on the electrode ring, and attention pooling to 20 UmeTrack
angles. YAML files under `config/network/` and `config/experiment/` are the
training specifications that produced the checkpoints.

Layout:

```
emg2pose/train.py              training
emg2pose/test_analysis.py      official-split evaluation
emg2pose/cs_tds_ct_arch.py     model
emg2pose/data.py               HDF5 window loader
emg2pose/lightning.py          Lightning module + WindowedEmgDataModule
config/experiment/*.yaml       3.53M / 6.80M / 56.90M recipes
checkpoints/*.ckpt             val-MAE selected weights
```



## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10, PyTorch 2.3, PyTorch Lightning 2.2, Hydra 1.3.

## Data

Download the [emg2pose](https://github.com/facebookresearch/emg2pose) corpus
and point `data_location` at the directory that contains the per-session
HDF5 files.

This release includes `metadata_splits/metadata_100.csv`, the 100% split used
for the reported numbers. Either:

```bash
cp metadata_splits/metadata_100.csv $DATA/metadata.csv
```

or pass `metadata_file=` explicitly as in the commands below.

## Train

```bash
python -m emg2pose.train \
  experiment=cycloformer_3p53m \
  data_location=$DATA \
  metadata_file=metadata_splits/metadata_100.csv \
  num_workers=8

python -m emg2pose.train \
  experiment=cycloformer_6p80m \
  data_location=$DATA \
  metadata_file=metadata_splits/metadata_100.csv \
  num_workers=8

python -m emg2pose.train \
  experiment=cycloformer_56p90m \
  data_location=$DATA \
  metadata_file=metadata_splits/metadata_100.csv \
  num_workers=8
```

3.53M and 6.80M: AdamW `3e-4`, cosine 100 epochs, `eta_min=5e-6`, no warmup,
effective batch 192, SWA from epoch 85. 56.90M: same peak LR with a 5-epoch
warmup from 1% of peak and a 15-epoch input-augmentation ramp; effective
batch 256 on 2 GPUs (`devices=2, strategy=ddp`). Override with
`trainer.devices=1 trainer.strategy=auto trainer.accumulate_grad_batches=16`
to keep that batch on one GPU.

Checkpoints are written under the Hydra run directory
`logs/<date>/<time>/lightning_logs/`.

## Evaluate

```bash
python -m emg2pose.test_analysis \
  experiment=cycloformer_3p53m \
  checkpoint=checkpoints/cycloformer_3p53m.ckpt \
  data_location=$DATA \
  metadata_file=metadata_splits/metadata_100.csv \
  train=false eval=false \
  +split=test \
  ++conditions=[generalization,user] \
  batch_size=64 \
  num_workers=8
```

Replace the experiment name and checkpoint for 6.80M / 56.90M. Writes
`results.csv` in the Hydra working directory (angular MAE in radians;
degrees = rad * 180 / pi). `metadata_file=` is the split used by the
evaluator; it does not have to live inside `$DATA`.

## Checkpoints

Weights only (optimizer stripped). `load_from_checkpoint` still works
because Lightning hyperparameters are kept. SHA-256:

- `cycloformer_3p53m.ckpt` `199a3291fd90daae1893ef302d4172960237439aae049f3d6494728833053111`
- `cycloformer_6p80m.ckpt` `435b15538ba8183816d0bc25ebdb01413b2083eb5b0d363aa4bf121ab615040a`
- `cycloformer_56p90m.ckpt` `40640b26391cf2bd63aba5d90d9609a623c98bc007a987c31d718db4fc4169e1`

