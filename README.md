# VARE-Tracker

Implementation of VARE-Tracker: Efficient online point tracking with multiscale
representation and reliable interaction. The architecture combines
StandaloneLRGConservativeFPNEncoder, ConservativeGuideRoutingFPN (bilinear
upsampling), and reliability-aware motion-proxy interaction.

The model is `cotracker.models.core.cotracker.full.CoTrackerThreeOnline`.
The directory includes the local Kubric training entry point, dataset loaders,
losses, and evaluation code. The online branch of `train_on_kubric.py` has been
redirected to the combined model; the offline branch remains the original
offline implementation. The online evaluation entry point defaults to `full`.

This repository distributes source code. CoTracker-derived source
retains its original attribution and the accompanying LICENSE.md.

## Setup and execution

Run commands from the repository root. Install PyTorch and torchvision for your
CUDA runtime, then install `requirements.txt`.

Inspect training options with `python train_on_kubric.py --help`. The following
is a launch example:

```bash
python train_on_kubric.py --dataset_root /path/to/data --ckpt_path ./checkpoints/run --model_name cotracker_three --train_datasets kubric --linear_layer_for_vis_conf
```

Evaluate a trained checkpoint using the first-visible-frame protocol:

```bash
python -m cotracker.evaluation.evaluate --config-name eval_tapvid_davis_first dataset_root=/path/to/data/tapvid checkpoint=/path/to/model.pth online_variant=full
python -m cotracker.evaluation.evaluate --config-name eval_tapvid_robotap_first dataset_root=/path/to/data/tapvid checkpoint=/path/to/model.pth online_variant=full
```

Training takes the parent data directory; standalone evaluation takes its
`tapvid` subdirectory.

## Data

### Training: Kubric MOVi-F

- [Kubric source and MOVi datasets](https://github.com/google-research/kubric)
- [CoTracker training and annotation instructions](https://github.com/facebookresearch/co-tracker#training)
- [TAP-Vid and Kubric point-tracking tools](https://github.com/google-deepmind/tapnet)

CoTracker's training instructions require generating point-track annotations
for MOVi-F. The local loader expects the following path under
`--dataset_root`:

```text
kubric/kubric_movi_f_120_frames_dense/movi_f/
```

### Evaluation: TAP-Vid DAVIS-first and RoboTAP

- [Official TAP-Vid download and evaluation instructions](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid)
- [Official RoboTAP dataset download](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip)
- [RoboTAP project and dataset information](https://robotap.github.io/)

`get_eval_dataloader` in `cotracker/utils/train_utils.py` expects these paths
under `--dataset_root`:

```text
tapvid/tapvid_davis/tapvid_davis.pkl
tapvid/tapvid_robotap/
```

Use the first-visible-frame query protocol for DAVIS-first. Follow the official
download instructions and check the extracted dataset structure against the
loaders.

### Additional evaluation set

The paper also evaluates an author-curated set of 15 videos sourced from LaSOT
and other publicly available online videos, with point tracks annotated and
checked using CVAT and manual review. Inquiries about access should be addressed
to the corresponding author at taoleiyan@ioe.ac.cn. Availability of source
videos is subject to their respective owners' terms and permissions; access
arrangements must be confirmed by the authors.
