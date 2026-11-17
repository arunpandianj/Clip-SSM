# CLiP-SSM

Code for the paper *CLiP-SSM: a hold-aware state-space generator with clinical constraints for synthetic intensive care time series* (Arun Pandian J and Shareena Basheer).

CLiP-SSM generates 48-hour hourly intensive care series. It combines a latent state-space model with a hold-aware observation model, a decoder bounded to physiological ranges, a differentiable clinical rule penalty, marginal, correlation, dynamics and cross-window matching terms, and a nearest-record leakage margin. This repository also contains the nine comparison methods, the evaluation code, the ablations and the analysis scripts used for the paper.

## Repository layout

| Path | Content |
|---|---|
| `clip_ssm.py` | Single-file implementation: data loaders, model, baselines, training, evaluation, ablations, figures |
| `notebooks/` | Kaggle notebooks used for the reported runs (PhysioNet 2012 final run and PhysioNet 2019 run) |
| `results/` | Per-seed result tables and the saved real and generated samples of both datasets |
| `analysis/` | Scripts that rebuild the paper tables, the Welch and Holm statistics and the figures from `results/` |
| `requirements.txt` | Python dependencies |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly advised. The reported runs used a single Kaggle GPU. On PhysioNet 2012 one CLiP-SSM training run took about 980 seconds, and the complete five-seed run with ablations took about 7.8 hours.

## Data

The data are openly licensed and are not redistributed here. Download them and keep the files in any folder structure under one directory, which you pass as `DATA_ROOT`.

1. **PhysioNet/Computing in Cardiology Challenge 2012.** Download the patient files of sets A and B (`set-a`, `set-b`, text files named by record number) from https://physionet.org/content/challenge-2012/1.0.0/. The `Outcomes-*.txt` files are ignored.
2. **PhysioNet/Computing in Cardiology Challenge 2019 (sepsis).** Download the training sets A and B (`.psv` files, 40,336 patients) from https://physionet.org/content/challenge-2019/1.0.0/ (DOI 10.13026/v64v-d857).

Please cite the dataset papers and PhysioNet as requested on the dataset pages.

## Usage

Smoke test on simulated data, a few minutes on a GPU:

```bash
SMOKE=1 python clip_ssm.py
```

Reproduce the PhysioNet 2012 results of the paper:

```bash
DATASET=physionet2012 DATA_ROOT=/path/to/data OUTPUT_DIR=outputs/p2012 python clip_ssm.py
```

Reproduce the PhysioNet 2019 results, with a random subset of 8,000 stays and no ablations:

```bash
DATASET=physionet2019 DATA_ROOT=/path/to/data OUTPUT_DIR=outputs/p2019 N_MAX=8000 RUN_ABLATIONS=0 python clip_ssm.py
```

The script stops with an error if the dataset is missing or has fewer than 200 usable stays. It never falls back to simulated data outside smoke mode.

Environment variables: `DATASET` (`physionet2012`, `physionet2019` or `mimic3`), `DATA_ROOT`, `OUTPUT_DIR`, `N_MAX` (maximum number of stays), `SEEDS` (comma separated, default `0,1,2,3,4`), `RUN_ABLATIONS` (`1` or `0`), `EP_CLIP` and `EP_BASE` (epochs), `N_SYN` (number of generated sequences, default 5,000), `SMOKE`.

The `mimic3` loader is included for completeness. It was not used for any result in the paper.

## What the script does

1. Builds hourly 48-hour sequences with nine variables, scaled to the unit interval with physiological bounds, and splits them 70, 15 and 15 percent at random.
2. Tunes the learning rates of the gradient-trained baselines and two loss weights of CLiP-SSM on the validation set.
3. Trains CLiP-SSM and the baselines for each seed, generates 5,000 sequences per method and evaluates them against the test stays.
4. Writes `results_all_<dataset>.csv` with all measures for every method and seed, the saved samples, the model checkpoint and the figures.

The evaluation measures are defined in the paper: Kolmogorov-Smirnov statistic, correlation preservation, autocorrelation curve error, dynamic time warping, skewness and kurtosis errors, rule compliance and its gap to real data, raw bounds adherence, the calibrated membership attack with subgroup results, and train-on-synthetic and test-on-real utility.

## Rebuilding the paper tables and figures

```bash
cd analysis
python make_tables.py      # table rows and Welch tests with Holm correction
python make_figures.py     # trajectories, autocorrelation curves, per-variable KS (600 dpi)
```

## Notes on the comparison methods

All baselines are re-implementations from the published descriptions, sized to train within the compute budget. They are not the official code. The two diffusion models are compact versions of the TimeDiff and Diffusion-TS designs. The DG-HMM model is rebuilt from the text of its paper. Results of these re-implementations should not be compared with the numbers in the original papers.

## Known limitations

See the Discussion of the paper. In short: two datasets, regimes that act as patient-level clusters, a hold gate that freezes some densely measured variables, no formal privacy guarantee, and training that is about ten times slower than the diffusion baselines.

## License

MIT, see `LICENSE`. The datasets have their own licenses.
