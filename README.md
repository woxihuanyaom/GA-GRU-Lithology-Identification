# GA-GRU-Lithology-Identification
GA‑tuned GRU workflow for within‑well lithology identification, Interpretation manuscript code
# GA‑GRU‑Workflow for within‑well lithology identification
This repository contains the source code for the manuscript:
> A GA‑tuned GRU workflow for within‑well lithology identification in the Hailar Basin, *Interpretation*, SEG.

## Overview
This workflow implements genetic‑algorithm‑tuned GRU for lithology interpolation inside individual logged wells.
It includes feature engineering for log‑derived electrical contrasts, SMOTE‑Tomek training‑only resampling,
hyperparameter search (GA, random search, TPE), component ablation, and evaluation scripts for random‑center and complete‑interval test design.

> Note: **Original proprietary oil‑field well‑log and lithology labels are NOT included in this repository**, due to data confidentiality restrictions.
> A small simulated demo input sample is provided under `demo_input/` for code syntax verification. Real field data requires formal application to the corresponding oil‑field institute.

## Environment
Python 3.12.7, see `requirements.txt` for full dependency list.
```bash
pip install -r requirements.txt
