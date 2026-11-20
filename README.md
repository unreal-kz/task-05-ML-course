# Task 5 — Clustering & PCA

Starter files for **Task 5** (graded, 25 points, due Sunday 6 December 2026, 23:59 Canvas time, UTC+5)
of *Machine Learning (Python)*, MOP 3231, Narxoz University.

The task text, the six parts, what to submit and the grading table are in the handout in the **Week 12** module
in Canvas. This repository holds only the files you start from; it has no instructions of its own beyond the setup
below.

## What is here

| File | What it is |
|---|---|
| `data/drybean.csv` | The dataset: 13,543 dry bean grains described by 16 shape measurements, and the column `Class` (the bean variety, 7 values) |
| `requirements.txt` | Pinned library versions the task is written and checked against. Do not change them |
| `make_dataset.py` | How `drybean.csv` was built from the UCI source. You do not need to run it |
| `.gitignore` | Keeps your virtual environment and notebook checkpoints out of git |

## How to start

You do not push to this repository. You make **your own** repository, as in Tasks 1–4, and submit the URL of its
final commit in Canvas.

1. Download this repository as a ZIP (**Code → Download ZIP**), unzip it, and rename the unzipped folder (GitHub names it
   `task-05-ML-course-main`) to `lastname_ML_task_5`. `README.md` and `data/` must sit directly inside it.
2. Create the virtual environment and install the pinned libraries. You need **Python 3.11, 3.12, 3.13 or 3.14**
   (check with `python3 --version`; on Windows `python --version`):

   ```bash
   cd lastname_ML_task_5
   python3 -m venv .lastname_ML_task_5
   source .lastname_ML_task_5/bin/activate      # Windows PowerShell: .lastname_ML_task_5\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Work in a notebook called `lastname_task5.ipynb` in the root of the folder, next to `data/`.
4. `git init`, commit as you go, push to your own GitHub repository and submit the final commit URL.
   What must be in the repository is in the handout.

## The data

`data/drybean.csv` is derived from the **Dry Bean** dataset of the UCI Machine Learning Repository, licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
(licence and DOI as stated on the UCI dataset page).

> Koklu, M., & Ozkan, I. A. (2020). Dry Bean [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C50S4B

**Changes made to the original file** (`data.csv` from the UCI dataset page):

- Column names are exactly as in the source: 16 numeric features (`Area`, `Perimeter`, `MajorAxisLength`,
  `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`,
  `Compactness`, `ShapeFactor1` … `ShapeFactor4`) and the text label `Class`.
- Raw rows: 13,611. **68** exact duplicate rows (all 17 values equal) were dropped, first copy kept, leaving **13,543** rows:
  DERMASON 3,546, SIRA 2,636, SEKER 2,027, HOROZ 1,860, CALI 1,630, BARBUNYA 1,322, BOMBAY 522.
- No feature vector occurs under two different labels (checked; the build script stops if one ever did).
- The rows were shuffled once with the fixed seed 3231 (the raw file lists the varieties one after another).

Nothing else was changed: values are not scaled, clipped or rounded.

`Class` is not something to predict in this task. How it is used is in the handout.

This is a teaching dataset of one laboratory's measurements, not evidence about bean quality in general.
