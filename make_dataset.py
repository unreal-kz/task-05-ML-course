"""Build the Task 5 data: UCI Dry Bean, deduplicated and shuffled.

Source: UCI Machine Learning Repository, "Dry Bean" (dataset 602), file ``data.csv`` from
    https://archive.ics.uci.edu/static/public/602/data.csv
Licence: CC BY 4.0 (stated on the dataset page). Citation: Koklu, M., & Ozkan, I. A. (2020).
Dry Bean [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B

Transformations, in order:
    1. Parse ``data.csv`` (header, comma separated): 13,611 rows, 16 numeric shape features and
       the text label ``Class`` (7 bean varieties). Column names are kept exactly as in the source.
    2. Drop exact duplicate rows (all 17 values equal), keeping the first copy.
    3. Check that no feature vector occurs under two different ``Class`` labels (none does in the
       UCI file; the script stops if one ever did).
    4. Shuffle the rows once with the fixed seed ``SHUFFLE_SEED`` (the raw file is sorted by
       class: all SEKER rows come first).
    5. Nothing else is changed: values are not scaled, clipped or rounded.

The counts of every step are printed; none of them is assumed.

Output (next to this script):
    data/drybean.csv     16 features + Class (header, no index)   -> students

The file is one table, not a train/test pair. ``Class`` is NOT a target to predict: in Task 5 it
is used only to judge clusters after they are found.

Run:
    python make_dataset.py                       # downloads the UCI file
    python make_dataset.py --csv data_raw.csv    # use a local copy of data.csv
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/602/data.csv"

LABEL = "Class"
N_FEATURES = 16
N_CLASSES = 7
SHUFFLE_SEED = 3231  # course code MOP 3231

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def read_source(url: str = UCI_URL, local_csv: Path | None = None) -> bytes:
    """Get the bytes of the UCI ``data.csv``.

    Parameters
    ----------
    url : str
        Location of the UCI Dry Bean CSV. Ignored when ``local_csv`` is given.
    local_csv : pathlib.Path or None
        A previously downloaded copy of the file.

    Returns
    -------
    bytes
        The content of the CSV.
    """
    if local_csv is not None:
        return local_csv.read_bytes()
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def build_frame(csv_bytes: bytes) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse, deduplicate, check label conflicts and shuffle.

    Parameters
    ----------
    csv_bytes : bytes
        Content of ``data.csv``.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The final table (16 features in source order, then ``Class``) and the counts of every
        step: ``n_raw``, ``n_duplicates_removed``, ``n_rows``.

    Raises
    ------
    ValueError
        If the column count is not ``N_FEATURES + 1``, the last column is not ``Class``, or a
        feature vector occurs under more than one label.
    """
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    if raw.shape[1] != N_FEATURES + 1 or raw.columns[-1] != LABEL:
        raise ValueError(f"expected {N_FEATURES} features followed by '{LABEL}', got {list(raw.columns)}")
    features = list(raw.columns[:-1])

    deduplicated = raw.drop_duplicates().reset_index(drop=True)
    labels_per_row = deduplicated.groupby(features)[LABEL].transform("nunique")
    if int((labels_per_row > 1).sum()) > 0:
        raise ValueError("a feature vector occurs under more than one Class label")
    shuffled = deduplicated.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)
    counts = {
        "n_raw": len(raw),
        "n_duplicates_removed": len(raw) - len(deduplicated),
        "n_rows": len(shuffled),
    }
    return shuffled, counts


def check_frame(frame: pd.DataFrame) -> None:
    """Assert the invariants the handout relies on.

    Raises
    ------
    AssertionError
        If values are missing, the label does not have ``N_CLASSES`` distinct values, a feature
        is not numeric, or a row (or a feature vector) is duplicated.
    """
    features = [c for c in frame.columns if c != LABEL]
    assert len(features) == N_FEATURES and list(frame.columns) == features + [LABEL]
    assert not frame.isna().any().any()
    assert frame[LABEL].nunique() == N_CLASSES
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in features)
    assert int(frame.duplicated().sum()) == 0
    assert int(frame.duplicated(subset=features).sum()) == 0


def main() -> None:
    """Fetch the UCI source, build and verify the table, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=None, help="local copy of data.csv (skips the download)")
    args = parser.parse_args()

    frame, counts = build_frame(read_source(local_csv=args.csv))
    check_frame(frame)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / "drybean.csv", index=False)

    print(f"raw rows: {counts['n_raw']}")
    print(f"exact duplicate rows removed: {counts['n_duplicates_removed']}")
    print(f"final rows: {counts['n_rows']}  features: {frame.shape[1] - 1}")
    print("class counts:", frame[LABEL].value_counts().to_dict())
    print(f"shuffle seed: {SHUFFLE_SEED}")
    print(f"wrote {DATA_DIR / 'drybean.csv'}")


if __name__ == "__main__":
    main()
