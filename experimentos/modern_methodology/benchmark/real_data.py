"""Readers of the real datasets, from files downloaded once into data/real.

Every file is checked against its SHA-256 checksum before it is read, so that a
stream is the same on every machine that has the files. A reader returns the
features that the documentation of its dataset describes as measurements, the
labels as integers from zero and, for the dataset recorded in batches, the
batch of every sample. Symbolic features, binary indicators and the time of a
record are not measurements of the kind the models take, and are left out.
Every value must be a finite number; a file with any other value is an error,
not something to repair.

Sources:
- INSECTS (Souza et al., 2020), balanced versions, from the repository of the
  authors, by the links of River's `datasets.Insects`;
- Gas Sensor Array Drift (Vergara et al., 2012), UCI dataset 224;
- KDD Cup 1999, its 10% subset, and Forest Covertype, from the mirror that
  scikit-learn downloads, which publishes the same checksums;
- Pen-Based Recognition of Handwritten Digits, UCI dataset 81;
- Statlog (Shuttle), UCI dataset 148; its training file comes compressed by the
  Unix `compress`, which Python does not read, and is kept decompressed by
  `gzip -dc shuttle.trn.Z > shuttle.trn`.
"""

import gzip
import hashlib
import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REAL_DIR = Path(__file__).resolve().parent.parent / "data" / "real"

CHECKSUMS = {
    "insects_abrupt/abrupt_balanced.csv": "f368a6f4b7f28ce2e9aa0a9e542e1f6924999cae3e8607f0637549798cb7b94a",
    "insects_incremental/incremental_balanced.csv": "0f2a87e7f1bf9a2ede162deff83dec21fa39a3bdd1d3311c93301fea5c088a34",
    "insects_incremental_reoccurring/incremental_reoccurring_balanced.csv": "f267c0fb1f4e1d68967e2a694305146425181c19a7d3c282b67b252b98407c7a",
    "gas_drift/gas+sensor+array+drift+dataset.zip": "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4",
    "kddcup99/kddcup.data_10_percent.gz": "8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
    "covertype/covtype.data.gz": "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
    "pendigits/pen+based+recognition+of+handwritten+digits.zip": "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64",
    "shuttle/shuttle.trn": "87b24ee9fb5137e1d417659cf905d84d0e15342bbaa60770f1ae83da1a38200a",
    "shuttle/shuttle.tst": "f776934a628d9b94c482cb058a76ddaa63823e0f74aec065813c9dff3b89d661",
}

KDD_SYMBOLIC = (1, 2, 3)  # protocol_type, service, flag
KDD_BINARY = (6, 11, 20, 21)  # land, logged_in, is_host_login, is_guest_login
GAS_FEATURES = 128
GAS_BATCHES = 10


def checked(name: str) -> Path:
    """Path of a data file whose checksum matches the recorded one."""
    path = REAL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; see the sources in benchmark.real_data"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CHECKSUMS[name]:
        raise ValueError(f"{path} does not have the recorded checksum")
    return path


def _valid(X, labels, batches=None):
    X = np.ascontiguousarray(X, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("a dataset has a value that is not a finite number")
    y = np.unique(np.asarray(labels).astype(str), return_inverse=True)[1]
    return (X, y) if batches is None else (X, y, np.asarray(batches))


@lru_cache(maxsize=None)
def insects(name: str):
    data = pd.read_csv(checked(name), header=None)
    return _valid(data.iloc[:, :-1], data.iloc[:, -1])


@lru_cache(maxsize=None)
def gas_drift():
    """Samples in the order of the ten batches, and the batch of each."""
    rows, labels, batches = [], [], []
    with zipfile.ZipFile(
        checked("gas_drift/gas+sensor+array+drift+dataset.zip")
    ) as archive:
        for batch in range(1, GAS_BATCHES + 1):
            text = archive.read(f"Dataset/batch{batch}.dat").decode()
            for line in filter(str.strip, text.splitlines()):
                label, *entries = line.split()
                values = dict(entry.split(":") for entry in entries)
                if len(values) != GAS_FEATURES:
                    raise ValueError(
                        f"a line of batch {batch} lacks some of the features"
                    )
                rows.append([float(values[str(i)]) for i in range(1, GAS_FEATURES + 1)])
                labels.append(label.split(";")[0])
                batches.append(batch)
    return _valid(rows, labels, batches)


@lru_cache(maxsize=None)
def kddcup99():
    continuous = [c for c in range(41) if c not in KDD_SYMBOLIC + KDD_BINARY]
    with gzip.open(checked("kddcup99/kddcup.data_10_percent.gz"), "rt") as handle:
        data = pd.read_csv(handle, header=None, usecols=continuous + [41])
    return _valid(data[continuous], data[41])


@lru_cache(maxsize=None)
def covertype():
    """The ten quantitative features; the 44 binary indicators are left out."""
    with gzip.open(checked("covertype/covtype.data.gz"), "rt") as handle:
        data = pd.read_csv(handle, header=None)
    return _valid(data.iloc[:, :10], data[54])


@lru_cache(maxsize=None)
def pendigits():
    with zipfile.ZipFile(
        checked("pendigits/pen+based+recognition+of+handwritten+digits.zip")
    ) as archive:
        data = pd.concat(
            pd.read_csv(io.BytesIO(archive.read(member)), header=None)
            for member in ("pendigits.tra", "pendigits.tes")
        )
    return _valid(data.iloc[:, :16], data[16])


@lru_cache(maxsize=None)
def shuttle():
    """The eight attributes after the first, which is the time of the record."""
    data = pd.concat(
        pd.read_csv(checked(f"shuttle/shuttle.{part}"), header=None, sep=r"\s+")
        for part in ("trn", "tst")
    )
    return _valid(data.iloc[:, 1:9], data[9])
