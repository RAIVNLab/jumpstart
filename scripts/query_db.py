#!/usr/bin/env python3
"""Print the mean of the top fraction of DB rewards for one algorithm/dataset."""

import math
import sqlite3
from pathlib import Path

import tyro


def fraction(value):
    value = float(value)
    if not 0 < value <= 1:
        raise ValueError("top fraction must be in (0, 1]")
    return value


def mean_top_fraction(scores, top_fraction=0.25):
    top_fraction = fraction(top_fraction)
    scores = sorted(score for score in scores if score is not None and math.isfinite(score))
    if not scores:
        raise ValueError("No finite trial rewards for this algorithm/dataset")
    count = max(1, int(len(scores) * top_fraction))
    return math.fsum(scores[-count:]) / count


def query_reward(db_path, algorithm, dataset, top_fraction=0.25):
    if not algorithm.isidentifier():
        raise ValueError(f"Invalid algorithm name: {algorithm}")
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(f'SELECT score FROM "{algorithm}" WHERE environment = ?', (dataset,))
        return mean_top_fraction((row[0] for row in rows), top_fraction)


def main(db: Path, algorithm: str, dataset: str, top_fraction: float = 0.25):
    """Print the DB mean; --top-fraction 0.1 uses the best 10% of finite trials."""
    print(query_reward(db, algorithm, dataset, top_fraction))


if __name__ == "__main__":
    tyro.cli(main)
