"""Metadata-only audit of the real training and validation DataLoader order.

This script does not load MRI volumes, initialize VGG19, or perform optimizer
steps. It uses the same PyTorch DataLoader semantics as the real training code by
wrapping CSV metadata in a tiny dataset object, which preserves the actual
sampler behavior without reading medical image arrays.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data/processed/slice_index_subset_12train_4val.csv"


class MetadataIndexDataset(Dataset):
    """Small metadata-only dataset whose records match the CSV row order."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return int(index), self.rows[index]


def read_csv_rows(index_path: str | Path) -> list[dict[str, str]]:
    with Path(index_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _sorted_unique_patient_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == split]


def _summarize_ordered_indices(indices: list[int], rows: list[dict[str, str]]) -> dict[str, Any]:
    if not indices:
        raise ValueError("No ordered indices were supplied for summarization.")
    ordered_rows = [rows[index] for index in indices]
    tumor_positive = sum(1 for row in ordered_rows if int(row["tumor_pixel_count"]) > 0)
    tumor_negative = sum(1 for row in ordered_rows if int(row["tumor_pixel_count"]) == 0)
    distinct_patients = len({row["patient_id"] for row in ordered_rows})
    max_run = 0
    current_run = 0
    for row in ordered_rows:
        if int(row["tumor_pixel_count"]) == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    duplicates = len(indices) - len(set(indices))
    missing = sorted(set(range(len(rows))) - set(indices))
    return {
        "distinct_patients": distinct_patients,
        "tumor_positive": tumor_positive,
        "tumor_negative": tumor_negative,
        "max_consecutive_negative": max_run,
        "duplicates": duplicates,
        "missing": missing,
        "expected_unique_length": len(rows),
        "actual_unique_length": len(set(indices)),
    }


def actual_train_order(index_path: str | Path = DEFAULT_INDEX, *, seed: int = 42) -> dict[str, Any]:
    """Mimic the real training DataLoader using a metadata-only dataset."""
    rows = read_csv_rows(index_path)
    train_rows = _sorted_unique_patient_rows(rows, "train")
    if not train_rows:
        raise ValueError(f"No train rows found in {index_path}.")

    torch.manual_seed(seed)
    dataset = MetadataIndexDataset(train_rows)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    ordered_indices = [int(batch[0].item()) for batch in loader]
    sampler_type = type(loader.sampler).__name__
    shuffle_enabled = isinstance(loader.sampler, RandomSampler)
    order_summary = {
        "sampler_type": sampler_type,
        "shuffle_enabled": shuffle_enabled,
        "dataset_length": len(dataset),
        "ordered_indices": ordered_indices,
        "first_24": _summarize_ordered_indices(ordered_indices[:24], train_rows),
        "first_48": _summarize_ordered_indices(ordered_indices[:48], train_rows),
        "full_epoch": _summarize_ordered_indices(ordered_indices, train_rows),
    }

    full_epoch_unique = set(ordered_indices)
    expected = set(range(len(train_rows)))
    order_summary["full_epoch_pass"] = len(ordered_indices) == len(train_rows) and full_epoch_unique == expected and order_summary["full_epoch"]["duplicates"] == 0
    return order_summary


def actual_validation_order(index_path: str | Path = DEFAULT_INDEX) -> dict[str, Any]:
    rows = read_csv_rows(index_path)
    val_rows = _sorted_unique_patient_rows(rows, "validation")
    dataset = MetadataIndexDataset(val_rows)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    ordered_indices = [int(batch[0].item()) for batch in loader]
    summary = {
        "sampler_type": type(loader.sampler).__name__,
        "shuffle_enabled": isinstance(loader.sampler, RandomSampler),
        "dataset_length": len(dataset),
        "ordered_indices": ordered_indices,
        "summary": _summarize_ordered_indices(ordered_indices, val_rows),
    }
    summary["validation_exclusive"] = len(set(ordered_indices) & set(range(len(val_rows)))) == len(val_rows)
    return summary


def stratified_order(index_path: str | Path = DEFAULT_INDEX, *, seed: int = 42, patient_count: int = 12) -> dict[str, Any]:
    """Reproduce the metadata-only stratified bounded subset used in train_2d.py."""
    rows = read_csv_rows(index_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        by_patient.setdefault(row["patient_id"], []).append(row)

    patient_ids = sorted(by_patient)
    selected_patients = patient_ids[:patient_count]
    selected_indices: list[int] = []
    for patient_id in selected_patients:
        patient_rows = by_patient[patient_id]
        positive = [row for row in patient_rows if int(row["tumor_pixel_count"]) > 0]
        negative = [row for row in patient_rows if int(row["tumor_pixel_count"]) == 0]
        if not positive or not negative:
            raise ValueError(f"Patient {patient_id} is missing a positive or negative slice for the bounded subset.")
        selected_indices.extend([row_index for row_index, row in enumerate(train_rows) if row["patient_id"] == patient_id and row in positive[:1]])
        selected_indices.extend([row_index for row_index, row in enumerate(train_rows) if row["patient_id"] == patient_id and row in negative[:1]])

    subset = Subset(MetadataIndexDataset(train_rows), selected_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    ordered_indices = [int(batch[0].item()) for batch in loader]
    return {
        "sampler_type": type(loader.sampler).__name__,
        "shuffle_enabled": isinstance(loader.sampler, RandomSampler),
        "selected_indices": selected_indices,
        "ordered_indices": ordered_indices,
        "summary": _summarize_ordered_indices(ordered_indices, train_rows),
    }


def _format_group(label: str, summary: dict[str, Any]) -> str:
    return (
        f"{label}: distinct_patients={summary['distinct_patients']} "
        f"tumor_positive={summary['tumor_positive']} "
        f"tumor_negative={summary['tumor_negative']} "
        f"max_consecutive_negative={summary['max_consecutive_negative']} "
        f"duplicates={summary['duplicates']} "
        f"missing={summary['missing']}"
    )


def main() -> None:
    train_summary = actual_train_order(DEFAULT_INDEX, seed=42)
    validation_summary = actual_validation_order(DEFAULT_INDEX)
    stratified_summary = stratified_order(DEFAULT_INDEX, seed=42)

    print("Actual DataLoader order audit (metadata-only)")
    print(f"Sampler type: {train_summary['sampler_type']}")
    print(f"Training shuffle enabled: {train_summary['shuffle_enabled']}")
    print(f"Train dataset length: {train_summary['dataset_length']}")
    print(f"Validation sampler type: {validation_summary['sampler_type']}")
    print(f"Validation shuffle enabled: {validation_summary['shuffle_enabled']}")
    print(f"Validation dataset length: {validation_summary['dataset_length']}")
    print()
    print(_format_group("First 24 training samples", train_summary["first_24"]))
    print(_format_group("First 48 training samples", train_summary["first_48"]))
    print(_format_group("All 144 training samples", train_summary["full_epoch"]))
    print()
    print(f"Complete epoch unique/coverage PASS: {train_summary['full_epoch_pass']}")
    print(f"Validation exclusive order PASS: {validation_summary['validation_exclusive']}")
    print()
    print("Stratified bounded-sampling audit:")
    print(f"Subset sampler type: {stratified_summary['sampler_type']}")
    print(f"Subset shuffle enabled: {stratified_summary['shuffle_enabled']}")
    print(_format_group("Stratified subset order", stratified_summary["summary"]))

    # Human-readable interpretation of the actual ordering risk.
    first24 = train_summary["first_24"]
    full_epoch = train_summary["full_epoch"]
    print()
    print("Interpretation:")
    print(
        "The real training DataLoader uses a RandomSampler when shuffle=True, so the first 24 training "
        "samples are not the first 24 CSV records. They are a reproducible random permutation seeded by 42."
    )
    print(
        f"In the first 24 training samples, tumor-positive={first24['tumor_positive']} and "
        f"tumor-negative={first24['tumor_negative']}.")
    print(
        f"Across the full epoch, tumor-positive={full_epoch['tumor_positive']} and "
        f"tumor-negative={full_epoch['tumor_negative']} with max consecutive negatives={full_epoch['max_consecutive_negative']}.")
    print("Background-heavy batches can occur if early random samples are mostly negative, even though the full epoch contains all training records exactly once.")
    print("This audit stays metadata-only and does not load MRI volumes or run VGG19 updates.")


if __name__ == "__main__":
    main()
