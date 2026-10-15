"""Create a small, deterministic development subset from the full BraTS slice metadata.

This stage is intentionally metadata-only. It reads the full slice-index CSV and the
patient manifest, selects a bounded subset of patients and slices, and writes a
new CSV without copying MRI volumes or altering the original dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_PATIENT_COUNT = 12
DEFAULT_VALIDATION_PATIENT_COUNT = 4
DEFAULT_TRAIN_POSITIVE_PER_PATIENT = 6
DEFAULT_TRAIN_NEGATIVE_PER_PATIENT = 6
DEFAULT_VALIDATION_SLICES_PER_PATIENT = 24
DEFAULT_SEED = 42
DEFAULT_OUTPUT_PATH = Path("data/processed/slice_index_subset_12train_4val.csv")


def read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV file returning row dictionaries and field order."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return rows, fieldnames


def read_patient_manifest(manifest_path: str | Path) -> dict[str, str]:
    """Read the patient manifest and return patient_id -> split."""
    rows, fieldnames = read_csv_rows(manifest_path)
    required = {"patient_id", "split"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Patient manifest is missing columns: {missing}")

    manifest: dict[str, str] = {}
    for row in rows:
        patient_id = row["patient_id"]
        split = row["split"]
        if patient_id in manifest and manifest[patient_id] != split:
            raise ValueError(f"Patient {patient_id} appears in multiple splits in the manifest.")
        manifest[patient_id] = split
    return manifest


def _sort_rows_by_slice(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(row["slice_index"]))


def _choose_patients(
    patient_ids: list[str],
    *,
    target_count: int,
    required_fn,
    seed: int,
) -> list[str]:
    """Choose a deterministic set of patient IDs that satisfy a requirement."""
    eligible = [patient_id for patient_id in patient_ids if required_fn(patient_id)]
    if len(eligible) < target_count:
        raise ValueError(
            f"Not enough eligible patients to select {target_count}. "
            f"Eligible count: {len(eligible)}."
        )
    rng = random.Random(seed)
    ordered = eligible[:]
    rng.shuffle(ordered)
    return ordered[:target_count]


def _interleave_rows_by_patient(patient_order: list[str], rows_by_patient: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Return patient-interleaved rows so batch order mixes patients within each split."""
    per_patient_lengths = {patient_id: len(rows_by_patient[patient_id]) for patient_id in patient_order}
    max_length = max(per_patient_lengths.values())
    interleaved: list[dict[str, str]] = []
    for position in range(max_length):
        for patient_id in patient_order:
            patient_rows = rows_by_patient[patient_id]
            if position < len(patient_rows):
                interleaved.append(patient_rows[position])
    return interleaved


def build_subset_rows(
    full_rows: list[dict[str, str]],
    patient_manifest: dict[str, str],
    *,
    seed: int = DEFAULT_SEED,
    train_patient_count: int = DEFAULT_TRAIN_PATIENT_COUNT,
    validation_patient_count: int = DEFAULT_VALIDATION_PATIENT_COUNT,
    train_positive_per_patient: int = DEFAULT_TRAIN_POSITIVE_PER_PATIENT,
    train_negative_per_patient: int = DEFAULT_TRAIN_NEGATIVE_PER_PATIENT,
    validation_slices_per_patient: int = DEFAULT_VALIDATION_SLICES_PER_PATIENT,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select a deterministic, metadata-only development subset.

    Returns the subset rows and a summary dictionary with selected patient IDs and
    the counts per split.
    """
    if train_patient_count <= 0 or validation_patient_count <= 0:
        raise ValueError("Training and validation patient counts must be positive.")
    if train_positive_per_patient <= 0 or train_negative_per_patient <= 0 or validation_slices_per_patient <= 0:
        raise ValueError("All per-patient slice counts must be positive.")

    rows_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in full_rows:
        patient_id = row["patient_id"]
        split = row["split"]
        if patient_id not in patient_manifest or patient_manifest[patient_id] != split:
            raise ValueError(f"Row for patient {patient_id} does not agree with the patient manifest split assignment.")
        rows_by_patient[patient_id].append(row)

    train_candidates = sorted(
        patient_id
        for patient_id, split in patient_manifest.items()
        if split == "train" and patient_id in rows_by_patient
    )
    validation_candidates = sorted(
        patient_id
        for patient_id, split in patient_manifest.items()
        if split == "validation" and patient_id in rows_by_patient
    )

    def train_eligible(patient_id: str) -> bool:
        patient_rows = rows_by_patient[patient_id]
        train_rows = [row for row in patient_rows if row["split"] == "train"]
        positive_rows = [row for row in train_rows if int(row.get("tumor_pixel_count", 0)) > 0]
        negative_rows = [row for row in train_rows if int(row.get("tumor_pixel_count", 0)) == 0]
        return len(positive_rows) >= train_positive_per_patient and len(negative_rows) >= train_negative_per_patient

    def validation_eligible(patient_id: str) -> bool:
        patient_rows = rows_by_patient[patient_id]
        validation_rows = [row for row in patient_rows if row["split"] == "validation"]
        return len(validation_rows) >= validation_slices_per_patient

    train_patient_ids = _choose_patients(
        train_candidates,
        target_count=train_patient_count,
        required_fn=train_eligible,
        seed=seed,
    )
    validation_patient_ids = _choose_patients(
        validation_candidates,
        target_count=validation_patient_count,
        required_fn=validation_eligible,
        seed=seed,
    )

    rng = random.Random(seed)

    train_rows_by_patient: dict[str, list[dict[str, str]]] = {}
    for patient_id in train_patient_ids:
        patient_rows = [row for row in rows_by_patient[patient_id] if row["split"] == "train"]
        pos_rows = [row for row in patient_rows if int(row.get("tumor_pixel_count", 0)) > 0]
        neg_rows = [row for row in patient_rows if int(row.get("tumor_pixel_count", 0)) == 0]
        if len(pos_rows) < train_positive_per_patient or len(neg_rows) < train_negative_per_patient:
            raise ValueError(
                f"Training patient {patient_id} does not have enough eligible slices. "
                f"Need {train_positive_per_patient} positive and {train_negative_per_patient} negative; "
                f"has {len(pos_rows)} and {len(neg_rows)}."
            )
        selected_pos = rng.sample(_sort_rows_by_slice(pos_rows), k=train_positive_per_patient)
        selected_neg = rng.sample(_sort_rows_by_slice(neg_rows), k=train_negative_per_patient)
        train_rows_by_patient[patient_id] = _sort_rows_by_slice(selected_pos + selected_neg)

    validation_rows_by_patient: dict[str, list[dict[str, str]]] = {}
    for patient_id in validation_patient_ids:
        patient_rows = [row for row in rows_by_patient[patient_id] if row["split"] == "validation"]
        if len(patient_rows) < validation_slices_per_patient:
            raise ValueError(
                f"Validation patient {patient_id} does not have enough eligible slices. "
                f"Need {validation_slices_per_patient}, have {len(patient_rows)}."
            )
        validation_rows_by_patient[patient_id] = rng.sample(_sort_rows_by_slice(patient_rows), k=validation_slices_per_patient)

    train_selected = _interleave_rows_by_patient(train_patient_ids, train_rows_by_patient)
    validation_selected = _interleave_rows_by_patient(validation_patient_ids, validation_rows_by_patient)

    subset_rows = train_selected + validation_selected

    # Validate the subset before returning it.
    seen_pairs: set[tuple[str, int]] = set()
    for row in subset_rows:
        patient_id = row["patient_id"]
        slice_index = int(row["slice_index"])
        if (patient_id, slice_index) in seen_pairs:
            raise ValueError(f"Duplicate patient/slice pair found in subset: {patient_id}, {slice_index}")
        seen_pairs.add((patient_id, slice_index))
        if patient_manifest.get(patient_id) != row["split"]:
            raise ValueError(f"Subset row split mismatch for patient {patient_id}: manifest says {patient_manifest.get(patient_id)}, row says {row['split']}")

    if set(train_patient_ids) & set(validation_patient_ids):
        raise ValueError("Train and validation patient selections overlap.")

    if len(train_selected) != len(train_patient_ids) * (train_positive_per_patient + train_negative_per_patient):
        raise ValueError(
            f"Unexpected training subset length: expected {len(train_patient_ids) * (train_positive_per_patient + train_negative_per_patient)}, got {len(train_selected)}."
        )

    if len(validation_selected) != len(validation_patient_ids) * validation_slices_per_patient:
        raise ValueError(
            f"Unexpected validation subset length: expected {len(validation_patient_ids) * validation_slices_per_patient}, got {len(validation_selected)}."
        )

    summary = {
        "train_patient_count": len(train_patient_ids),
        "validation_patient_count": len(validation_patient_ids),
        "train_patient_ids": train_patient_ids,
        "validation_patient_ids": validation_patient_ids,
        "train_rows": len(train_selected),
        "validation_rows": len(validation_selected),
        "total_rows": len(subset_rows),
        "train_positive_rows": sum(
            1 for row in train_selected if int(row.get("tumor_pixel_count", 0)) > 0
        ),
        "train_negative_rows": sum(
            1 for row in train_selected if int(row.get("tumor_pixel_count", 0)) == 0
        ),
    }
    return subset_rows, summary


def validate_subset_rows(
    full_rows: list[dict[str, str]],
    subset_rows: list[dict[str, str]],
    patient_manifest: dict[str, str],
) -> None:
    """Check that every subset row is valid and belongs to the original index."""
    all_rows_by_key = {
        (row["patient_id"], int(row["slice_index"])): row
        for row in full_rows
    }
    seen: set[tuple[str, int]] = set()
    for row in subset_rows:
        patient_id = row["patient_id"]
        slice_index = int(row["slice_index"])
        if (patient_id, slice_index) not in all_rows_by_key:
            raise ValueError(f"Subset row missing from original index: {patient_id}, slice {slice_index}")
        if patient_manifest.get(patient_id) != row["split"]:
            raise ValueError(f"Manifest split mismatch: patient {patient_id} expects {patient_manifest.get(patient_id)}, row says {row['split']}")
        if (patient_id, slice_index) in seen:
            raise ValueError(f"Duplicate subset record for patient {patient_id}, slice {slice_index}")
        seen.add((patient_id, slice_index))

    train_patients = {row["patient_id"] for row in subset_rows if row["split"] == "train"}
    val_patients = {row["patient_id"] for row in subset_rows if row["split"] == "validation"}
    if train_patients & val_patients:
        raise ValueError("Train and validation subset patients overlap.")
    if any(patient_id in {"test"} for patient_id in train_patients | val_patients):
        raise ValueError("The subset must not include any test-split patients.")


def create_subset_csv(
    full_index_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    train_patient_count: int = DEFAULT_TRAIN_PATIENT_COUNT,
    validation_patient_count: int = DEFAULT_VALIDATION_PATIENT_COUNT,
    train_positive_per_patient: int = DEFAULT_TRAIN_POSITIVE_PER_PATIENT,
    train_negative_per_patient: int = DEFAULT_TRAIN_NEGATIVE_PER_PATIENT,
    validation_slices_per_patient: int = DEFAULT_VALIDATION_SLICES_PER_PATIENT,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create the subset CSV and return the output path plus summary data."""
    full_index = Path(full_index_path)
    manifest = Path(manifest_path)
    destination = Path(output_path)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists: {destination}. Use overwrite=True to replace it explicitly.")

    full_rows, fieldnames = read_csv_rows(full_index)
    patient_assignments = read_patient_manifest(manifest)
    subset_rows, summary = build_subset_rows(
        full_rows,
        patient_assignments,
        seed=seed,
        train_patient_count=train_patient_count,
        validation_patient_count=validation_patient_count,
        train_positive_per_patient=train_positive_per_patient,
        train_negative_per_patient=train_negative_per_patient,
        validation_slices_per_patient=validation_slices_per_patient,
    )
    validate_subset_rows(full_rows, subset_rows, patient_assignments)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset_rows)

    return destination, summary


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a deterministic, metadata-only BraTS subset for development training experiments.")
    parser.add_argument("--full-index", type=Path, default=project_root / "data/processed/slice_index_full.csv", help="Path to the full sliced metadata CSV.")
    parser.add_argument("--patient-manifest", type=Path, default=project_root / "data/processed/patient_splits.csv", help="Path to the patient manifest CSV.")
    parser.add_argument("--output", type=Path, default=project_root / "data/processed/slice_index_subset_12train_4val.csv", help="Where to save the subset CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for deterministic patient and slice selection.")
    parser.add_argument("--train-patient-count", type=int, default=DEFAULT_TRAIN_PATIENT_COUNT, help="Number of training patients to include in the subset.")
    parser.add_argument("--validation-patient-count", type=int, default=DEFAULT_VALIDATION_PATIENT_COUNT, help="Number of validation patients to include in the subset.")
    parser.add_argument("--train-positive-per-patient", type=int, default=DEFAULT_TRAIN_POSITIVE_PER_PATIENT, help="Number of positive slices to select per training patient.")
    parser.add_argument("--train-negative-per-patient", type=int, default=DEFAULT_TRAIN_NEGATIVE_PER_PATIENT, help="Number of negative slices to select per training patient.")
    parser.add_argument("--validation-slices-per-patient", type=int, default=DEFAULT_VALIDATION_SLICES_PER_PATIENT, help="Number of validation slices to select per validation patient.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output CSV.")
    args = parser.parse_args(argv)

    if args.train_patient_count <= 0 or args.validation_patient_count <= 0:
        raise ValueError("Training and validation patient counts must be positive integers.")
    if args.train_positive_per_patient <= 0 or args.train_negative_per_patient <= 0 or args.validation_slices_per_patient <= 0:
        raise ValueError("All per-patient slice counts must be positive integers.")

    output_path, summary = create_subset_csv(
        args.full_index,
        args.patient_manifest,
        args.output,
        seed=args.seed,
        train_patient_count=args.train_patient_count,
        validation_patient_count=args.validation_patient_count,
        train_positive_per_patient=args.train_positive_per_patient,
        train_negative_per_patient=args.train_negative_per_patient,
        validation_slices_per_patient=args.validation_slices_per_patient,
        overwrite=args.overwrite,
    )

    print(f"Selected training patient count: {summary['train_patient_count']}")
    print(f"Selected validation patient count: {summary['validation_patient_count']}")
    print(f"Training rows: {summary['train_rows']}")
    print(f"Validation rows: {summary['validation_rows']}")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Train positive rows: {summary['train_positive_rows']}")
    print(f"Train negative rows: {summary['train_negative_rows']}")
    print(f"Selected training patient IDs: {summary['train_patient_ids']}")
    print(f"Selected validation patient IDs: {summary['validation_patient_ids']}")
    print(f"Subset saved to: {output_path}")


if __name__ == "__main__":
    main()
