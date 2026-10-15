"""Index brain-containing axial slices without preprocessing or saving MRI data."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


SPLITS = ("train", "validation", "test")
COLUMNS = (
    "patient_id", "split", "flair_path", "seg_path", "slice_index",
    "brain_pixel_count", "tumor_pixel_count", "has_tumor",
)


def read_manifest(path):
    """Inspect the CSV and validate each patient's unique split assignment."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        print(f"Manifest columns: {columns}")
        required = {"patient_id", "split", "flair_path", "seg_path"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        patients = list(reader)

    counts = Counter(patient["split"] for patient in patients)
    print(f"Manifest split labels and patient counts: {dict(counts)}")
    # These are the split labels verified in the existing project manifest.
    if set(counts) != set(SPLITS):
        raise ValueError(f"Expected split labels {SPLITS}; found {tuple(counts)}.")
    assignments = {}
    for patient in patients:
        if any(not patient.get(column) for column in required):
            raise ValueError(f"Manifest row has an empty required value: {patient}")
        patient_id = patient["patient_id"]
        if patient_id in assignments:
            raise ValueError(f"Patient {patient_id} appears more than once in the manifest.")
        assignments[patient_id] = patient["split"]
    return patients, assignments


def dataset_path(dataset_root, relative_path):
    """Resolve a CSV path against the dataset link, leaving that link untouched."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a dataset-relative path: {relative_path!r}")
    full_path = dataset_root / path
    if not full_path.is_file():
        raise ValueError(f"Required NIfTI file not found: {full_path}")
    return full_path


def index_patient(patient, dataset_root):
    """Read one slice at a time and return only small metadata records."""
    patient_id = patient["patient_id"]
    flair_path = dataset_path(dataset_root, patient["flair_path"])
    seg_path = dataset_path(dataset_root, patient["seg_path"])

    # NiBabel's proxies support read-only memory mapping for uncompressed NIfTI
    # files. Slice the proxies directly instead of loading full float64 volumes.
    flair = nib.load(flair_path, mmap="r", keep_file_open=False)
    segmentation = nib.load(seg_path, mmap="r", keep_file_open=False)
    if flair.shape != segmentation.shape:
        raise ValueError(
            f"{patient_id}: volume shapes do not match: "
            f"FLAIR {flair.shape}, segmentation {segmentation.shape}."
        )
    if len(flair.shape) != 3 or any(size == 0 for size in flair.shape):
        raise ValueError(f"{patient_id}: expected nonempty three-dimensional volumes.")
    if not np.allclose(flair.affine, segmentation.affine, rtol=0, atol=1e-5):
        raise ValueError(f"{patient_id}: FLAIR and segmentation spatial coordinates differ.")

    records = []
    labels = set()
    for slice_index in range(flair.shape[2]):
        # Axis 2 is the axial direction for these BraTS volumes. Always use the
        # same index for the MRI and mask so their pixel counts stay aligned.
        image = np.asanyarray(flair.dataobj[:, :, slice_index])
        mask = np.asanyarray(segmentation.dataobj[:, :, slice_index])
        if not np.isfinite(image).all():
            raise ValueError(f"{patient_id}, slice {slice_index}: nonfinite FLAIR values.")
        slice_labels = np.unique(mask)
        # Check every mask slice, including slices later excluded for no brain.
        if not np.isin(slice_labels, [0, 1, 2, 4]).all():
            raise ValueError(
                f"{patient_id}, slice {slice_index}: unexpected segmentation labels "
                f"{slice_labels.tolist()}; expected only 0, 1, 2, 4."
            )
        labels.update(slice_labels.tolist())
        brain_count = int(np.count_nonzero(image))
        tumor_count = int(np.count_nonzero(mask))
        if brain_count == 0:
            continue

        # Keep BOTH tumor-positive and tumor-negative brain-containing slices.
        # Nonzero mask labels 1, 2, and 4 all count toward the whole tumor.
        records.append({
            "patient_id": patient_id,
            "split": patient["split"],
            "flair_path": patient["flair_path"],
            "seg_path": patient["seg_path"],
            "slice_index": slice_index,
            "brain_pixel_count": brain_count,
            "tumor_pixel_count": tumor_count,
            "has_tumor": int(tumor_count > 0),
        })
    print(
        f"{patient['split']}: {patient_id}, shape {flair.shape}, "
        f"labels {sorted(labels)}, {len(records)} slice records, "
        f"{flair.shape[2] - len(records)} empty-FLAIR slices skipped"
    )
    return records


def validate_records(records, assignments):
    """Check uniqueness, split assignments, and flags against mask pixel counts."""
    seen = set()
    for record in records:
        key = (record["patient_id"], record["slice_index"])
        if key in seen:
            raise ValueError(f"Duplicate patient/slice record: {key}")
        seen.add(key)
        if assignments.get(record["patient_id"]) != record["split"]:
            raise ValueError(f"Incorrect split assignment for {record['patient_id']}.")
        if record["brain_pixel_count"] <= 0 or record["tumor_pixel_count"] < 0:
            raise ValueError(f"Invalid brain/tumor pixel counts for {key}.")
        if record["has_tumor"] not in (0, 1) or record["has_tumor"] != int(record["tumor_pixel_count"] > 0):
            raise ValueError(f"Tumor flag is inconsistent with segmentation pixel count for {key}.")
    print("Unique (patient_id, slice_index) check: PASS")
    print("Patient-to-split assignment check: PASS")
    print("Tumor flags/segmentation pixel counts consistency check: PASS")


def select_patients(patients, patients_per_split=1, all_patients=False):
    """Keep the existing split order and CSV order within each split."""
    selected = []
    for split in SPLITS:
        split_patients = [patient for patient in patients if patient["split"] == split]
        if not all_patients:
            split_patients = split_patients[:patients_per_split]
        selected.extend(split_patients)
    return selected


def validate_patient_coverage(expected_patients, completed_ids, records):
    """Require every expected patient to be processed once and appear in the CSV."""
    expected_ids = {patient["patient_id"] for patient in expected_patients}
    completed = Counter(completed_ids)
    if set(completed) != expected_ids or any(count != 1 for count in completed.values()):
        raise ValueError("Patient coverage failed: each expected patient must be processed exactly once.")
    indexed_ids = {record["patient_id"] for record in records}
    missing = expected_ids - indexed_ids
    if missing:
        raise ValueError(f"Patients have no brain-containing slice records: {sorted(missing)}")
    if indexed_ids != expected_ids:
        raise ValueError("Slice records contain patients outside the expected selection.")
    print(f"Patient coverage check: PASS ({len(expected_ids)} patients processed once and included)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--patients-per-split", type=int, default=1,
        help="Select up to this many patients per split, in CSV order (default: 1).",
    )
    selection.add_argument(
        "--all-patients", action="store_true",
        help="Process every patient in the verified manifest, without a per-split limit.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output CSV, relative to the project root. Defaults to data/processed/"
        "slice_index_full.csv with --all-patients, otherwise slice_index_smoke.csv.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly approve replacing an existing output CSV (default: refuse).",
    )
    args = parser.parse_args(argv)
    if args.patients_per_split < 1:
        parser.error("--patients-per-split must be a positive integer.")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data/raw/brats2020"
    manifest_path = project_root / "data/processed/patient_splits.csv"
    default_name = "slice_index_full.csv" if args.all_patients else "slice_index_smoke.csv"
    output_path = args.output if args.output is not None else Path("data/processed") / default_name
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if output_path.suffix.lower() != ".csv":
        raise ValueError("--output must name a .csv file.")
    if output_path.is_symlink() or output_path.resolve() == manifest_path.resolve():
        raise ValueError("Output must be a separate CSV path, not the patient manifest or a symlink.")
    smoke_path = project_root / "data/processed/slice_index_smoke.csv"
    if args.all_patients and output_path.resolve() == smoke_path.resolve():
        raise ValueError("Full indexing output must differ from the existing smoke-test CSV.")
    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"Output already exists: {output_path}\nNothing was overwritten. "
            "Choose a new --output path, or explicitly approve replacement with --overwrite."
        )

    patients, assignments = read_manifest(manifest_path)
    selected = select_patients(patients, args.patients_per_split, args.all_patients)
    if args.all_patients:
        print(f"Indexing all {len(patients)} patients in the verified manifest.")
    else:
        print(f"Selecting up to {args.patients_per_split} patients per split in CSV order.")
    records = []
    processed_counts = Counter()
    completed_ids = []
    for patient in selected:
        # Finish this patient before opening the next. Only small CSV metadata
        # records are retained across patients, never MRI or mask volume arrays.
        records.extend(index_patient(patient, dataset_root))
        processed_counts[patient["split"]] += 1
        completed_ids.append(patient["patient_id"])
        print(
            f"Progress: {len(completed_ids)}/{len(selected)} patients completed "
            f"({patient['patient_id']}, {patient['split']})",
            flush=True,
        )

    validate_records(records, assignments)
    # All-patient coverage is checked against the entire manifest, independently
    # of the selection helper. For this project's manifest that means 368 patients.
    expected_patients = patients if args.all_patients else selected
    validate_patient_coverage(expected_patients, completed_ids, records)
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        # These counts come from actual segmentation pixel counts, which were
        # also checked against has_tumor above. Both slice types must sum to total.
        positive = sum(record["tumor_pixel_count"] > 0 for record in split_records)
        negative = sum(record["tumor_pixel_count"] == 0 for record in split_records)
        if positive + negative != len(split_records):
            raise ValueError(f"{split}: tumor-positive/negative counts do not sum to the slice total.")
        print(
            f"{split} summary: patients processed={processed_counts[split]}, "
            f"slice records={len(split_records)}, tumor-positive={positive}, "
            f"tumor-negative={negative}"
        )

    # Save only after all selected patients and metadata checks have succeeded.
    # Exclusive creation ('x') also protects against an output appearing later.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} slice records to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, nib.filebasedimages.ImageFileError) as error:
        sys.exit(f"Error: {error}")
