"""Create a reproducible patient-level CSV manifest without reading MRI data."""

import csv
from pathlib import Path
import random
import sys


def main():
    # Build paths from this script's location, regardless of the working directory.
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data/raw/brats2020"
    training_dir = (
        dataset_root
        / "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    )
    output_path = project_root / "data/processed/patient_splits.csv"
    modalities = ("flair", "t1", "t1ce", "t2", "seg")
    excluded_patient = "BraTS20_Training_355"

    if not training_dir.is_dir():
        raise ValueError(f"Training patient directory not found: {training_dir}")

    # Sorting before shuffling makes seed 42 independent of filesystem order.
    patient_dirs = sorted(
        path for path in training_dir.glob("BraTS20_Training_*") if path.is_dir()
    )
    print(f"Patient folders found: {len(patient_dirs)}")
    complete_patients = []
    exclusions = []
    errors = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        # Check filenames only: no NIfTI files are opened or loaded into RAM.
        files = {
            modality: patient_dir / f"{patient_id}_{modality}.nii"
            for modality in modalities
        }
        missing = [modality for modality, path in files.items() if not path.is_file()]
        if patient_id == excluded_patient and missing == ["seg"]:
            exclusions.append(patient_id)
            print(f"Excluded {patient_id}: missing {patient_id}_seg.nii")
            continue
        if missing:
            errors.append(
                f"{patient_id}: missing "
                + ", ".join(files[modality].name for modality in missing)
            )
            continue

        # Keep paths relative to the dataset link, without resolving its target.
        complete_patients.append({
            "patient_id": patient_id,
            **{
                f"{modality}_path": path.relative_to(dataset_root).as_posix()
                for modality, path in files.items()
            },
        })

    if errors:
        raise ValueError("Unexpected missing required files:\n" + "\n".join(errors))
    print(f"Complete patients: {len(complete_patients)}")
    print(f"Excluded patients: {len(exclusions)}")
    if len(complete_patients) != 368 or exclusions != [excluded_patient]:
        raise ValueError(
            "Expected 368 complete patients and only BraTS20_Training_355 "
            "excluded for a missing segmentation mask. No CSV was written."
        )

    # Shuffle whole patients, so their MRI modalities always stay together.
    random.Random(42).shuffle(complete_patients)
    splits = {
        "train": complete_patients[:258],
        "validation": complete_patients[258:313],
        "test": complete_patients[313:],
    }
    patient_ids = {
        name: {patient["patient_id"] for patient in patients}
        for name, patients in splits.items()
    }
    print("Random seed: 42")
    for name, patients in splits.items():
        print(f"{name} patients: {len(patients)}")

    # Check every pair of splits and verify every complete patient occurs once.
    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ):
        overlap = patient_ids[first] & patient_ids[second]
        print(f"Overlap {first}/{second}: {len(overlap)}")
        if overlap:
            raise ValueError(f"Patients occur in both {first} and {second}.")
    all_ids = set().union(*patient_ids.values())
    expected_ids = {patient["patient_id"] for patient in complete_patients}
    total_rows = sum(len(patients) for patients in splits.values())
    if all_ids != expected_ids or len(all_ids) != total_rows:
        raise ValueError("Each complete patient must appear exactly once.")
    print("Exactly one split per complete patient: PASS")

    # Existing manifests require explicit confirmation before being overwritten.
    # Exclusive creation ('x') also protects against a new file appearing later.
    mode = "x"
    if output_path.exists():
        try:
            answer = input(f"CSV already exists: {output_path}\nOverwrite it? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Existing CSV preserved. No changes made.")
            return
        mode = "w"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["patient_id", "split"] + [f"{item}_path" for item in modalities]
    with output_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for name, patients in splits.items():
            for patient in patients:
                writer.writerow({**patient, "split": name})
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")
