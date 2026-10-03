"""Read and validate explicit study partitions and image-level fold labels."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED = {"image_id", "image_path", "mask_path", "participant_id", "recording_id", "split", "fold"}


def read_manifest(path: str | Path, *, group_folds_by_participant: bool = True) -> list[dict[str, str | Path]]:
    if not group_folds_by_participant:
        raise ValueError("Participant grouping cannot be disabled in this release")
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must have columns: {', '.join(sorted(REQUIRED))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")

    seen_images: set[Path] = set()
    seen_ids: set[str] = set()
    participants: dict[str, str] = {}
    participant_folds: dict[str, str] = {}
    recordings: dict[str, str] = {}
    for line, row in enumerate(rows, start=2):
        for column in REQUIRED:
            if row[column] is None or not row[column].strip():
                if column == "fold" and row["split"] == "test":
                    continue
                raise ValueError(f"Line {line}: {column} is missing")
        if row["split"] not in {"development", "test"}:
            raise ValueError(f"Line {line}: split must be development or test")
        if row["split"] == "development" and row["fold"] not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"Line {line}: development fold must be 1..5")
        participant = row["participant_id"]
        if participant in participants and participants[participant] != row["split"]:
            raise ValueError(f"Participant {participant} appears in development and test")
        participants[participant] = row["split"]
        if group_folds_by_participant and row["split"] == "development":
            if participant in participant_folds and participant_folds[participant] != row["fold"]:
                raise ValueError(f"Participant {participant} appears in multiple development folds")
            participant_folds[participant] = row["fold"]
        recording = row["recording_id"]
        if recording in recordings and recordings[recording] != participant:
            raise ValueError(f"Recording {recording} belongs to multiple participants")
        recordings[recording] = participant
        for column in ("image_path", "mask_path"):
            file = (path.parent / row[column]).resolve()
            if not file.is_file():
                raise ValueError(f"Line {line}: missing {column}: {file}")
            row[column] = file
        if row["image_path"] in seen_images:
            raise ValueError(f"Duplicate image on line {line}")
        seen_images.add(row["image_path"])
        if row["image_id"] in seen_ids:
            raise ValueError(f"Duplicate image_id on line {line}")
        seen_ids.add(row["image_id"])
    return rows


def fold_partition(rows: list[dict[str, str | Path]], fold: int):
    if fold not in range(1, 6):
        raise ValueError("fold must be 1..5")
    development = [row for row in rows if row["split"] == "development"]
    train = [row for row in development if row["fold"] != str(fold)]
    val = [row for row in development if row["fold"] == str(fold)]
    test = [row for row in rows if row["split"] == "test"]
    training_participants = {row["participant_id"] for row in train}
    validation_participants = {row["participant_id"] for row in val}
    testing_participants = {row["participant_id"] for row in test}
    if training_participants & validation_participants:
        raise ValueError("A participant appears in training and validation")
    if (training_participants | validation_participants) & testing_participants:
        raise ValueError("A participant appears in development and test")
    if not train or not val:
        raise ValueError(f"Fold {fold} has empty train or validation partition")
    return train, val, test
