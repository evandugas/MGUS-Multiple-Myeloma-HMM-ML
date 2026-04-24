#!/usr/bin/env python3
"""Sort raw sample files into mgus/mm/excluded subdirectories based on
GEO series matrix metadata.

GSE77975 has 4 platforms (GPL11358, GPL10152, GPL16237, GPL21461).
All are Agilent aCGH arrays at different resolutions. We use all of them:
  - 103 MGUS (all on GPL11358)
  - 33 MM (16 on GPL11358, 17 across other platforms)
  - 13 Normal controls -> excluded

Usage: python pipeline/01_sort_samples.py
Run from project root after 00_download.sh has completed.
"""

import os
import gzip
import shutil
import csv
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "GSE77975")


def parse_series_matrix(matrix_path):
    """Parse a GEO series matrix file. Returns dict of field -> [values]."""
    fields = {}
    multi_fields = {}
    with gzip.open(matrix_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_"):
                parts = line.strip().split("\t")
                key = parts[0]
                vals = [v.strip('"') for v in parts[1:]]
                if key in fields:
                    if key not in multi_fields:
                        multi_fields[key] = [fields[key]]
                    multi_fields[key].append(vals)
                fields[key] = vals
            elif line.startswith("!series_matrix_table_begin"):
                break
    for key, rows in multi_fields.items():
        fields[key + "_all"] = [" || ".join(row[i] for row in rows if i < len(row))
                                 for i in range(len(rows[0]))]
    return fields


def assign_labels_gpl11358(matrix_path, platform="GPL11358"):
    """GPL11358: 103 MGUS + 16 MM from title/characteristics.

    Returns dict gsm -> (label, platform).
    """
    fields = parse_series_matrix(matrix_path)
    gsm_ids = fields["!Sample_geo_accession"]
    titles = fields["!Sample_title"]
    all_chars = fields.get("!Sample_characteristics_ch1_all",
                           fields.get("!Sample_characteristics_ch1", [""] * len(gsm_ids)))

    labels = {}
    for gsm, title, chars in zip(gsm_ids, titles, all_chars):
        title_upper = title.upper()
        chars_upper = chars.upper()

        if "NORMAL" in title_upper or "PAIRED" in title_upper or \
           (title_upper.startswith("MGUS-") and "N_" in title.split("-")[1]):
            labels[gsm] = ("excluded", platform)
        elif "DISEASE STATE: MULTIPLE MYELOMA" in chars_upper or title_upper.startswith("MM-"):
            labels[gsm] = ("MM", platform)
        elif "DISEASE STATE: MONOCLONAL GAMMOPATHY" in chars_upper or title_upper.startswith("MGUS"):
            labels[gsm] = ("MGUS", platform)
        else:
            labels[gsm] = ("excluded", platform)

    return labels


def assign_labels_other_platforms(matrix_path, platform):
    """Other platforms (GPL10152, GPL16237, GPL21461): all MM samples."""
    fields = parse_series_matrix(matrix_path)
    gsm_ids = fields["!Sample_geo_accession"]
    return {gsm: ("MM", platform) for gsm in gsm_ids}


def sort_files(labels):
    """Move raw files into mgus/mm/excluded subdirectories based on labels."""
    raw_dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(raw_dir):
        print(f"  WARNING: {raw_dir} does not exist")
        return

    for subdir in ["mgus", "mm", "excluded"]:
        os.makedirs(os.path.join(raw_dir, subdir), exist_ok=True)

    # Move files back from subdirectories for re-sorting
    raw_files = []
    for subdir in ["mgus", "mm", "excluded", "normal", "mm_excluded"]:
        subdir_path = os.path.join(raw_dir, subdir)
        if os.path.exists(subdir_path):
            for f in os.listdir(subdir_path):
                if f.startswith("GSM"):
                    src = os.path.join(subdir_path, f)
                    dst = os.path.join(raw_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                    raw_files.append(f)

    # Also collect any at top level
    for f in os.listdir(raw_dir):
        if os.path.isfile(os.path.join(raw_dir, f)) and f.startswith("GSM"):
            raw_files.append(f)

    raw_files = list(set(raw_files))

    if not raw_files:
        print("  No sample files found")
        return

    moved = {"mgus": 0, "mm": 0, "excluded": 0}

    for filename in raw_files:
        gsm_id = filename.split("_")[0].split(".")[0]
        entry = labels.get(gsm_id, ("excluded", ""))
        label = entry[0] if isinstance(entry, tuple) else entry

        if label == "MGUS":
            dest = "mgus"
        elif label == "MM":
            dest = "mm"
        else:
            dest = "excluded"

        src = os.path.join(raw_dir, filename)
        dst = os.path.join(raw_dir, dest, filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved[dest] += 1

    print(f"  Sorted: {moved['mgus']} MGUS, {moved['mm']} MM, "
          f"{moved['excluded']} excluded")


if __name__ == "__main__":
    print("=" * 50)
    print(" Sorting GSE77975 samples (all 4 platforms)")
    print("=" * 50)

    # Collect labels from all platform matrices (gsm -> (label, platform))
    all_labels = {}

    # GPL11358: main platform (103 MGUS + 16 MM + 13 normal)
    matrix_11358 = os.path.join(DATA_DIR, "GSE77975-GPL11358_series_matrix.txt.gz")
    if os.path.exists(matrix_11358):
        labels = assign_labels_gpl11358(matrix_11358, "GPL11358")
        all_labels.update(labels)
        counts = Counter(v[0] for v in labels.values())
        print(f"  GPL11358: {dict(counts)}")
    else:
        print(f"  WARNING: {matrix_11358} not found")

    # Other platforms: all MM
    for gpl in ["GPL10152", "GPL16237", "GPL21461"]:
        matrix_path = os.path.join(DATA_DIR, f"GSE77975-{gpl}_series_matrix.txt.gz")
        if os.path.exists(matrix_path):
            labels = assign_labels_other_platforms(matrix_path, gpl)
            all_labels.update(labels)
            print(f"  {gpl}: {len(labels)} MM")
        else:
            print(f"  WARNING: {matrix_path} not found")

    # Sort files
    label_counts = Counter(v[0] for v in all_labels.values())
    print(f"\n  Total labels: {dict(label_counts)}")
    sort_files(all_labels)

    # Save labels CSV (now includes platform)
    csv_path = os.path.join(DATA_DIR, "sample_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "platform"])
        for gsm, (label, platform) in sorted(all_labels.items()):
            if label != "excluded":
                writer.writerow([gsm, label, platform])
    print(f"  Saved {csv_path}")

    # Summary
    n_mgus = sum(1 for v in all_labels.values() if v[0] == "MGUS")
    n_mm = sum(1 for v in all_labels.values() if v[0] == "MM")
    n_exc = sum(1 for v in all_labels.values() if v[0] == "excluded")
    print(f"\n  TOTAL: {n_mgus} MGUS + {n_mm} MM = {n_mgus + n_mm} usable")
    print(f"  ({n_exc} excluded normals)")
