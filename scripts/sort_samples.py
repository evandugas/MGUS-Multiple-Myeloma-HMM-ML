#!/usr/bin/env python3
###############################################################################
# Sort raw sample files into mgus/mm/excluded subdirectories based on
# GEO series matrix metadata.
#
# Usage: python scripts/sort_samples.py
# Run from project root after download_geo_data.sh has completed.
###############################################################################

import os
import gzip
import shutil
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def parse_series_matrix(matrix_path):
    """Parse a GEO series matrix file. Returns dict of field -> [values].
    For fields with multiple rows (like characteristics), stores as list of lists."""
    fields = {}
    multi_fields = {}  # key -> list of list of values (one per row)
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
                fields[key] = vals  # always store last row
            elif line.startswith("!series_matrix_table_begin"):
                break
    # For multi-row fields, concatenate all rows
    for key, rows in multi_fields.items():
        fields[key + "_all"] = [" || ".join(row[i] for row in rows if i < len(row))
                                 for i in range(len(rows[0]))]
    return fields


def assign_labels_gse77975(matrix_path):
    """GSE77975-GPL11358: labels from disease state in characteristics.
    The 13 paired normals are labeled MGUS in metadata but have 'Normal'
    or 'paired' in their title — these must be excluded."""
    fields = parse_series_matrix(matrix_path)
    gsm_ids = fields["!Sample_geo_accession"]
    titles = fields["!Sample_title"]
    # All characteristics concatenated
    all_chars = fields.get("!Sample_characteristics_ch1_all",
                           fields.get("!Sample_characteristics_ch1", [""] * len(gsm_ids)))

    labels = {}
    for gsm, title, chars in zip(gsm_ids, titles, all_chars):
        title_upper = title.upper()
        chars_upper = chars.upper()

        # Normals: paired normal samples have "N" suffix in patient ID (e.g. MGUS-100N)
        if "NORMAL" in title_upper or "PAIRED" in title_upper or \
           (title_upper.startswith("MGUS-") and "N_" in title.split("-")[1]):
            labels[gsm] = "excluded"
        elif "DISEASE STATE: MULTIPLE MYELOMA" in chars_upper or title_upper.startswith("MM-"):
            labels[gsm] = "MM"
        elif "DISEASE STATE: MONOCLONAL GAMMOPATHY" in chars_upper or title_upper.startswith("MGUS"):
            labels[gsm] = "MGUS"
        else:
            labels[gsm] = "excluded"

    return labels


def assign_labels_gse77975_other_platforms(matrix_path):
    """GSE77975 non-GPL11358 platforms: all MM, exclude from main (different probes)."""
    fields = parse_series_matrix(matrix_path)
    gsm_ids = fields["!Sample_geo_accession"]
    return {gsm: "mm_excluded" for gsm in gsm_ids}


def assign_labels_gse33685(matrix_path):
    """GSE33685: labels from title (MM clinical sample / MGUS clinical sample)."""
    fields = parse_series_matrix(matrix_path)
    gsm_ids = fields["!Sample_geo_accession"]
    titles = fields["!Sample_title"]

    labels = {}
    for gsm, title in zip(gsm_ids, titles):
        if title.upper().startswith("MM"):
            labels[gsm] = "MM"
        elif title.upper().startswith("MGUS"):
            labels[gsm] = "MGUS"
        else:
            labels[gsm] = "excluded"
    return labels



def sort_dataset(dataset_dir, labels, dataset_name):
    """Move raw files into mgus/mm/excluded subdirectories based on labels."""
    raw_dir = os.path.join(dataset_dir, "raw")
    if not os.path.exists(raw_dir):
        print(f"  WARNING: {raw_dir} does not exist, skipping")
        return

    # Create subdirectories
    for subdir in ["mgus", "mm", "excluded"]:
        os.makedirs(os.path.join(raw_dir, subdir), exist_ok=True)

    # mm_excluded goes into excluded


    # Collect all raw files — both at top level and in subdirectories
    raw_files = [f for f in os.listdir(raw_dir)
                 if os.path.isfile(os.path.join(raw_dir, f)) and f.startswith("GSM")]

    # If files are already in subdirectories, move them back to raw/ first (re-sort)
    for subdir in ["mgus", "mm", "excluded", "normal"]:
        subdir_path = os.path.join(raw_dir, subdir)
        if os.path.exists(subdir_path):
            for f in os.listdir(subdir_path):
                if f.startswith("GSM"):
                    src = os.path.join(subdir_path, f)
                    dst = os.path.join(raw_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        raw_files.append(f)

    if not raw_files:
        print(f"  No sample files found in {raw_dir}")
        return

    moved = {"mgus": 0, "mm": 0, "excluded": 0}

    for filename in raw_files:
        gsm_id = filename.split("_")[0].split(".")[0]

        label = labels.get(gsm_id, "excluded")
        if label == "MGUS":
            dest = "mgus"
        elif label == "MM":
            dest = "mm"
        else:
            dest = "excluded"

        src = os.path.join(raw_dir, filename)
        dst = os.path.join(raw_dir, dest, filename)
        shutil.move(src, dst)
        moved[dest] += 1

    print(f"  Sorted: {moved['mgus']} MGUS, {moved['mm']} MM, "
          f"{moved['excluded']} excluded")

    # Save labels as CSV for reference
    csv_path = os.path.join(dataset_dir, "sample_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "dataset"])
        for gsm, label in sorted(labels.items()):
            writer.writerow([gsm, label, dataset_name])
    print(f"  Saved {csv_path}")


if __name__ == "__main__":
    print("=" * 50)
    print(" Sorting samples by disease label")
    print("=" * 50)

    # --- GSE77975 (primary) ---
    print("\n--- GSE77975 (primary, Agilent GPL11358) ---")
    gse77975_dir = os.path.join(DATA_DIR, "GSE77975")

    # Main platform (GPL11358)
    matrix_11358 = os.path.join(gse77975_dir, "GSE77975-GPL11358_series_matrix.txt.gz")
    if os.path.exists(matrix_11358):
        labels_77975 = assign_labels_gse77975(matrix_11358)

        # Add other platform samples as mm_excluded
        for gpl in ["GPL16237", "GPL10152", "GPL21461"]:
            matrix_other = os.path.join(gse77975_dir, f"GSE77975-{gpl}_series_matrix.txt.gz")
            if os.path.exists(matrix_other):
                labels_77975.update(assign_labels_gse77975_other_platforms(matrix_other))

        # Count
        from collections import Counter
        counts = Counter(labels_77975.values())
        print(f"  Labels: {dict(counts)}")
        sort_dataset(gse77975_dir, labels_77975, "GSE77975")
    else:
        print(f"  Series matrix not found: {matrix_11358}")

    # --- GSE33685 ---
    print("\n--- GSE33685 (Agilent GPL10152) ---")
    gse33685_dir = os.path.join(DATA_DIR, "GSE33685")
    matrix = os.path.join(gse33685_dir, "GSE33685_series_matrix.txt.gz")
    if os.path.exists(matrix):
        labels_33685 = assign_labels_gse33685(matrix)
        counts = Counter(labels_33685.values())
        print(f"  Labels: {dict(counts)}")
        sort_dataset(gse33685_dir, labels_33685, "GSE33685")
    else:
        print(f"  Series matrix not found")

    # --- Summary ---
    print("\n" + "=" * 50)
    print(" Summary")
    print("=" * 50)
    total_mgus = total_mm = total_excluded = 0
    for ds_name, ds_labels in [
        ("GSE77975", labels_77975 if 'labels_77975' in dir() else {}),
        ("GSE33685", labels_33685 if 'labels_33685' in dir() else {}),
    ]:
        mgus = sum(1 for v in ds_labels.values() if v == "MGUS")
        mm = sum(1 for v in ds_labels.values() if v == "MM")
        exc = sum(1 for v in ds_labels.values() if v in ("excluded", "mm_excluded"))
        total_mgus += mgus
        total_mm += mm
        total_excluded += exc
        print(f"  {ds_name}: {mgus} MGUS, {mm} MM, {exc} excluded")

    print(f"\n  TOTAL: {total_mgus} MGUS + {total_mm} MM = {total_mgus + total_mm} usable")
    print(f"  ({total_excluded} excluded: normals, wrong platform)")
