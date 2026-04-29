"""Platform-specific parsers for copy number array data.

All parsers return a list of tuples with the same schema:
    (sample_id, label, chrom, start, end, log_ratio, log_ratio_err)

Supported platforms:
    - Agilent Feature Extraction .txt.gz (aCGH)
"""

import gzip
import numpy as np

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


def parse_agilent_file(filepath, sample_id, label):
    """Parse an Agilent Feature Extraction .txt.gz file.

    Extracts LogRatio, LogRatioError, and genomic coordinates from
    non-control probes on autosomes (chr1-22).
    """
    rows = []
    try:
        with gzip.open(filepath, "rt", errors="replace") as f:
            in_features = False
            header = None
            for line in f:
                if line.startswith("FEATURES"):
                    parts = line.strip().split("\t")
                    header = {col: i for i, col in enumerate(parts)}
                    in_features = True
                    continue
                if not in_features or header is None:
                    continue

                parts = line.strip().split("\t")
                if len(parts) < len(header):
                    continue

                try:
                    control_type = int(parts[header["ControlType"]])
                except (ValueError, KeyError):
                    continue

                if control_type != 0:
                    continue

                try:
                    log_ratio = float(parts[header["LogRatio"]])
                    log_ratio_err = float(parts[header["LogRatioError"]])
                    systematic_name = parts[header["SystematicName"]]
                except (ValueError, KeyError):
                    continue

                if ":" not in systematic_name or "-" not in systematic_name:
                    continue
                chrom, pos_range = systematic_name.split(":", 1)
                pos_parts = pos_range.split("-")
                if len(pos_parts) != 2:
                    continue
                try:
                    start = int(pos_parts[0])
                    end = int(pos_parts[1])
                except ValueError:
                    continue

                if chrom not in CHROMOSOMES:
                    continue

                rows.append((sample_id, label, chrom, start, end,
                             log_ratio, log_ratio_err))
    except Exception as e:
        print(f"    ERROR parsing {filepath}: {e}")
        return None

    return rows
