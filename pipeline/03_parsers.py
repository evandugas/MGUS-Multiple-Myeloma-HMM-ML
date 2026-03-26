"""Platform-specific parsers for copy number array data.

All parsers return a list of tuples with the same schema:
    (sample_id, label, chrom, start, end, log_ratio, log_ratio_err)

Supported platforms:
    - Agilent Feature Extraction .txt.gz (aCGH)
    - Affymetrix SNP 6.0 processed data (birdseed/log2 ratio text files)
"""

import gzip
import os
import struct
import numpy as np

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]

# Affymetrix SNP 6.0 chromosome mapping (integer -> chr name)
_AFFY_CHROM_MAP = {i: f"chr{i}" for i in range(1, 23)}
_AFFY_CHROM_MAP[23] = "chrX"
_AFFY_CHROM_MAP[24] = "chrY"


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


def parse_affymetrix_processed(filepath, sample_id, label):
    """Parse a processed Affymetrix log2 ratio file.

    Expects a tab-separated file with columns including:
        - Chromosome (or Chr): chromosome number (1-22, X, Y)
        - Position (or Start): genomic position
        - Log2Ratio (or Log.R.Ratio, CN.Log2Ratio, etc.): copy number log2 ratio

    Handles multiple common column naming conventions from GEO processed files.
    """
    rows = []
    chrom_col = None
    pos_col = None
    lr_col = None

    try:
        opener = gzip.open if filepath.endswith(".gz") else open
        with opener(filepath, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue

                parts = line.strip().split("\t")
                if chrom_col is None:
                    # Detect header columns
                    header = {col.lower().replace(" ", "_"): i for i, col in enumerate(parts)}
                    # Chromosome column
                    for name in ["chromosome", "chr", "chrom"]:
                        if name in header:
                            chrom_col = header[name]
                            break
                    # Position column
                    for name in ["position", "start", "physical_position",
                                 "physical.position", "pos"]:
                        if name in header:
                            pos_col = header[name]
                            break
                    # Log2 ratio column
                    for name in ["log2ratio", "log.r.ratio", "log_r_ratio",
                                 "cn.log2ratio", "log2_ratio", "logratio",
                                 "log_ratio", "value"]:
                        if name in header:
                            lr_col = header[name]
                            break

                    if chrom_col is None or pos_col is None or lr_col is None:
                        print(f"    WARNING: Could not find required columns in {filepath}")
                        print(f"    Found headers: {list(header.keys())}")
                        return None
                    continue

                if len(parts) <= max(chrom_col, pos_col, lr_col):
                    continue

                try:
                    chrom_raw = parts[chrom_col].strip()
                    # Normalize chromosome name
                    if not chrom_raw.startswith("chr"):
                        chrom_raw = "chr" + chrom_raw
                    if chrom_raw not in CHROMOSOMES:
                        continue

                    position = int(parts[pos_col])
                    log_ratio = float(parts[lr_col])
                except (ValueError, IndexError):
                    continue

                if np.isnan(log_ratio) or np.isinf(log_ratio):
                    continue

                rows.append((sample_id, label, chrom_raw, position,
                             position + 1, log_ratio, 0.0))
    except Exception as e:
        print(f"    ERROR parsing {filepath}: {e}")
        return None

    return rows


def parse_cnchp_file(filepath, sample_id, label):
    """Parse Affymetrix CNCHP file (Calvin/HDF5 format) for log2 ratios.

    CNCHP files from Affymetrix ChAS contain copy number results.
    These are Calvin-format binary files (Command Console generic data).
    Uses h5py if available, falls back to attempting text extraction.
    """
    try:
        import h5py
    except ImportError:
        print(f"    WARNING: h5py not installed, cannot parse CNCHP: {filepath}")
        return None

    rows = []
    try:
        opener = gzip.open if filepath.endswith(".gz") else open
        # CNCHP files may be gzipped
        if filepath.endswith(".gz"):
            import tempfile
            with gzip.open(filepath, "rb") as gz_in:
                with tempfile.NamedTemporaryFile(suffix=".CNCHP", delete=False) as tmp:
                    tmp.write(gz_in.read())
                    tmp_path = tmp.name
            h5_path = tmp_path
        else:
            h5_path = filepath
            tmp_path = None

        try:
            with h5py.File(h5_path, "r") as h5:
                # Navigate Calvin file structure
                # CopyNumber results are typically in /CopyNumber/CopyNumberResults
                # or /AlgorithmData/CN5/CNNeutralLOH
                cn_data = None
                for group_path in ["/CopyNumber/CopyNumberResults",
                                   "/CopyNumber",
                                   "/AlgorithmData"]:
                    if group_path in h5:
                        cn_data = h5[group_path]
                        break

                if cn_data is None:
                    # Try to find any dataset with chromosome/position/log2ratio
                    print(f"    WARNING: Unknown CNCHP structure in {filepath}")
                    print(f"    Available groups: {list(h5.keys())}")
                    return None

                # Extract arrays
                chroms = cn_data.get("Chromosome", cn_data.get("chromosome"))
                positions = cn_data.get("Position", cn_data.get("position",
                                        cn_data.get("PhysicalPosition")))
                log2ratios = cn_data.get("Log2Ratio", cn_data.get("log2ratio",
                                          cn_data.get("SmoothSignal")))

                if chroms is None or positions is None or log2ratios is None:
                    print(f"    WARNING: Missing data arrays in {filepath}")
                    return None

                chroms = np.array(chroms)
                positions = np.array(positions)
                log2ratios = np.array(log2ratios)

                for i in range(len(chroms)):
                    chrom = _AFFY_CHROM_MAP.get(int(chroms[i]))
                    if chrom is None or chrom not in CHROMOSOMES:
                        continue
                    lr = float(log2ratios[i])
                    if np.isnan(lr) or np.isinf(lr):
                        continue
                    pos = int(positions[i])
                    rows.append((sample_id, label, chrom, pos, pos + 1, lr, 0.0))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"    ERROR parsing CNCHP {filepath}: {e}")
        return None

    return rows


def detect_and_parse_affymetrix(filepath, sample_id, label):
    """Auto-detect Affymetrix file format and parse accordingly.

    Handles: .CNCHP.gz, processed .txt/.gz, .CEL.gz (skipped with warning)
    """
    fname = os.path.basename(filepath).upper()

    if "CNCHP" in fname:
        return parse_cnchp_file(filepath, sample_id, label)
    elif fname.endswith(".CEL.GZ") or fname.endswith(".CEL"):
        # CEL files need R preprocessing — skip with warning
        print(f"    SKIP: {filepath} is a raw CEL file (needs R preprocessing)")
        return None
    else:
        # Try as processed text file
        return parse_affymetrix_processed(filepath, sample_id, label)
