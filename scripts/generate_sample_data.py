#!/usr/bin/env python3
"""Generate sample .kohort files for testing Atlas.

Usage:
    uv run python scripts/generate_sample_data.py --rows 10000 --output data/
    uv run python scripts/generate_sample_data.py --rows 100000 --output data/ --name large_test
"""

import argparse
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Sample data generators
STATES = ["TX", "CA", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
GENDERS = ["M", "F"]
PAYERS = ["Medicare", "Medicaid", "Commercial", "Self-Pay"]
DIAGNOSES = [
    "E11.9",  # Type 2 diabetes
    "I10",    # Essential hypertension
    "J06.9",  # Acute upper respiratory infection
    "M54.5",  # Low back pain
    "F32.9",  # Major depressive disorder
    "K21.0",  # GERD
    "J45.909", # Asthma
    "E78.5",  # Hyperlipidemia
    "Z00.00", # General adult medical exam
    "R10.9",  # Abdominal pain
]
PROCEDURES = [
    "99213",  # Office visit, established patient
    "99214",  # Office visit, moderate complexity
    "36415",  # Blood draw
    "81001",  # Urinalysis
    "93000",  # ECG
    "71046",  # Chest X-ray
    "80053",  # Comprehensive metabolic panel
    "85025",  # Complete blood count
    "90471",  # Immunization administration
    "99395",  # Preventive visit
]


def random_patient_id():
    """Generate a random patient ID."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


def random_date(start_year=2020, end_year=2024):
    """Generate a random date."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def random_yob():
    """Generate a random year of birth."""
    return random.randint(1940, 2020)


def generate_patients(n_rows: int) -> pa.Table:
    """Generate patient demographic data."""
    data = {
        "PATIENT_ID": [random_patient_id() for _ in range(n_rows)],
        "PATIENT_GENDER": [random.choice(GENDERS) for _ in range(n_rows)],
        "PATIENT_STATE": [random.choice(STATES) for _ in range(n_rows)],
        "PATIENT_YOB": [random_yob() for _ in range(n_rows)],
        "PRIMARY_PAYER": [random.choice(PAYERS) for _ in range(n_rows)],
        "ENROLLMENT_START": [random_date(2018, 2022) for _ in range(n_rows)],
        "ENROLLMENT_END": [random_date(2023, 2024) for _ in range(n_rows)],
    }
    return pa.table(data)


def generate_claims(patient_ids: list, avg_claims_per_patient: int = 5) -> pa.Table:
    """Generate medical claims data."""
    claims = []
    claim_id = 1

    for patient_id in patient_ids:
        n_claims = random.randint(1, avg_claims_per_patient * 2)
        for _ in range(n_claims):
            claims.append({
                "CLAIM_ID": f"CLM{claim_id:010d}",
                "PATIENT_ID": patient_id,
                "SERVICE_DATE": random_date(),
                "DIAGNOSIS_CODE": random.choice(DIAGNOSES),
                "PROCEDURE_CODE": random.choice(PROCEDURES),
                "CHARGE_AMOUNT": round(random.uniform(50, 5000), 2),
                "PAID_AMOUNT": round(random.uniform(25, 2500), 2),
                "PROVIDER_STATE": random.choice(STATES),
            })
            claim_id += 1

    return pa.table({
        "CLAIM_ID": [c["CLAIM_ID"] for c in claims],
        "PATIENT_ID": [c["PATIENT_ID"] for c in claims],
        "SERVICE_DATE": [c["SERVICE_DATE"] for c in claims],
        "DIAGNOSIS_CODE": [c["DIAGNOSIS_CODE"] for c in claims],
        "PROCEDURE_CODE": [c["PROCEDURE_CODE"] for c in claims],
        "CHARGE_AMOUNT": [c["CHARGE_AMOUNT"] for c in claims],
        "PAID_AMOUNT": [c["PAID_AMOUNT"] for c in claims],
        "PROVIDER_STATE": [c["PROVIDER_STATE"] for c in claims],
    })


def write_kohort_file(
    output_path: Path,
    patients: pa.Table,
    claims: pa.Table | None = None,
    name: str = "sample",
):
    """Write a .kohort file with embedded Parquet data.

    .kohort file format:
    - 8 bytes: magic "KOHORT01"
    - 4 bytes: version (1)
    - 4 bytes: number of tags
    - For each tag:
      - 4 bytes: tag ID
      - 8 bytes: offset
      - 8 bytes: length
      - variable: label (null-terminated string)
    - Tag data sections
    """
    import struct
    import io

    # Prepare Parquet buffers
    patient_buf = io.BytesIO()
    pq.write_table(patients, patient_buf, compression="snappy")
    patient_bytes = patient_buf.getvalue()

    claims_bytes = None
    if claims is not None:
        claims_buf = io.BytesIO()
        pq.write_table(claims, claims_buf, compression="snappy")
        claims_bytes = claims_buf.getvalue()

    # Build tag directory
    tags = []
    current_offset = 0  # Will be updated after header

    # Tag 100: Patients
    tags.append({
        "id": 100,
        "label": f"{name}_patients",
        "data": patient_bytes,
    })

    # Tag 101: Claims (if provided)
    if claims_bytes:
        tags.append({
            "id": 101,
            "label": f"{name}_claims",
            "data": claims_bytes,
        })

    # Calculate header size
    header_size = 8 + 4 + 4  # magic + version + tag_count
    for tag in tags:
        header_size += 4 + 8 + 8 + len(tag["label"]) + 1  # id + offset + length + label + null

    # Assign offsets
    current_offset = header_size
    for tag in tags:
        tag["offset"] = current_offset
        tag["length"] = len(tag["data"])
        current_offset += tag["length"]

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(b"KOHORT01")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(tags)))  # tag count

        # Tag directory
        for tag in tags:
            f.write(struct.pack("<I", tag["id"]))
            f.write(struct.pack("<Q", tag["offset"]))
            f.write(struct.pack("<Q", tag["length"]))
            f.write(tag["label"].encode("utf-8") + b"\x00")

        # Tag data
        for tag in tags:
            f.write(tag["data"])

    print(f"Written: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - {len(patients)} patients")
    if claims is not None:
        print(f"  - {len(claims)} claims")


def main():
    parser = argparse.ArgumentParser(description="Generate sample .kohort files")
    parser.add_argument("--rows", type=int, default=10000, help="Number of patient rows")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--name", type=str, default="sample", help="Dataset name prefix")
    parser.add_argument("--with-claims", action="store_true", help="Include claims data")
    args = parser.parse_args()

    print(f"Generating {args.rows} patients...")
    patients = generate_patients(args.rows)

    claims = None
    if args.with_claims:
        print("Generating claims...")
        patient_ids = patients.column("PATIENT_ID").to_pylist()
        claims = generate_claims(patient_ids)

    output_path = Path(args.output) / f"{args.name}.kohort"
    write_kohort_file(output_path, patients, claims, args.name)

    print("\nDone! Run Atlas with:")
    print(f"  uv run python -m cohort_viz.server {args.output}")


if __name__ == "__main__":
    main()
