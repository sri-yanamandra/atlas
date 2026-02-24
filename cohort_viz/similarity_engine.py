"""Patient Similarity Engine using BioLORD + NumPy.

Simple, fast implementation using brute-force cosine similarity.
For production: implement in Rust with ANN index.

Usage:
    from cohort_viz.similarity_engine import SimilarityEngine

    engine = SimilarityEngine()
    engine.build_index("m.cohort")

    # Find patients similar to #424968838 in Florida
    results = engine.search(
        query_patient_id=424968838,
        filters={"PATIENT_STATE": "FL"},
        limit=10
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cohort_sdk import read as cohort_read


# Lazy-load the model to avoid slow import
_model = None


def _get_model():
    """Load BioLORD-2023 model (cached)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading BioLORD-2023 model...")
        _model = SentenceTransformer("FremyCompany/BioLORD-2023-C")
        print(f"Model loaded: {_model.get_sentence_embedding_dimension()} dimensions")
    return _model


def _parse_diagnosis_codes(codes_str: str | None) -> list[str]:
    """Parse pipe-delimited diagnosis codes."""
    if not codes_str:
        return []
    codes = [c.strip() for c in codes_str.split("|") if c.strip()]
    return codes


def _build_clinical_text(row: dict) -> str:
    """Build a clinical description from a patient row for embedding."""
    parts = []

    diag_codes = _parse_diagnosis_codes(row.get("DIAGNOSIS_CODES") or row.get("diagnosis_codes"))
    if diag_codes:
        parts.append(f"Diagnoses: {', '.join(diag_codes)}")

    proc = row.get("PROCEDURE_CODE") or row.get("procedure_code")
    if proc:
        parts.append(f"Procedure: {proc}")

    visit = row.get("VISIT_TYPE") or row.get("visit_type")
    if visit:
        parts.append(f"Visit: {visit}")

    admit_dx = row.get("ADMIT_DIAGNOSIS_CODE") or row.get("admit_diagnosis_code")
    if admit_dx:
        parts.append(f"Admission reason: {admit_dx}")

    ndc = row.get("NDC11") or row.get("ndc11")
    if ndc:
        parts.append(f"Drug: {ndc}")

    return ". ".join(parts) if parts else "Unknown clinical profile"


def embed_patients(rows: list[dict], batch_size: int = 32) -> np.ndarray:
    """Embed patient rows using BioLORD-2023."""
    model = _get_model()
    texts = [_build_clinical_text(row) for row in rows]
    print(f"Embedding {len(texts)} patients...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )
    return embeddings


class SimilarityEngine:
    """Hybrid search engine using numpy brute-force (fast for <100K patients)."""

    def __init__(self, cache_dir: str = ".vector_cache"):
        self.cache_dir = Path(cache_dir)
        self.embeddings: np.ndarray | None = None
        self.rows: list[dict] = []
        self.patient_id_to_idx: dict[int, int] = {}

    def build_index(self, cohort_path: str, rebuild: bool = False) -> None:
        """Build vector index from cohort file."""
        cache_path = self.cache_dir / f"{Path(cohort_path).stem}_vectors.npz"

        # Try to load from cache
        if cache_path.exists() and not rebuild:
            print(f"Loading cached vectors: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.rows = data["rows"].tolist()
            self._build_patient_index()
            print(f"Loaded {len(self.rows)} patients from cache")
            return

        # Load cohort data
        print(f"Loading cohort: {cohort_path}")
        cohort = cohort_read(cohort_path)
        dataset = cohort.datasets[0]
        table = cohort[dataset]

        # Convert to list of dicts
        self.rows = table.to_pylist()

        # Generate embeddings
        self.embeddings = embed_patients(self.rows)
        self._build_patient_index()

        # Cache for next time
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=self.embeddings, rows=np.array(self.rows, dtype=object))
        print(f"Cached {len(self.rows)} patient vectors to {cache_path}")

    def _build_patient_index(self):
        """Build patient ID to index lookup."""
        self.patient_id_to_idx = {}
        for i, row in enumerate(self.rows):
            pid = row.get("PATIENT_NUMBER") or row.get("patient_number")
            if pid:
                self.patient_id_to_idx[pid] = i

    def get_patient_embedding(self, patient_id: int) -> np.ndarray | None:
        """Get embedding vector for a patient."""
        idx = self.patient_id_to_idx.get(patient_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def search(
        self,
        query_patient_id: int | None = None,
        query_text: str | None = None,
        query_vector: np.ndarray | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        exclude_query_patient: bool = True,
    ) -> list[dict]:
        """Find similar patients.

        Args:
            query_patient_id: Find patients similar to this patient
            query_text: Or search by clinical description
            query_vector: Or use a pre-computed vector
            filters: Dict of column -> value for filtering
            limit: Number of results
            exclude_query_patient: Exclude query patient from results

        Returns:
            List of dicts with patient data + similarity score
        """
        if self.embeddings is None:
            raise RuntimeError("No index built. Call build_index() first.")

        t0 = time.perf_counter()

        # Get query vector
        if query_vector is not None:
            vec = query_vector
        elif query_patient_id is not None:
            vec = self.get_patient_embedding(query_patient_id)
            if vec is None:
                raise ValueError(f"Patient {query_patient_id} not found")
        elif query_text is not None:
            model = _get_model()
            vec = model.encode([query_text], normalize_embeddings=True)[0]
        else:
            raise ValueError("Must provide query_patient_id, query_text, or query_vector")

        # Cosine similarity (embeddings are pre-normalized)
        similarities = self.embeddings @ vec

        # Apply filters to create mask
        mask = np.ones(len(self.rows), dtype=bool)
        if filters:
            for col, val in filters.items():
                for i, row in enumerate(self.rows):
                    row_val = row.get(col) or row.get(col.upper()) or row.get(col.lower())
                    if row_val != val:
                        mask[i] = False

        # Exclude query patient
        if exclude_query_patient and query_patient_id is not None:
            idx = self.patient_id_to_idx.get(query_patient_id)
            if idx is not None:
                mask[idx] = False

        # Apply mask
        masked_similarities = np.where(mask, similarities, -np.inf)

        # Get top-k indices
        top_indices = np.argsort(masked_similarities)[::-1][:limit]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build results
        results = []
        for idx in top_indices:
            if masked_similarities[idx] == -np.inf:
                continue
            result = dict(self.rows[idx])
            result["_similarity"] = float(similarities[idx])
            results.append(result)

        print(f"Search completed in {elapsed_ms:.2f}ms ({len(results)} results)")
        return results

    def search_by_text(self, clinical_description: str, filters: dict | None = None, limit: int = 10) -> list[dict]:
        """Search by clinical description."""
        return self.search(query_text=clinical_description, filters=filters, limit=limit)


def demo():
    """Demo the similarity engine."""
    engine = SimilarityEngine()
    engine.build_index("m.cohort")

    print("\n" + "=" * 70)
    print("Finding patients similar to #424968838 (FL hip replacement)")
    print("=" * 70)

    results = engine.search(query_patient_id=424968838, limit=5)

    print(f"\nTop 5 similar patients:")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i}. Patient #{r['PATIENT_NUMBER']} ({r['PATIENT_STATE']}) — Similarity: {r['_similarity']:.3f}")
        print(f"   Diagnoses: {r['DIAGNOSIS_CODES'][:60]}...")
        print(f"   Visit: {r['VISIT_TYPE']}")
        print()

    print("=" * 70)
    print("Same search, filtered to Texas only")
    print("=" * 70)

    results_tx = engine.search(
        query_patient_id=424968838,
        filters={"PATIENT_STATE": "TX"},
        limit=5
    )

    print(f"\nTop 5 similar patients in Texas:")
    print("-" * 70)
    for i, r in enumerate(results_tx, 1):
        print(f"{i}. Patient #{r['PATIENT_NUMBER']} ({r['PATIENT_STATE']}) — Similarity: {r['_similarity']:.3f}")
        print(f"   Diagnoses: {r['DIAGNOSIS_CODES'][:60]}...")
        print()

    print("=" * 70)
    print("Semantic search: 'diabetes with hip complications'")
    print("=" * 70)

    results_semantic = engine.search_by_text(
        "diabetes mellitus with hip joint complications and fracture",
        limit=5
    )

    print(f"\nTop 5 patients matching semantic query:")
    print("-" * 70)
    for i, r in enumerate(results_semantic, 1):
        print(f"{i}. Patient #{r['PATIENT_NUMBER']} ({r['PATIENT_STATE']}) — Similarity: {r['_similarity']:.3f}")
        print(f"   Diagnoses: {r['DIAGNOSIS_CODES'][:60]}...")
        print()


if __name__ == "__main__":
    demo()
