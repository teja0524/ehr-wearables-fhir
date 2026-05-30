"""
ChromaDB-backed terminology store.
Embeds LOINC and SNOMED seed CSVs using sentence-transformers for semantic search.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Settings

logger = logging.getLogger(__name__)


def _sentence_transformer_ef(model_name: str):
    """Return a ChromaDB-compatible embedding function backed by sentence-transformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


class TerminologyStore:
    """Manages two ChromaDB collections: one for LOINC, one for SNOMED."""

    def __init__(
        self,
        chroma_dir: Path,
        loinc_csv: Path,
        snomed_csv: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        loinc_collection: str = "loinc_terms",
        snomed_collection: str = "snomed_terms",
    ):
        self.chroma_dir = chroma_dir
        self.loinc_csv = loinc_csv
        self.snomed_csv = snomed_csv
        self.embedding_model = embedding_model
        self.loinc_collection_name = loinc_collection
        self.snomed_collection_name = snomed_collection

        self._ef = _sentence_transformer_ef(embedding_model)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._loinc: chromadb.Collection | None = None
        self._snomed: chromadb.Collection | None = None

    @classmethod
    def from_config(cls, cfg) -> "TerminologyStore":
        """Construct from config module."""
        return cls(
            chroma_dir=cfg.CHROMA_DIR,
            loinc_csv=cfg.TERMINOLOGY_DIR / "loinc_seed.csv",
            snomed_csv=cfg.TERMINOLOGY_DIR / "snomed_seed.csv",
            embedding_model=cfg.EMBEDDING_MODEL,
            loinc_collection=cfg.CHROMA_COLLECTION_LOINC,
            snomed_collection=cfg.CHROMA_COLLECTION_SNOMED,
        )

    def build_or_load(self) -> None:
        """Build collections from CSV seeds if empty, otherwise just open them."""
        self._loinc = self._get_or_create(self.loinc_collection_name)
        self._snomed = self._get_or_create(self.snomed_collection_name)

        if self._loinc.count() == 0:
            logger.info("LOINC collection empty — seeding from %s", self.loinc_csv)
            self._seed_loinc()
        else:
            logger.info("LOINC collection already has %d terms", self._loinc.count())

        if self._snomed.count() == 0:
            logger.info("SNOMED collection empty — seeding from %s", self.snomed_csv)
            self._seed_snomed()
        else:
            logger.info("SNOMED collection already has %d terms", self._snomed.count())

    def search_loinc(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search over LOINC terms. Returns candidates with distance (lower = closer)."""
        assert self._loinc is not None, "Call build_or_load() first"
        results = self._loinc.query(
            query_texts=[query],
            n_results=min(top_k, self._loinc.count()),
            include=["metadatas", "distances", "documents"],
        )
        return self._format_results(results, "loinc_code")

    def search_snomed(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search over SNOMED terms. Returns candidates with distance (lower = closer)."""
        assert self._snomed is not None, "Call build_or_load() first"
        results = self._snomed.query(
            query_texts=[query],
            n_results=min(top_k, self._snomed.count()),
            include=["metadatas", "distances", "documents"],
        )
        return self._format_results(results, "snomed_code")

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _seed_loinc(self) -> None:
        ids, docs, metas = [], [], []
        with open(self.loinc_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["loinc_code"]
                # The document that gets embedded — rich text for good recall
                doc = (
                    f"{row['long_common_name']} | "
                    f"component: {row['component']} | "
                    f"units: {row['units']} | "
                    f"system: {row['system']} | "
                    f"{row['search_text']}"
                )
                ids.append(code)
                docs.append(doc)
                metas.append({
                    "loinc_code": code,
                    "long_common_name": row["long_common_name"],
                    "component": row["component"],
                    "units": row["units"],
                    "category": row["category"],
                    "system": row["system"],
                })
        self._loinc.add(ids=ids, documents=docs, metadatas=metas)
        logger.info("Seeded %d LOINC terms", len(ids))

    def _seed_snomed(self) -> None:
        ids, docs, metas = [], [], []
        with open(self.snomed_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["snomed_code"]
                doc = (
                    f"{row['preferred_term']} | "
                    f"synonyms: {row['synonyms']} | "
                    f"hierarchy: {row['hierarchy']} | "
                    f"{row['search_text']}"
                )
                ids.append(code)
                docs.append(doc)
                metas.append({
                    "snomed_code": code,
                    "preferred_term": row["preferred_term"],
                    "hierarchy": row["hierarchy"],
                })
        self._snomed.add(ids=ids, documents=docs, metadatas=metas)
        logger.info("Seeded %d SNOMED terms", len(ids))

    @staticmethod
    def _format_results(
        chroma_result: dict, code_key: str
    ) -> list[dict[str, Any]]:
        out = []
        metadatas = chroma_result["metadatas"][0]
        distances = chroma_result["distances"][0]
        documents = chroma_result["documents"][0]
        for meta, dist, doc in zip(metadatas, distances, documents):
            entry = dict(meta)
            entry["distance"] = round(dist, 4)
            entry["document"] = doc
            out.append(entry)
        return out


# UMLS production loader (to be completed)

def load_umls_mrconso(
    mrconso_path: Path,
    store: TerminologyStore,
    target_sabs: list[str] | None = None,
) -> None:
    """
    Loads the full UMLS MRCONSO.RRF into the ChromaDB store.

    This replaces the seed CSV approach with the complete UMLS Metathesaurus,
    which consolidates LOINC, SNOMED CT, ICD-10-CM, RxNorm, and 150+ other
    vocabulary sources.

    Parameters
    ----------
    mrconso_path : Path
        Path to MRCONSO.RRF (download from https://www.nlm.nih.gov/research/umls/)
    store : TerminologyStore
        Initialised (but not yet seeded) store instance
    target_sabs : list[str], optional
        Vocabulary source abbreviations to import.
        Defaults to ["LNC", "SNOMEDCT_US", "ICD10CM", "RXNORM"].

    Notes
    -----
    MRCONSO.RRF has ~15M rows. Batch the .add() calls in chunks of ~1000 to
    avoid memory issues. This function is a stub — implement it when you have
    a UMLS license and access to the release files.
    """
    if target_sabs is None:
        target_sabs = ["LNC", "SNOMEDCT_US", "ICD10CM", "RXNORM"]

    raise NotImplementedError(
        "Full UMLS loader not yet implemented. "
        "Obtain UMLS release from https://www.nlm.nih.gov/research/umls/ "
        "then implement the MRCONSO.RRF parsing here."
    )
