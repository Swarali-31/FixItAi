"""Unit tests for information retrieval utilities and Reciprocal Rank Fusion (RRF)."""
import numpy as np
from src.fixitai import analyze_query, chunk_documents, reciprocal_rank_fusion


def test_analyze_query_extraction():
    q = "Python script raises NullPointerException and ECONNREFUSED on Docker"
    info = analyze_query(q)
    assert "Python" in info["technologies"]
    assert "Docker" in info["technologies"]
    assert "NullPointerException" in info["errors"]
    assert "ECONNREFUSED" in info["errors"]


def test_chunk_documents():
    docs = [{
        "title": "Test Doc",
        "text": "word " * 1200,
        "source": "unit_test.md",
        "technologies": ["Python", "FastAPI"]
    }]
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert chunks[0]["title"] == "Test Doc"
    assert "Python" in chunks[0]["technologies"]


def test_reciprocal_rank_fusion():
    # Modality 1 ranks doc 0 first, doc 1 second
    scores_m1 = np.array([0.9, 0.4, 0.0])
    # Modality 2 ranks doc 0 first, doc 2 second
    scores_m2 = np.array([0.8, 0.0, 0.5])

    rankings = {"M1": scores_m1, "M2": scores_m2}
    rrf = reciprocal_rank_fusion(rankings, total_items=3, k=60)

    # Doc 0 should clearly win as it ranked #1 in both modalities
    assert np.argmax(rrf) == 0
    assert rrf[0] > rrf[1]
    assert rrf[0] > rrf[2]
