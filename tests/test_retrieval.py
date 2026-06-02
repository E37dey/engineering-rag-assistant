"""Unit tests for the retrieval layer.

Kept small and focused — the eval suite covers end-to-end quality;
these tests cover contracts (shapes, metadata, top_k behavior).
"""
import pytest


def test_retrieve_returns_at_most_top_k():
    # TODO: seed a small collection, query with top_k=3, assert len <= 3
    pytest.skip("not implemented yet")


def test_retrieve_results_carry_metadata():
    # TODO: assert each result has 'filename' and 'page' fields
    pytest.skip("not implemented yet")


def test_retrieve_handles_empty_collection():
    # TODO: query against empty collection, assert returns [] not error
    pytest.skip("not implemented yet")
