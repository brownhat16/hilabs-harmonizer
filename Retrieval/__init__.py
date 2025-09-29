"""Retrieval package exposing terminology matching utilities."""

from .matcher import RetrievalEngine, MatchResult, Candidate

__all__ = [
    "RetrievalEngine",
    "MatchResult",
    "Candidate",
]
