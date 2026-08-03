"""Dataset loaders. Every loader returns a list[Record] (see schema.py)."""
from .schema import Record, load_dataset

__all__ = ["Record", "load_dataset"]
