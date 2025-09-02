
"""Tokenization classes for HeliumBERT model."""

from transformers import AlbertTokenizer
import re

class HeliumbertTokenizer(AlbertTokenizer):
    """Represents a HeliumBERT tokenizer."""
    def _tokenize(self, text: str, **kwargs) -> list[str]:
        text = re.sub(r'[\u0600-\u06FF]+', '', text)

        return super()._tokenize(text, **kwargs)

__all__ = ["HeliumbertTokenizer"]
