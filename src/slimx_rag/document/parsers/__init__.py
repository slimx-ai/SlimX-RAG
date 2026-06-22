"""Native document parsers (no Unstructured/Docling/OCR/hosted services)."""

from __future__ import annotations

from .code import CodeParser
from .docx import DocxParser
from .markdown import MarkdownParser
from .pdf import PdfParser
from .text import TextParser

__all__ = ["PdfParser", "DocxParser", "MarkdownParser", "CodeParser", "TextParser"]
