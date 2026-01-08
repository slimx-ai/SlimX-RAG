from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.settings import Settings



def build_arg_parser() -> argparse.ArgumentParser: 
    p = argparse.ArgumentParser(
        prog="slimx-ingest",
        description="Ingest documents into the SlimX knowledge base.",
    )
    p.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Path to the knowledge base directory. Defaults to <repo-root>/knowledge-base.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="**/*.md",
        help="Glob pattern to match documents. Defaults to '**/*.md'.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose logging. Use -v for INFO, -vv for DEBUG.",
    )

    return p


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING  # default
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    settings = Settings.default()
    documents = fetch_documents(
        settings=settings,
        kb_dir=args.kb_dir,
        glob=args.glob,
    )

    logging.info(f"Total documents ingested: {len(documents)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())