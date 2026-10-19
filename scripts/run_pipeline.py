"""Full pipeline runner: Layer 1 → Layer 2 → Layer 3 → Layer 4 (scripts/run_pipeline.py).

Processes every DOCX/PDF/TXT file found in a samples directory (default: all
of samples/ and samples/portfolio/), saving intermediate outputs to data/.

Usage:
    # Run all sample docs (original 5 + portfolio 12)
    python scripts/run_pipeline.py

    # Run only the original samples
    python scripts/run_pipeline.py --samples-dir samples

    # Run on a specific file
    python scripts/run_pipeline.py --file samples/msa_acme.docx

    # Skip LLM fill (rules-only, faster, fully offline)
    python scripts/run_pipeline.py --no-llm

    # Change as_of date for graph building
    python scripts/run_pipeline.py --as-of 2027-01-01

Design decisions:
  - Idempotent: re-running overwrites existing outputs.
  - Each layer's failure is reported per-file but does not abort the whole run.
  - LLM is skipped by default when --no-llm is passed (NullLLM used).
  - Graph is rebuilt from all docs in data/ regardless of which files were processed.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.extractor import extract_document
from app.graph.build import build_graph
from app.ingestion.pipeline import ingest
from app.llm.client import LLMClient, NullLLM
from app.resolution.pipeline import resolve

DATA_DIR = Path("data")
SAMPLES_DIRS = [Path("samples"), Path("samples/portfolio")]
SUPPORTED_EXTS = {".docx", ".pdf", ".txt"}


def _collect_files(args: argparse.Namespace) -> list[Path]:
    """Collect input files based on CLI arguments."""
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)
        return [p]

    dirs = [Path(d) for d in args.samples_dir] if args.samples_dir else SAMPLES_DIRS
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTS and f.is_file():
                    files.append(f)
        else:
            print(f"[WARN] Samples directory not found, skipping: {d}")
    return files


def run_pipeline(
    files: list[Path],
    data_dir: Path,
    llm: LLMClient,
    as_of: str,
    verbose: bool = False,
) -> dict[str, dict[str, bool]]:
    """Process each file through Layers 1–3; build Layer 4 graph at the end.

    Returns:
        dict mapping filename -> {layer_name -> success bool}
    """
    results: dict[str, dict[str, bool]] = {}

    for f in files:
        fname = f.name
        status: dict[str, bool] = {}
        print(f"\n[{fname}]")

        # ----------------------------------------------------------
        # Layer 1: Ingestion
        # ----------------------------------------------------------
        try:
            parsed = ingest(path=f, data_dir=data_dir, save=True)
            print(f"  L1 ingestion  OK   doc_id={parsed.doc_id}  "
                  f"clauses={len(parsed.clauses)}  type={parsed.doc_type}")
            status["L1_ingestion"] = True
        except Exception as exc:
            print(f"  L1 ingestion  FAIL {exc}")
            if verbose:
                traceback.print_exc()
            status["L1_ingestion"] = False
            results[fname] = status
            continue

        # ----------------------------------------------------------
        # Layer 2: Extraction
        # ----------------------------------------------------------
        try:
            extracted = extract_document(parsed, llm, use_llm_fill=not isinstance(llm, NullLLM))
            # Persist extracted document
            ext_dir = data_dir / "extracted"
            ext_dir.mkdir(parents=True, exist_ok=True)
            (ext_dir / f"{parsed.doc_id}.v{parsed.version}.json").write_text(
                extracted.model_dump_json(indent=2), encoding="utf-8"
            )
            typed = [c for c in extracted.clauses if c.clause_type.value != "other"]
            print(f"  L2 extraction OK   clauses={len(extracted.clauses)}  "
                  f"typed={len(typed)}")
            status["L2_extraction"] = True
        except Exception as exc:
            print(f"  L2 extraction FAIL {exc}")
            if verbose:
                traceback.print_exc()
            status["L2_extraction"] = False
            results[fname] = status
            continue

        # ----------------------------------------------------------
        # Layer 3: Resolution
        # ----------------------------------------------------------
        try:
            resolved = resolve(
                parsed=parsed,
                extracted=extracted,
                data_dir=data_dir,
                llm=llm,
                save=True,
            )
            matches = [m for m in resolved.template_matches if m.template_id]
            print(f"  L3 resolution OK   parties={len(resolved.parties)}  "
                  f"template_matches={len(matches)}")
            status["L3_resolution"] = True
        except Exception as exc:
            print(f"  L3 resolution FAIL {exc}")
            if verbose:
                traceback.print_exc()
            status["L3_resolution"] = False

        results[fname] = status

    # ----------------------------------------------------------
    # Layer 4: Knowledge Graph (built from all data/ docs)
    # ----------------------------------------------------------
    print(f"\n[LAYER 4] Building knowledge graph (as_of={as_of}) ...")
    try:
        graph = build_graph(data_dir=data_dir, as_of=as_of)
        nc = graph.node_counts()
        ec = graph.edge_counts()
        print(f"  L4 graph      OK   nodes={len(graph.nodes)}  edges={len(graph.edges)}  "
              f"review_items={len(graph.review_items)}")
        print(f"  Node types: {nc}")
        print(f"  Edge types: {ec}")

        errors = [r for r in graph.review_items if r.severity == "error"]
        warnings = [r for r in graph.review_items if r.severity == "warning"]
        infos = [r for r in graph.review_items if r.severity == "info"]
        print(f"\n  REVIEW ITEMS ({len(graph.review_items)})  "
              f"[errors={len(errors)}, warnings={len(warnings)}, info={len(infos)}]")
        for group_label, group in [("ERROR", errors), ("WARNING", warnings), ("INFO", infos)]:
            if not group:
                continue
            print(f"    [{group_label}]")
            for r in group:
                span_str = f" [span: {r.span_id}]" if r.span_id else ""
                print(f"      ! [{r.kind}] ({r.doc_id}){span_str}: {r.description}")
        if not graph.review_items:
            print("    (no review items generated)")
    except Exception as exc:
        print(f"  L4 graph      FAIL {exc}")
        if verbose:
            traceback.print_exc()

    return results


def _print_summary(results: dict[str, dict[str, bool]]) -> None:
    """Print a tabular summary of layer pass/fail per file."""
    print("\n" + "=" * 65)
    print("PIPELINE SUMMARY")
    print("=" * 65)
    layers = ["L1_ingestion", "L2_extraction", "L3_resolution"]
    header = f"  {'File':<35}" + "".join(f"  {l:<15}" for l in layers)
    print(header)
    print("  " + "-" * 63)
    ok_count = 0
    for fname, status in sorted(results.items()):
        row = f"  {fname:<35}"
        all_ok = True
        for l in layers:
            val = status.get(l)
            if val is True:
                row += f"  {'OK':<15}"
            elif val is False:
                row += f"  {'FAIL':<15}"
                all_ok = False
            else:
                row += f"  {'-':<15}"
        print(row)
        if all_ok:
            ok_count += 1
    print(f"\n  {ok_count}/{len(results)} files fully processed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Layer 1–4 pipeline on contract documents"
    )
    parser.add_argument(
        "--file", help="Process a single file instead of a directory"
    )
    parser.add_argument(
        "--samples-dir",
        action="append",
        help="One or more directories to scan for contract files "
             "(default: samples/ and samples/portfolio/)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root data directory for outputs (default: data)"
    )
    parser.add_argument(
        "--as-of", default="2026-10-19",
        help="Evaluation date for graph effective state (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM classification; use rules only (faster, fully offline)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks on errors"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    llm: LLMClient = NullLLM()
    if not args.no_llm:
        try:
            from app.llm.client import OllamaClient  # type: ignore
            llm = OllamaClient()
            print("[INFO] Using Ollama LLM for Layer 2 classification.")
        except Exception:
            print("[INFO] Ollama not available; using rules-only (NullLLM).")
            llm = NullLLM()

    files = _collect_files(args)
    if not files:
        print("[WARN] No input files found. Check --samples-dir or --file.")
        sys.exit(0)

    print(f"\nCONTRACT GRAPH PIPELINE RUNNER")
    print(f"  Input files : {len(files)}")
    print(f"  Data dir    : {data_dir.resolve()}")
    print(f"  As-of date  : {args.as_of}")
    print(f"  LLM         : {'NullLLM (rules only)' if isinstance(llm, NullLLM) else 'Ollama'}")
    print(f"{'='*65}")

    results = run_pipeline(
        files=files,
        data_dir=data_dir,
        llm=llm,
        as_of=args.as_of,
        verbose=args.verbose,
    )

    _print_summary(results)


if __name__ == "__main__":
    main()
