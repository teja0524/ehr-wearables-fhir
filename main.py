"""
implementation of a multi-agent pipeline to transform EHR and wearables data
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EHR/Wearables → FHIR R4 multi-agent transformation pipeline"
    )
    parser.add_argument(
        "--dataset",
        default="mfu",
        help="Which dataset to process: 'mfu' or 'ace' (default: mfu)",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=None,
        help="Which clusters to process. Use 'all' for every cluster in the "
             "dataset. Defaults to a small demo set for mfu, all clusters for ace.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=3,
        help="Max subjects to process (0 = all, default: 3 for fast demo)",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Skip LLM code mapping step (uses cached mapping_index if available)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model id for code mapping (default: config.CLAUDE_MODEL). "
             "Non-Anthropic models route through LiteLLM and need that provider's API key.",
    )
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Run schema parse + code mapping only and write mapping_report.json; "
             "do NOT build FHIR bundles",
    )
    parser.add_argument(
        "--rebuild-store",
        action="store_true",
        help="Delete and rebuild the ChromaDB vector store",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    # --- build-time seed enrichment (offline; not part of a pipeline run) ---
    parser.add_argument(
        "--enrich-seed",
        action="store_true",
        help="BUILD-TIME: query UMLS for standard codes for rag-mappable variables "
             "(all datasets by default) and write a review CSV to "
             "data/terminology/seed_candidates/. Does NOT run the pipeline.",
    )
    parser.add_argument(
        "--apply-seed",
        action="store_true",
        help="BUILD-TIME: append accept=Y rows from the seed review CSV into the "
             "shared seed CSVs (deduped). Does NOT run the pipeline.",
    )
    parser.add_argument(
        "--seed-test",
        metavar="TERM",
        default=None,
        help="BUILD-TIME: print raw UMLS hits for one term (diagnostic).",
    )
    parser.add_argument(
        "--seed-only-unmapped",
        metavar="PATH",
        default="",
        help="With --enrich-seed: restrict to variables currently UNMAPPED, read "
             "from a mapping_cache.json / mapping_report.json.",
    )
    parser.add_argument(
        "--seed-datasets",
        default="",
        help="With --enrich-seed: comma list of datasets to source variable labels "
             "from (default: all). The seed itself is always shared.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    console.print(Panel.fit(
        "[bold cyan]EHR / Wearables → FHIR R4 Transformation MVP[/bold cyan]\n"
        "[dim]LangGraph  ·  Claude API  ·  ChromaDB  ·  FHIR R4[/dim]",
        border_style="cyan",
    ))

    import config as cfg

    # --- build-time seed enrichment: short-circuit before the pipeline ------
    # These modes are offline terminology management (UMLS), not a pipeline run,
    # and enrich the SHARED seed across all datasets — so no dataset is required
    # and the Claude key is not needed.
    if args.enrich_seed or args.apply_seed or args.seed_test:
        from tools.umls_seed_builder import run_build, run_apply, run_test
        if args.seed_test:
            run_test(args.seed_test)
        elif args.enrich_seed:
            names = [d.strip() for d in args.seed_datasets.split(",") if d.strip()] or None
            run_build(dataset_names=names, only_unmapped=args.seed_only_unmapped)
        elif args.apply_seed:
            run_apply()
        return 0

    if not cfg.ANTHROPIC_API_KEY:
        console.print("[red]ERROR:[/red] ANTHROPIC_API_KEY not set. "
                      "Create a .env file with ANTHROPIC_API_KEY=sk-ant-...")
        return 1

    # Resolve the active dataset.
    if args.dataset not in cfg.DATASETS:
        console.print(f"[red]ERROR:[/red] Unknown dataset '{args.dataset}'. "
                      f"Choose from: {', '.join(cfg.DATASETS)}")
        return 1
    dataset = cfg.DATASETS[args.dataset]

    # Default cluster selection per dataset; "all" expands to every cluster.
    if not args.clusters:
        args.clusters = (
            ["blood_labs", "blood_cytokines", "wearables"]
            if args.dataset == "mfu" else list(dataset["all_clusters"])
        )
    if len(args.clusters) == 1 and args.clusters[0].lower() == "all":
        args.clusters = list(dataset["all_clusters"])

    max_subjects = args.max_subjects if args.max_subjects > 0 else None
    console.print(
        f"[green]✓[/green] Config loaded | "
        f"Dataset: {args.dataset} | "
        f"Clusters: {args.clusters} | "
        f"Max subjects: {max_subjects or 'all'} | "
        f"Model: {args.model or cfg.CLAUDE_MODEL}"
    )

    from src.vector_store.store import TerminologyStore

    if args.rebuild_store and cfg.CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(cfg.CHROMA_DIR)
        log.info("Deleted existing ChromaDB at %s", cfg.CHROMA_DIR)

    console.print("\n[bold]Step 1/5:[/bold] Building terminology vector store (ChromaDB)...")
    t0 = time.time()

    store = TerminologyStore.from_config(cfg)
    store.build_or_load()

    console.print(f"[green]✓[/green] Vector store ready ({time.time()-t0:.1f}s)")

    from src.graph import build_graph, initial_state, export_output

    state = initial_state(
        clusters=args.clusters,
        max_subjects=max_subjects,
        dataset=dataset,
        skip_mapping=args.skip_mapping,
        model=args.model or cfg.CLAUDE_MODEL,
    )

    # Mapping-only mode: run just schema parse + code mapping, then write the
    # mapping report (no bundles are built). Fast way to inspect/iterate on
    # terminology coverage without the expensive FHIR build.
    if args.mapping_only:
        from src.agents.schema_parser import parse_schema
        from src.agents.code_mapper import map_codes

        console.print("\n[bold]Mapping-only mode:[/bold] schema parse + code mapping (no bundles)\n")
        t1 = time.time()
        state = parse_schema(state, cfg)
        state = map_codes(state, cfg, store)
        # export_output writes only mapping_report.json when there are no bundles.
        state = export_output(state, cfg)
        elapsed = time.time() - t1
        console.print(f"\n[green]✓[/green] Mapping completed in {elapsed:.1f}s")
        _print_summary(state, elapsed)
        subdir = dataset.get("output_subdir", "")
        report = (cfg.OUTPUT_DIR / subdir / "mapping_report.json") if subdir else (cfg.OUTPUT_DIR / "mapping_report.json")
        console.print(f"\n[bold]Mapping report:[/bold] [cyan]{report}[/cyan]")
        return 1 if state.get("errors") else 0

    console.print("\n[bold]Step 2/5:[/bold] Building LangGraph agent pipeline...")
    pipeline = build_graph(cfg, store)
    console.print("[green]✓[/green] Graph compiled (4 agent nodes + export)")

    console.print(f"\n[bold]Step 3-5/5:[/bold] Running pipeline on {args.clusters}...\n")
    t1 = time.time()

    final_state = pipeline.invoke(state)

    elapsed = time.time() - t1
    console.print(f"\n[green]✓[/green] Pipeline completed in {elapsed:.1f}s")

    _print_summary(final_state, elapsed)

    # Point the user at the clinical dashboard.
    viewer = cfg.BASE_DIR / "viewer" / "fhir_viewer.html"
    if viewer.exists():
        console.print(Panel.fit(
            f"[bold cyan]Clinical dashboard[/bold cyan]\n"
            f"Open this file in your browser:\n"
            f"[link=file://{viewer}]{viewer}[/link]\n"
            f"[dim]then click “Open bundle file(s)…” and select the JSON files in[/dim]\n"
            f"[dim]{cfg.OUTPUT_DIR}[/dim]",
            border_style="cyan",
        ))

    if final_state.get("errors"):
        for err in final_state["errors"]:
            console.print(f"[red]ERROR:[/red] {err}")
        return 1

    return 0


def _print_summary(state: dict, elapsed: float) -> None:
    """Print a rich table summary of the pipeline results."""
    console.print()

    # Code mapping table
    mappings = state.get("code_mappings", [])
    if mappings:
        table = Table(title="Code Mappings (sample)", show_header=True, header_style="bold magenta")
        table.add_column("Cluster", style="dim", width=18)
        table.add_column("Variable", width=22)
        table.add_column("Code", width=12)
        table.add_column("Display", width=38)
        table.add_column("Conf.", width=7)

        # Show first 20 mappings
        for m in mappings[:20]:
            conf_style = {
                "high": "green",
                "medium": "yellow",
                "low": "red",
            }.get(m.get("confidence", "low"), "white")
            display = str(m.get("display") or "")
            table.add_row(
                str(m.get("cluster", "")),
                str(m.get("variable", "")),
                str(m.get("code") or ""),
                display[:36] + ("…" if len(display) > 36 else ""),
                f"[{conf_style}]{m.get('confidence', 'low')}[/{conf_style}]",
            )

        if len(mappings) > 20:
            table.add_row("…", f"(+{len(mappings)-20} more)", "", "", "")

        console.print(table)

    # Output files
    output_paths = state.get("output_paths", [])
    if output_paths:
        console.print(f"\n[bold]Output files[/bold] ({len(output_paths)} total):")
        for p in output_paths[:6]:
            console.print(f"  [cyan]{p}[/cyan]")
        if len(output_paths) > 6:
            console.print(f"  … and {len(output_paths)-6} more bundle files")

    # Validation summary
    issues = state.get("validation_issues", [])
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    console.print(
        f"\n[bold]Validation:[/bold] "
        f"[red]{len(errors)} error(s)[/red]  "
        f"[yellow]{len(warnings)} warning(s)[/yellow]"
    )


if __name__ == "__main__":
    sys.exit(main())
