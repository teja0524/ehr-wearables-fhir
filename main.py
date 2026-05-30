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
        "--clusters",
        nargs="+",
        default=["blood_labs", "blood_cytokines", "wearables"],
        help="Which clusters to process (default: blood_labs blood_cytokines wearables)",
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
        "--rebuild-store",
        action="store_true",
        help="Delete and rebuild the ChromaDB vector store",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
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

    if not cfg.ANTHROPIC_API_KEY:
        console.print("[red]ERROR:[/red] ANTHROPIC_API_KEY not set. "
                      "Create a .env file with ANTHROPIC_API_KEY=sk-ant-...")
        return 1

    max_subjects = args.max_subjects if args.max_subjects > 0 else None
    console.print(
        f"[green]✓[/green] Config loaded | "
        f"Clusters: {args.clusters} | "
        f"Max subjects: {max_subjects or 'all'} | "
        f"Model: {cfg.CLAUDE_MODEL}"
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

    from src.graph import build_graph, initial_state

    console.print("\n[bold]Step 2/5:[/bold] Building LangGraph agent pipeline...")
    pipeline = build_graph(cfg, store)
    console.print("[green]✓[/green] Graph compiled (4 agent nodes + export)")

    console.print(f"\n[bold]Step 3-5/5:[/bold] Running pipeline on {args.clusters}...\n")
    t1 = time.time()

    state = initial_state(
        clusters=args.clusters,
        max_subjects=max_subjects,
    )

    final_state = pipeline.invoke(state)

    elapsed = time.time() - t1
    console.print(f"\n[green]✓[/green] Pipeline completed in {elapsed:.1f}s")

    _print_summary(final_state, elapsed)

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
            table.add_row(
                m["cluster"],
                m["variable"],
                m["code"],
                m["display"][:36] + ("…" if len(m["display"]) > 36 else ""),
                f"[{conf_style}]{m['confidence']}[/{conf_style}]",
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
