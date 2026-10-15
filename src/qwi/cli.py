"""Command-line entry point. Commands land milestone by milestone (PLAN.md §5 CLI reference)."""

import typer

app = typer.Typer(no_args_is_help=True, help="Local text-to-image pipeline for Apple Silicon.")


@app.callback()
def main() -> None:
    """qwi: guard → rewrite → generate N seeds → rerank."""
