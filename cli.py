"""
codesearch CLI — semantic code search for any local codebase.

Commands:
  index   <path>   [--name NAME]
  search  <query>  [--codebase NAME] [--top-k N]
  list
  drop    <name>
"""

import os
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="codesearch",
    help="⚡ Semantic code search for any local codebase.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index(
    path: str = typer.Argument(..., help="Path to the codebase root directory."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for this codebase index. Defaults to the directory name."),
):
    """Index a local codebase so it can be searched semantically."""
    from pathlib import Path
    from indexer import index_codebase

    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        console.print(f"[red]Error:[/red] '{path}' is not a valid directory.")
        raise typer.Exit(1)

    codebase_name = name or root.name
    console.print(f"\n[bold cyan]Indexing[/bold cyan] [green]{root}[/green] as [bold]{codebase_name}[/bold]\n")

    progress_state = {"current": 0, "total": 0, "task": None}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=None)
        progress_state["task"] = task

        def on_progress(current, total, filepath):
            rel = filepath
            try:
                rel = str(Path(filepath).relative_to(root))
            except ValueError:
                pass
            progress.update(task, total=total, completed=current, description=f"[cyan]{rel}")

        summary = index_codebase(path=str(root), name=codebase_name, progress_callback=on_progress)

    # Summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]Files scanned[/dim]",    f"[bold]{summary['total_files']}[/bold]")
    table.add_row("[dim]Files skipped (cached)[/dim]", f"[bold]{summary['skipped_files']}[/bold]")
    table.add_row("[dim]Chunks extracted[/dim]", f"[bold]{summary['total_chunks']}[/bold]")
    table.add_row("[dim]New chunks indexed[/dim]", f"[bold green]{summary['new_chunks']}[/bold green]")

    console.print(Panel(table, title=f"[bold]✅ {codebase_name}[/bold]", border_style="green"))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language description of what you're looking for."),
    codebase: Optional[str] = typer.Option(None, "--codebase", "-c", help="Name of a specific codebase to search. Omit to search all."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results to return."),
):
    """Search indexed codebases with a natural language query."""
    from indexer import search_codebase

    target = f"[bold]{codebase}[/bold]" if codebase else "[dim]all codebases[/dim]"
    console.print(f"\n[bold cyan]Searching[/bold cyan] {target} for: [italic]\"{query}\"[/italic]\n")

    with console.status("[cyan]Running semantic search + re-ranking…", spinner="dots"):
        results = search_codebase(query=query, codebase_name=codebase, top_k_return=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow] Try rephrasing your query or check that the codebase is indexed.")
        return

    for i, r in enumerate(results):
        score_color = "green" if r["score"] > 2 else "yellow" if r["score"] > 0 else "red"
        header = (
            f"[bold]#{i+1}[/bold]  "
            f"[{score_color}]score {r['score']:.3f}[/{score_color}]  "
            f"[dim]{r.get('file_path', '?')}[/dim]"
            f"  [dim italic]:{r.get('start_line','?')}-{r.get('end_line','?')}[/dim italic]"
        )
        if r.get("function_name"):
            header += f"  [cyan]{r['function_name']}[/cyan]"

        console.print(f"\n{header}")
        console.print(f"[dim]{r.get('description', '')}[/dim]")

        syntax = Syntax(
            r.get("code", ""),
            r.get("language", "python"),
            theme="github-dark",
            line_numbers=True,
            start_line=r.get("start_line", 1),
        )
        console.print(syntax)
        console.print("[dim]" + "─" * 80 + "[/dim]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_codebases():
    """List all indexed codebases and their snippet counts."""
    import chromadb
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    from indexer import get_all_codebase_collections

    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    names = get_all_codebase_collections(chroma)

    if not names:
        console.print("[yellow]No codebases indexed yet.[/yellow] Run [bold]codesearch index <path>[/bold] to get started.")
        return

    table = Table(title="Indexed Codebases", show_lines=True)
    table.add_column("Name",    style="bold cyan", no_wrap=True)
    table.add_column("Chunks",  justify="right")
    table.add_column("Languages", style="dim")

    for col_name in names:
        display_name = col_name.removeprefix("codebase_")
        col = chroma.get_collection(col_name)
        count = col.count()

        # Sample to detect languages
        sample = col.peek(limit=50)
        langs = set()
        for meta in (sample.get("metadatas") or []):
            if meta and meta.get("language"):
                langs.add(meta["language"])

        table.add_row(display_name, str(count), ", ".join(sorted(langs)) or "—")

    console.print(table)


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

@app.command()
def drop(
    name: str = typer.Argument(..., help="Name of the codebase index to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete a codebase index from the vector database."""
    import chromadb
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    col_name = f"codebase_{name}"
    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing = [c.name for c in chroma.list_collections()]
    if col_name not in existing:
        console.print(f"[red]No index found for '[bold]{name}[/bold]'.[/red]")
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"Delete index '{name}'? This cannot be undone.", abort=True)

    chroma.delete_collection(col_name)
    console.print(f"[green]✓[/green] Deleted index [bold]{name}[/bold].")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
