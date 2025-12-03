import contextlib
from typing import Any, Dict, Optional
import sys
import platform

def _fallback_console():
    class Simple:
        def rule(self, text: str):
            print("\n" + "=" * 80)
            print(text)
            print("=" * 80)

        def log(self, *args, **kwargs):
            print(*args)

        def print(self, *args, **kwargs):
            print(*args)
    return Simple()

_CONSOLE = None

def get_console():
    global _CONSOLE
    if _CONSOLE is not None:
        return _CONSOLE
    try:
        from rich.console import Console
        from rich.theme import Theme

        # Premium "Trading Terminal" Theme
        # Dark Slate / Cyan / Neon Green palette
        theme = Theme({
            "banner.title": "bold white on #0f172a",
            "banner.subtitle": "italic #38bdf8 on #0f172a",
            "panel.border": "#38bdf8",
            "table.header": "bold #f0f9ff on #0f172a",
            "table.row": "#e2e8f0",
            "table.row.even": "#cbd5e1",
            "progress.description": "bold #f0f9ff",
            "progress.percentage": "bold #38bdf8",
            "progress.remaining": "italic #94a3b8",
            "kv.key": "bold #0ea5e9",
            "kv.value": "#f8fafc",
        })
        _CONSOLE = Console(theme=theme, highlight=False)
        return _CONSOLE
    except Exception:
        return _fallback_console()

def has_rich() -> bool:
    try:
        import rich
        return True
    except Exception:
        return False

@contextlib.contextmanager
def epoch_progress(total: int, description: str = "Training"):
    if has_rich():
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            TextColumn,
        )
        # Custom "Radar" Spinner and detailed columns
        with Progress(
            SpinnerColumn(spinner_name="dots", style="#38bdf8"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="#1e293b", complete_style="#38bdf8", finished_style="#22c55e"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•", style="#475569"),
            TimeElapsedColumn(),
            TextColumn("•", style="#475569"),
            TimeRemainingColumn(),
            TextColumn("•", style="#475569"),
            TextColumn("{task.fields[status]}", style="#94a3b8"),
            expand=True,
            transient=False, # Keep the bar after completion for history
        ) as progress:
            task_id = progress.add_task(description, total=total, status="Initializing...")
            yield progress, task_id
    else:
        yield None, None

def update_epoch(progress: Any, task_id: Any, epoch: int, total: int, metrics: Dict[str, float]):
    if progress is None:
        msg = " | ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        print(f"Epoch {epoch:03d}/{total:03d} | {msg}")
        return
    
    # Format metrics for the progress bar
    status_parts = []
    for k, v in metrics.items():
        color = "#22c55e" if "loss" not in k.lower() else "#f43f5e" # Green for metrics, Red for loss (heuristic)
        if "rmse" in k.lower(): color = "#eab308" # Yellow for RMSE
        status_parts.append(f"[{color}]{k}[/{color}]: {v:.6f}")
    
    status = " | ".join(status_parts)
    progress.update(task_id, advance=1, status=status)

def print_banner(console, title: str, subtitle: str = ""):
    if has_rich():
        from rich.panel import Panel
        from rich.text import Text
        from rich.align import Align
        from rich.table import Table
        from rich import box
        
        # Create a grid for the banner content
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        
        # Title with Gradient (simulated with colors if gradient not supported, but let's try standard styles)
        title_text = Text(title.upper(), style="bold white")
        subtitle_text = Text(subtitle, style="italic #38bdf8")
        
        grid.add_row(title_text)
        grid.add_row(subtitle_text)
        grid.add_row(Text("─" * 40, style="dim #475569")) # Separator

        # System Info Row
        sys_info = f"Python {sys.version.split()[0]} | {platform.system()} | Rich UI"
        grid.add_row(Text(sys_info, style="dim #94a3b8"))

        console.print(Panel(
            grid,
            style="on #0f172a",
            border_style="#38bdf8",
            padding=(1, 2),
            box=box.HEAVY_EDGE
        ))
    else:
        console.rule(title)
        print(subtitle)

def print_kv_table(console, title: str, rows: Dict[str, Any]):
    if has_rich():
        from rich import box
        from rich.table import Table

        table = Table(
            title=title,
            box=box.ROUNDED,
            title_style="bold #f0f9ff",
            header_style="bold #38bdf8 on #1e293b",
            border_style="#475569",
            show_edge=True,
            expand=True,
            row_styles=["none", "dim"] # Alternating rows
        )
        table.add_column("Configuration", style="bold #0ea5e9", no_wrap=True, ratio=1)
        table.add_column("Value", style="#f8fafc", ratio=2)
        
        for k, v in rows.items():
            table.add_row(str(k), str(v))
        
        console.print(table)
    else:
        console.rule(title)
        for k, v in rows.items():
            console.log(f"- {k}: {v}")
