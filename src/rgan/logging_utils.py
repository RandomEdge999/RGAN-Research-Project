import contextlib
from typing import Any, Dict


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
        from rich.console import Console  # type: ignore
        from rich.theme import Theme  # type: ignore

        theme = Theme(
            {
                "banner.title": "bold white",
                "banner.subtitle": "italic bright_cyan",
                "panel.border": "bright_magenta",
                "progress.description": "bold bright_white",
                "progress.spinner": "magenta",
                "progress.bar": "magenta",
                "progress.complete": "bright_white",
                "progress.percentage": "bold bright_cyan",
                "progress.status": "dim white",
                "kv.key": "bold cyan",
                "kv.value": "white",
                "table.title": "bold magenta",
                "table.header": "bright_cyan",
            }
        )
        _CONSOLE = Console(theme=theme, highlight=False)
        return _CONSOLE
    except Exception:
        return _fallback_console()


def has_rich() -> bool:
    try:
        import rich  # noqa: F401
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
        with Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("{task.description}", style="bold bright_white"),
            BarColumn(bar_width=None, style="magenta", complete_style="bright_white"),
            TextColumn("{task.completed}/{task.total}", style="dim white"),
            TextColumn("{task.percentage:>5.1f}%", style="bold bright_cyan"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("| {task.fields[status]}", style="dim white"),
            expand=True,
        ) as progress:
            task_id = progress.add_task(description, total=total, status="initialising")
            yield progress, task_id
    else:
        yield None, None


def update_epoch(progress: Any, task_id: Any, epoch: int, total: int, metrics: Dict[str, float]):
    if progress is None:
        msg = " | ".join([f"{k}={v:.8f}" for k, v in metrics.items()])
        print(f"Epoch {epoch:03d}/{total:03d} | {msg}")
        return
    status = ", ".join([f"{k}:{v:.8f}" for k, v in metrics.items()])
    progress.update(task_id, advance=1, status=status)


def print_banner(console, title: str, subtitle: str = ""):
    if has_rich():
        from rich.panel import Panel
        from rich.text import Text
        from rich.align import Align

        header = Text(title.upper(), justify="center")
        if hasattr(header, "apply_gradient"):
            header.apply_gradient("#8b5cf6", "#ec4899")
        else:
            header.stylize("#8b5cf6")
        if subtitle:
            sub = Text(subtitle, justify="center", style="banner.subtitle")
            if hasattr(sub, "apply_gradient"):
                sub.apply_gradient("#22d3ee", "#8b5cf6")
            body = Align.center(Text.assemble(header, "\n", sub))
        else:
            body = Align.center(header)

        console.print(Panel.fit(body, border_style="panel.border", padding=(1, 2)))
    else:
        console.rule(title)


def print_kv_table(console, title: str, rows: Dict[str, Any]):
    if has_rich():
        from rich import box
        from rich.table import Table

        table = Table(
            title=title,
            box=box.ROUNDED,
            title_style="table.title",
            header_style="table.header",
            show_edge=True,
            expand=False,
        )
        table.add_column("Key", style="kv.key", no_wrap=True)
        table.add_column("Value", style="kv.value")
        for k, v in rows.items():
            table.add_row(str(k), str(v))
        console.print(table)
    else:
        console.rule(title)
        for k, v in rows.items():
            console.log(f"- {k}: {v}")


