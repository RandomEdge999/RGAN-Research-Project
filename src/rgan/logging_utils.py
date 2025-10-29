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


def get_console():
    try:
        from rich.console import Console  # type: ignore
        return Console()
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
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("| {task.fields[status]}")
        ) as progress:
            task_id = progress.add_task(description, total=total, status="...")
            yield progress, task_id
    else:
        yield None, None


def update_epoch(progress: Any, task_id: Any, epoch: int, total: int, metrics: Dict[str, float]):
    if progress is None:
        msg = " | ".join([f"{k}={v:.5f}" for k, v in metrics.items()])
        print(f"Epoch {epoch:03d}/{total:03d} | {msg}")
        return
    status = ", ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
    progress.update(task_id, advance=1, status=status)


def print_banner(console, title: str, subtitle: str = ""):
    if has_rich():
        from rich.panel import Panel
        from rich.text import Text
        header = Text(title, style="bold magenta")
        sub = Text(subtitle, style="dim") if subtitle else ""
        console.print(Panel.fit(Text.assemble(header, "\n", sub) if sub else header))
    else:
        console.rule(title)


def print_kv_table(console, title: str, rows: Dict[str, Any]):
    if has_rich():
        from rich.table import Table
        table = Table(title=title)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for k, v in rows.items():
            table.add_row(str(k), str(v))
        console.print(table)
    else:
        console.rule(title)
        for k, v in rows.items():
            console.log(f"- {k}: {v}")


