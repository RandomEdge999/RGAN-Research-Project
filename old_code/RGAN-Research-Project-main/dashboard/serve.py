"""Utility to serve the dashboard with a localhost-bound HTTP server.

Usage::

        python serve.py [--port 8000] [--metrics ../results/metrics.json]

Features:
        • Binds to 127.0.0.1 so the printed URL is browser-friendly.
        • Optional metrics shortcut that opens the requested JSON even if it
            lives outside ``dashboard/``.
        • Graceful fallback when automatic browser launch is unsupported (e.g.
            WSL without an X server).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, metrics_root: Path | None = None, **kwargs):
        self.metrics_root = Path(metrics_root).resolve() if metrics_root else None
        super().__init__(*args, **kwargs)

    # Keep default logging behaviour

    def _send_metrics_file(self, target: Path) -> None:
        try:
            data = target.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Metrics file not found: {target}")
            return
        except OSError as exc:  # pragma: no cover - unlikely but defensive
            self.send_error(500, f"Unable to read metrics file: {exc}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - method signature is fixed
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/__metrics__":
            params = urllib.parse.parse_qs(parsed.query)
            raw_path = params.get("path", [""])[0]
            if not raw_path:
                self.send_error(400, "Missing 'path' query parameter")
                return

            candidate = Path(urllib.parse.unquote(raw_path))
            if not candidate.is_absolute():
                base = self.metrics_root or Path(self.directory or Path.cwd())
                candidate = (base / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if not candidate.exists() or not candidate.is_file():
                self.send_error(404, f"Metrics file not found: {candidate}")
                return

            self._send_metrics_file(candidate)
            return

        super().do_GET()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the dashboard locally")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server on (default: 8000)",
    )
    parser.add_argument(
        "--metrics",
        default="../results/metrics.json",
        help="Metrics JSON the dashboard should load (used for convenience link)",
    )
    parser.add_argument(
        "--metrics-root",
        default=None,
        help="Base directory for resolving relative metric paths (default: dashboard directory)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = "127.0.0.1"
    directory = Path(__file__).resolve().parent
    metrics_root = Path(args.metrics_root).resolve() if args.metrics_root else directory
    handler = functools.partial(
        DashboardRequestHandler,
        directory=str(directory),
        metrics_root=metrics_root,
    )

    with socketserver.TCPServer((host, args.port), handler) as httpd:
        httpd.allow_reuse_address = True

        url = f"http://{host}:{args.port}/?metrics={Path(args.metrics).as_posix()}"
        print(f"Serving dashboard from {directory}")
        print(f"Metrics source: {Path(args.metrics)}")
        print(f"Metrics root: {metrics_root}")
        print(f"Dashboard URL: {url}")
        print("Press Ctrl+C to stop.")

        if not args.no_browser:
            try:
                opened = webbrowser.open(url)
                if not opened:
                    print("Automatic browser launch not supported; open the URL manually.")
            except Exception:
                print("Automatic browser launch failed; open the URL manually.")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
