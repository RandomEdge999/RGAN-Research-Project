#!/usr/bin/env python3
"""Download curated natural/noisy time-series datasets into data/ with README metadata."""

from __future__ import annotations

import csv
import json
import math
import textwrap
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def _http_get(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RGAN-Research-Project/1.0 (dataset fetcher)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read().decode("utf-8")


def _month_starts(year: int) -> Iterable[tuple[date, date]]:
    for month in range(1, 13):
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - pd.Timedelta(days=1)
        yield start, end


def _sanitize_columns(columns: Iterable[str]) -> List[str]:
    cleaned = []
    for col in columns:
        value = col.strip().lower()
        value = value.replace("(for verified)", "verified")
        value = value.replace("%", "pct")
        for old, new in (
            ("date time", "time"),
            (" ", "_"),
            ("/", "_per_"),
            ("-", "_"),
            ("(", ""),
            (")", ""),
            (",", ""),
            (".", ""),
        ):
            value = value.replace(old, new)
        while "__" in value:
            value = value.replace("__", "_")
        cleaned.append(value.strip("_"))
    return cleaned


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _time_summary(df: pd.DataFrame, time_col: str = "time") -> str:
    start = pd.to_datetime(df[time_col]).min()
    end = pd.to_datetime(df[time_col]).max()
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    source_name: str
    source_url: str
    target_col: str
    time_col: str
    resolution: str
    noise_type: str
    fetcher: Callable[[], pd.DataFrame]
    notes: List[str]


def fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M,WS2M,PS",
        "community": "RE",
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start,
        "end": end,
        "format": "csv",
        "header": "false",
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point?" + urllib.parse.urlencode(params)
    df = pd.read_csv(StringIO(_http_get(url)))
    df["time"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]),
        utc=True,
    )
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"])
    cols = ["time"] + [c for c in df.columns if c != "time"]
    return df[cols]


def fetch_noaa_coops_water_level(station: str, year: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for start, end in _month_starts(year):
        params = {
            "product": "water_level",
            "application": "RGANResearch",
            "begin_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "datum": "MSL",
            "station": station,
            "time_zone": "gmt",
            "units": "metric",
            "format": "csv",
        }
        url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?" + urllib.parse.urlencode(params)
        frame = pd.read_csv(StringIO(_http_get(url)), skipinitialspace=True)
        frame.columns = _sanitize_columns(frame.columns)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").reset_index(drop=True)


def fetch_ndbc_stdmet(station: str, year: int) -> pd.DataFrame:
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"
    raw = _http_get(url)
    frame = pd.read_csv(StringIO(raw), sep=r"\s+", skiprows=[1], engine="python")
    frame["time"] = pd.to_datetime(
        dict(year=frame["#YY"], month=frame["MM"], day=frame["DD"], hour=frame["hh"], minute=frame["mm"]),
        utc=True,
    )
    frame = frame.drop(columns=["#YY", "MM", "DD", "hh", "mm"])
    frame.columns = _sanitize_columns(frame.columns)
    cols = ["time"] + [c for c in frame.columns if c != "time"]
    return frame[cols]


USGS_PARAM_MAP: Dict[str, str] = {
    "00060": "streamflow_cfs",
}


def fetch_usgs_iv(site: str, parameter: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "format": "json",
        "sites": site,
        "startDT": start,
        "endDT": end,
        "parameterCd": parameter,
        "siteStatus": "all",
    }
    url = "https://nwis.waterservices.usgs.gov/nwis/iv/?" + urllib.parse.urlencode(params)
    payload = json.loads(_http_get(url))
    series = payload["value"]["timeSeries"]
    if not series:
        raise ValueError(f"No USGS time series returned for site={site} parameter={parameter}")
    values = series[0]["values"][0]["value"]
    rows = [
        {
            "time": pd.to_datetime(item["dateTime"], utc=True),
            USGS_PARAM_MAP.get(parameter, f"parameter_{parameter}"): float(item["value"]),
            "qualifier": ",".join(item.get("qualifiers", [])),
        }
        for item in values
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("time").reset_index(drop=True)


SPECS: List[DatasetSpec] = [
    DatasetSpec(
        slug="nasa_power_austin_hourly",
        title="NASA POWER Austin Hourly Meteorology",
        source_name="NASA POWER API",
        source_url="https://power.larc.nasa.gov/docs/services/api/temporal/hourly/",
        target_col="T2M",
        time_col="time",
        resolution="Hourly",
        noise_type="Atmospheric variability, storms, humidity shifts, reanalysis/observation uncertainty",
        fetcher=lambda: fetch_nasa_power(30.2672, -97.7431, "20170101", "20241231"),
        notes=[
            "Location: Austin, Texas, USA",
            "Recommended target: T2M (2m air temperature)",
        ],
    ),
    DatasetSpec(
        slug="nasa_power_phoenix_hourly",
        title="NASA POWER Phoenix Hourly Meteorology",
        source_name="NASA POWER API",
        source_url="https://power.larc.nasa.gov/docs/services/api/temporal/hourly/",
        target_col="T2M",
        time_col="time",
        resolution="Hourly",
        noise_type="Dry-climate temperature swings, convective events, reanalysis uncertainty",
        fetcher=lambda: fetch_nasa_power(33.4484, -112.0740, "20170101", "20241231"),
        notes=[
            "Location: Phoenix, Arizona, USA",
            "Recommended target: T2M (2m air temperature)",
        ],
    ),
    DatasetSpec(
        slug="nasa_power_denver_hourly",
        title="NASA POWER Denver Hourly Meteorology",
        source_name="NASA POWER API",
        source_url="https://power.larc.nasa.gov/docs/services/api/temporal/hourly/",
        target_col="T2M",
        time_col="time",
        resolution="Hourly",
        noise_type="Mountain-plain weather shifts, snow/cold fronts, reanalysis uncertainty",
        fetcher=lambda: fetch_nasa_power(39.7392, -104.9903, "20170101", "20241231"),
        notes=[
            "Location: Denver, Colorado, USA",
            "Recommended target: T2M (2m air temperature)",
        ],
    ),
    DatasetSpec(
        slug="noaa_coops_san_francisco_water_level",
        title="NOAA CO-OPS San Francisco Water Level",
        source_name="NOAA CO-OPS Data API",
        source_url="https://api.tidesandcurrents.noaa.gov/api/prod/",
        target_col="water_level",
        time_col="time",
        resolution="6-minute",
        noise_type="Tides, storm surge, harbor dynamics, sensor and meteorological disturbance",
        fetcher=lambda: fetch_noaa_coops_water_level("9414290", 2024),
        notes=[
            "Station: 9414290 (San Francisco, California)",
            "Datum: Mean Sea Level (MSL)",
        ],
    ),
    DatasetSpec(
        slug="noaa_coops_seattle_water_level",
        title="NOAA CO-OPS Seattle Water Level",
        source_name="NOAA CO-OPS Data API",
        source_url="https://api.tidesandcurrents.noaa.gov/api/prod/",
        target_col="water_level",
        time_col="time",
        resolution="6-minute",
        noise_type="Tides, estuarine forcing, storms, sensor disturbance",
        fetcher=lambda: fetch_noaa_coops_water_level("9447130", 2024),
        notes=[
            "Station: 9447130 (Seattle, Washington)",
            "Datum: Mean Sea Level (MSL)",
        ],
    ),
    DatasetSpec(
        slug="noaa_coops_key_west_water_level",
        title="NOAA CO-OPS Key West Water Level",
        source_name="NOAA CO-OPS Data API",
        source_url="https://api.tidesandcurrents.noaa.gov/api/prod/",
        target_col="water_level",
        time_col="time",
        resolution="6-minute",
        noise_type="Tides, coastal weather, pressure effects, sensor disturbance",
        fetcher=lambda: fetch_noaa_coops_water_level("8724580", 2024),
        notes=[
            "Station: 8724580 (Key West, Florida)",
            "Datum: Mean Sea Level (MSL)",
        ],
    ),
    DatasetSpec(
        slug="noaa_ndbc_44013_stdmet",
        title="NOAA NDBC Buoy 44013 Standard Meteorology",
        source_name="NOAA National Data Buoy Center",
        source_url="https://www.ndbc.noaa.gov/historical_data.shtml",
        target_col="wvht",
        time_col="time",
        resolution="10-minute",
        noise_type="Open-ocean wave and meteorological variability, storms, sensor dropout",
        fetcher=lambda: fetch_ndbc_stdmet("44013", 2024),
        notes=[
            "Station: 44013 (Boston buoy)",
            "Recommended target: WVHT (significant wave height)",
        ],
    ),
    DatasetSpec(
        slug="noaa_ndbc_41009_stdmet",
        title="NOAA NDBC Buoy 41009 Standard Meteorology",
        source_name="NOAA National Data Buoy Center",
        source_url="https://www.ndbc.noaa.gov/historical_data.shtml",
        target_col="wvht",
        time_col="time",
        resolution="10-minute",
        noise_type="Open-ocean wave and meteorological variability, tropical systems, sensor dropout",
        fetcher=lambda: fetch_ndbc_stdmet("41009", 2024),
        notes=[
            "Station: 41009 (Canaveral East buoy)",
            "Recommended target: WVHT (significant wave height)",
        ],
    ),
    DatasetSpec(
        slug="usgs_napa_river_streamflow",
        title="USGS Napa River Streamflow",
        source_name="USGS NWIS Instantaneous Values",
        source_url="https://waterdata.usgs.gov/nwis/uv",
        target_col="streamflow_cfs",
        time_col="time",
        resolution="15-minute",
        noise_type="River flow variability, storm pulses, hydrologic measurement noise",
        fetcher=lambda: fetch_usgs_iv("11458000", "00060", "2023-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        notes=[
            "Site: 11458000 (Napa River near Napa, CA)",
            "Parameter: 00060 discharge (cubic feet per second)",
        ],
    ),
    DatasetSpec(
        slug="usgs_potomac_river_streamflow",
        title="USGS Potomac River Streamflow",
        source_name="USGS NWIS Instantaneous Values",
        source_url="https://waterdata.usgs.gov/nwis/uv",
        target_col="streamflow_cfs",
        time_col="time",
        resolution="15-minute",
        noise_type="River flow variability, storm pulses, regulation effects, measurement noise",
        fetcher=lambda: fetch_usgs_iv("01646500", "00060", "2023-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        notes=[
            "Site: 01646500 (Potomac River near Washington, DC)",
            "Parameter: 00060 discharge (cubic feet per second)",
        ],
    ),
]


README_TEMPLATE = """# {title}

- **Rows:** {rows:,}
- **Resolution:** {resolution}
- **Period:** {period}
- **Source:** {source_name}
- **Source URL:** {source_url}
- **Target column:** `{target_col}`
- **Time column:** `{time_col}`
- **File:** `{filename}`
- **Missing values:** {missing}
- **Noise type:** {noise_type}
{notes_block}
- **CLI usage:**
  ```
  rgan-train --csv {rel_path} --target {target_col} --time_col {time_col}
  ```
"""


def build_readme(spec: DatasetSpec, df: pd.DataFrame, csv_name: str) -> str:
    missing = int(df.isna().sum().sum())
    notes_block = "\n".join(f"- **Note:** {note}" for note in spec.notes)
    return README_TEMPLATE.format(
        title=spec.title,
        rows=len(df),
        resolution=spec.resolution,
        period=_time_summary(df, spec.time_col),
        source_name=spec.source_name,
        source_url=spec.source_url,
        target_col=spec.target_col,
        time_col=spec.time_col,
        filename=csv_name,
        missing=f"{missing:,} cells total" if missing else "None",
        noise_type=spec.noise_type,
        notes_block=notes_block,
        rel_path=f"data/{spec.slug}/{csv_name}",
    )


def main() -> None:
    results = []
    for spec in SPECS:
        print(f"[fetch] {spec.slug}")
        df = spec.fetcher()
        out_dir = DATA_ROOT / spec.slug
        csv_name = f"{spec.slug}.csv"
        csv_path = out_dir / csv_name
        _write_csv(df, csv_path)
        _write_text(out_dir / "README.md", build_readme(spec, df, csv_name))
        results.append((spec.slug, len(df), csv_path))

    print("\nFetched datasets:")
    for slug, rows, path in results:
        print(f"- {slug}: {rows:,} rows -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
