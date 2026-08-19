"""Download and validate the 2021-2025 CEDA NIMROD 1 km archive."""

from __future__ import annotations

import csv
import argparse
import email.utils
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests


YEARS = (2021, 2022, 2023, 2024, 2025)
REMOTE_ROOT = "https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/"
REPO_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = REPO_ROOT / "data_raw" / "dynamic_rainfall" / "nimrod_1km_5min"
TOKEN_PATH = DESTINATION / "ceda_token.txt"
MANIFEST_PATH = DESTINATION / "nimrod_download_manifest.csv"
SUMMARY_PATH = DESTINATION / "nimrod_download_summary.txt"
MANIFEST_COLUMNS = (
    "year",
    "filename",
    "remote_url",
    "expected_size_bytes",
    "local_path",
    "download_status",
    "local_size_bytes",
    "http_status",
)
MIN_FREE_BYTES = 120 * 1024**3
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 6
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 120


class AuthenticationError(RuntimeError):
    """Raised when CEDA rejects the bearer token."""


class ListingParser(HTMLParser):
    """Parse links and the trailing metadata from an nginx directory index."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, str]] = []
        self._href: str | None = None
        self._awaiting_metadata = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._href = dict(attrs).get("href")
        self._awaiting_metadata = False

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self._awaiting_metadata = True

    def handle_data(self, data: str) -> None:
        if not self._awaiting_metadata or self._href is None:
            return
        metadata = data.strip()
        if metadata:
            self.entries.append((self._href, metadata))
            self._href = None
            self._awaiting_metadata = False


@dataclass
class DownloadTotals:
    expected: int
    completed_before_run: int
    transferred: int = 0

    @property
    def completed(self) -> int:
        return min(self.expected, self.completed_before_run + self.transferred)


class DownloadProgress:
    """Render byte-level file, year, and overall progress on one terminal line."""

    def __init__(
        self,
        *,
        year: int,
        file_number: int,
        file_count: int,
        filename: str,
        file_expected: int,
        file_existing: int,
        year_totals: DownloadTotals,
        overall_totals: DownloadTotals,
    ) -> None:
        self.year = year
        self.file_number = file_number
        self.file_count = file_count
        self.filename = filename
        self.file_expected = file_expected
        self.file_downloaded = file_existing
        self.year_totals = year_totals
        self.overall_totals = overall_totals
        self.started = time.monotonic()
        self.last_rendered = 0.0
        self.last_non_tty_percent = -10
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.render(force=True)

    @staticmethod
    def _size(value: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(value) < 1024 or unit == "TB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{value:d} B"
            value /= 1024
        raise AssertionError("unreachable")

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def update(self, byte_count: int) -> None:
        self.file_downloaded += byte_count
        self.year_totals.transferred += byte_count
        self.overall_totals.transferred += byte_count
        self.render()

    def render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_rendered < 0.25:
            return
        elapsed = max(now - self.started, 0.001)
        fraction = self.file_downloaded / self.file_expected if self.file_expected else 1.0
        percent = min(100, int(fraction * 100))
        if not self.is_tty and not force and percent < self.last_non_tty_percent + 10:
            return
        width = 24
        filled = min(width, int(width * min(1.0, fraction)))
        bar = "#" * filled + "-" * (width - filled)
        detail = (
            f"Year {self.year} | File {self.file_number}/{self.file_count} "
            f"[{bar}] {percent:3d}% | {self.filename} | "
            f"{self._size(self.file_downloaded)}/{self._size(self.file_expected)} | "
            f"year {self._size(self.year_totals.completed)} | "
            f"overall {self._size(self.overall_totals.completed)} | "
            f"{self._size(int(max(0, self.file_downloaded) / elapsed))}/s | "
            f"elapsed {self._duration(elapsed)}"
        )
        prefix = "\r" if self.is_tty else ""
        suffix = "" if self.is_tty else "\n"
        sys.stderr.write(prefix + detail + suffix)
        sys.stderr.flush()
        self.last_rendered = now
        self.last_non_tty_percent = percent

    def close(self) -> None:
        self.file_downloaded = self.file_expected
        self.render(force=True)
        if self.is_tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def format_size(value: int) -> str:
    return f"{value / 1024**3:.2f} GB"


def load_token() -> str:
    if not TOKEN_PATH.is_file():
        raise SystemExit(f"Token file is absent: {TOKEN_PATH}")
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"Token file is empty: {TOKEN_PATH}")
    return token


def retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            return min(120.0, float(retry_after))
        if retry_after:
            try:
                when = email.utils.parsedate_to_datetime(retry_after)
                return min(120.0, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError):
                pass
    return min(60.0, 2.0 ** (attempt - 1))


def checked_request(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: object,
) -> requests.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response: requests.Response | None = None
        try:
            response = session.request(
                method,
                url,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                **kwargs,
            )
            if response.status_code in (401, 403):
                response.close()
                raise AuthenticationError
            if response.status_code != 429 and response.status_code < 500:
                return response
        except AuthenticationError:
            raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Request failed after {MAX_ATTEMPTS} attempts: {url}") from exc
        if attempt == MAX_ATTEMPTS:
            status = response.status_code if response is not None else "network error"
            if response is not None:
                response.close()
            raise RuntimeError(f"Request failed after {MAX_ATTEMPTS} attempts ({status}): {url}")
        delay = retry_delay(response, attempt)
        if response is not None:
            response.close()
        print(f"Temporary CEDA error; retrying in {delay:.0f}s ({attempt}/{MAX_ATTEMPTS})", file=sys.stderr)
        time.sleep(delay)
    raise AssertionError("unreachable")


def enumerate_year(session: requests.Session, year: int) -> tuple[list[dict[str, object]], int]:
    year_url = urljoin(REMOTE_ROOT, f"{year}/")
    response = checked_request(session, "GET", year_url)
    status = response.status_code
    response.raise_for_status()
    parser = ListingParser()
    parser.feed(response.text)
    response.close()

    rows: list[dict[str, object]] = []
    expected_prefix = urlparse(year_url).path
    for href, metadata in parser.entries:
        remote_url = urljoin(year_url, href)
        parsed = urlparse(remote_url)
        if parsed.scheme != "https" or parsed.netloc != "dap.ceda.ac.uk":
            continue
        if not parsed.path.startswith(expected_prefix) or parsed.path.endswith("/"):
            continue
        filename = Path(unquote(parsed.path)).name
        size_text = metadata.rsplit(maxsplit=1)[-1]
        if not size_text.isdigit():
            raise RuntimeError(f"CEDA listing has no byte size for {filename}")
        expected_size = int(size_text)
        local_path = DESTINATION / str(year) / filename
        local_size = local_path.stat().st_size if local_path.is_file() else 0
        part_path = local_path.with_name(local_path.name + ".part")
        part_size = part_path.stat().st_size if part_path.is_file() else 0
        complete = local_size == expected_size
        rows.append(
            {
                "year": year,
                "filename": filename,
                "remote_url": remote_url,
                "expected_size_bytes": expected_size,
                "local_path": str(local_path.relative_to(REPO_ROOT)),
                "download_status": "COMPLETE" if complete else "INCOMPLETE",
                "local_size_bytes": local_size if complete else max(local_size, part_size),
                "http_status": status,
            }
        )
    if not rows:
        raise RuntimeError(f"No archive files detected in the {year} CEDA listing")
    rows.sort(key=lambda row: str(row["filename"]))
    return rows, status


def write_manifest(rows: list[dict[str, object]]) -> None:
    temporary = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, MANIFEST_PATH)


def prepare_part_path(local_path: Path, expected_size: int) -> tuple[Path, int]:
    part_path = local_path.with_name(local_path.name + ".part")
    if local_path.is_file() and local_path.stat().st_size != expected_size:
        if part_path.is_file() and part_path.stat().st_size >= local_path.stat().st_size:
            local_path.unlink()
        else:
            if part_path.exists():
                part_path.unlink()
            os.replace(local_path, part_path)
    if part_path.is_file() and part_path.stat().st_size > expected_size:
        part_path.unlink()
    return part_path, part_path.stat().st_size if part_path.is_file() else 0


def download_row(
    session: requests.Session,
    row: dict[str, object],
    progress_factory: object,
) -> int:
    local_path = REPO_ROOT / str(row["local_path"])
    expected_size = int(row["expected_size_bytes"])
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_file() and local_path.stat().st_size == expected_size:
        row["download_status"] = "COMPLETE"
        row["local_size_bytes"] = expected_size
        return 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        part_path, existing_size = prepare_part_path(local_path, expected_size)
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        response: requests.Response | None = None
        progress: DownloadProgress | None = None
        try:
            response = checked_request(session, "GET", str(row["remote_url"]), headers=headers, stream=True)
            row["http_status"] = response.status_code
            response.raise_for_status()
            append = existing_size > 0 and response.status_code == 206
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {existing_size}-"):
                    raise RuntimeError(f"Invalid Content-Range for {row['filename']}: {content_range}")
            elif existing_size:
                existing_size = 0
            progress = progress_factory(existing_size)
            mode = "ab" if append else "wb"
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
            response.close()
            actual_size = part_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {row['filename']}: expected {expected_size}, received {actual_size}"
                )
            os.replace(part_path, local_path)
            row["download_status"] = "COMPLETE"
            row["local_size_bytes"] = expected_size
            progress.close()
            return expected_size - existing_size
        except AuthenticationError:
            if response is not None:
                response.close()
            raise
        except (OSError, requests.RequestException, RuntimeError) as exc:
            if response is not None:
                response.close()
            if progress is not None and progress.is_tty:
                sys.stderr.write("\n")
            if attempt == MAX_ATTEMPTS:
                row["download_status"] = "FAILED"
                part_size = part_path.stat().st_size if part_path.is_file() else 0
                row["local_size_bytes"] = part_size
                print(f"FAILED after {MAX_ATTEMPTS} attempts: {row['filename']} ({exc})", file=sys.stderr)
                return 0
            delay = min(60.0, 2.0 ** (attempt - 1))
            print(
                f"Download interrupted; preserving .part and retrying in {delay:.0f}s "
                f"({attempt}/{MAX_ATTEMPTS})",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def validate_year(year: int, rows: list[dict[str, object]]) -> dict[str, object]:
    expected = {str(row["filename"]): int(row["expected_size_bytes"]) for row in rows}
    year_dir = DESTINATION / str(year)
    local_files = {path.name: path for path in year_dir.iterdir() if path.is_file() and not path.name.endswith(".part")}
    complete = [name for name, size in expected.items() if name in local_files and local_files[name].stat().st_size == size]
    missing = [name for name in expected if name not in local_files]
    incomplete = [name for name, size in expected.items() if name in local_files and local_files[name].stat().st_size != size]
    unexpected = [name for name in local_files if name not in expected]
    passed = not missing and not incomplete and not unexpected
    print(f"\n{year} VALIDATION")
    print(f"Remote files: {len(expected)}")
    print(f"Local complete files: {len(complete)}")
    print(f"Missing files: {len(missing)}")
    print(f"Incomplete files: {len(incomplete)}")
    print(f"Unexpected files: {len(unexpected)}")
    print(f"Validation: {'PASS' if passed else 'FAIL'}")
    return {
        "remote": len(expected),
        "complete": len(complete),
        "missing_failed": len(missing) + len(incomplete),
        "size": sum(expected.values()),
        "passed": passed,
    }


def write_summary(validations: dict[int, dict[str, object]], rows: list[dict[str, object]]) -> None:
    lines = [
        "=" * 60,
        "NIMROD RAW DOWNLOAD - FINAL REPORT",
        "=" * 60,
        "",
        "SOURCE",
        "CEDA dataset:",
        "1 km Resolution UK Composite Rainfall Data from the Met Office Nimrod System",
        "",
        "REMOTE ROOT:",
        REMOTE_ROOT,
        "",
        "YEARS REQUESTED:",
        *(str(year) for year in YEARS),
        "",
        "DESTINATION:",
        "data_raw/dynamic_rainfall/nimrod_1km_5min/",
        "",
        "YEAR    REMOTE FILES    COMPLETE    MISSING/FAILED    SIZE",
    ]
    for year in YEARS:
        result = validations.get(year, {})
        lines.append(
            f"{year}    {result.get('remote', 0):12}    {result.get('complete', 0):8}    "
            f"{result.get('missing_failed', 0):14}    {format_size(int(result.get('size', 0)))}"
        )
    total_size = sum(int(row["expected_size_bytes"]) for row in rows)
    token_ignored = "data_raw/dynamic_rainfall/nimrod_1km_5min/ceda_token.txt" in (
        REPO_ROOT / ".gitignore"
    ).read_text(encoding="utf-8").splitlines()
    lines.extend(
        [
            "",
            f"TOTAL FILES: {len(rows)}",
            f"TOTAL SIZE: {format_size(total_size)}",
            "",
            "SECURITY",
            "Token printed: NO",
            "Token committed to Git: NO",
            f"Token file ignored by Git: {'YES' if token_ignored else 'NO'}",
            "",
            "RAW DATA",
            "Archives extracted: NO",
            "Files modified: NO",
            "",
            "VALIDATION",
            *(f"{year}: {'PASS' if validations.get(year, {}).get('passed') else 'FAIL'}" for year in YEARS),
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="authenticate, enumerate CEDA, and write the manifest without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        (DESTINATION / str(year)).mkdir(exist_ok=True)
    token = load_token()
    free_bytes = __import__("shutil").disk_usage(DESTINATION).free
    if free_bytes < MIN_FREE_BYTES:
        raise SystemExit(f"Available disk space is below 120 GB: {format_size(free_bytes)}")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "individual-cso-analysis/1.0"})
    all_rows: list[dict[str, object]] = []
    try:
        print("CEDA authentication: PASS")
        for index, year in enumerate(YEARS):
            year_rows, _ = enumerate_year(session, year)
            if index == 0:
                print("2021 directory listing: PASS")
                print(f"Files detected: {len(year_rows)}")
            all_rows.extend(year_rows)
            print(f"{year}: {len(year_rows)} files, approximately {format_size(sum(int(r['expected_size_bytes']) for r in year_rows))}")
    except AuthenticationError:
        print_authentication_failure()
        return 2

    print(f"\nTOTAL: {len(all_rows)} files, approximately {format_size(sum(int(r['expected_size_bytes']) for r in all_rows))}")
    write_manifest(all_rows)
    if args.manifest_only:
        session.close()
        return 0

    overall_expected = sum(int(row["expected_size_bytes"]) for row in all_rows)
    completed_before = sum(
        int(row["expected_size_bytes"]) for row in all_rows if row["download_status"] == "COMPLETE"
    )
    overall_totals = DownloadTotals(overall_expected, completed_before)
    validations: dict[int, dict[str, object]] = {}
    try:
        for year in YEARS:
            year_rows = [row for row in all_rows if int(row["year"]) == year]
            year_expected = sum(int(row["expected_size_bytes"]) for row in year_rows)
            year_complete_before = sum(
                int(row["expected_size_bytes"]) for row in year_rows if row["download_status"] == "COMPLETE"
            )
            year_totals = DownloadTotals(year_expected, year_complete_before)
            print(f"\nStarting {year}: {len(year_rows)} files, {format_size(year_expected)}")
            for file_number, row in enumerate(year_rows, start=1):
                expected_size = int(row["expected_size_bytes"])

                def make_progress(existing_size: int, *, _row: dict[str, object] = row) -> DownloadProgress:
                    return DownloadProgress(
                        year=year,
                        file_number=file_number,
                        file_count=len(year_rows),
                        filename=str(_row["filename"]),
                        file_expected=expected_size,
                        file_existing=existing_size,
                        year_totals=year_totals,
                        overall_totals=overall_totals,
                    )

                download_row(session, row, make_progress)
                write_manifest(all_rows)
            validations[year] = validate_year(year, year_rows)
            write_summary(validations, all_rows)
    except AuthenticationError:
        write_manifest(all_rows)
        write_summary(validations, all_rows)
        print_authentication_failure()
        return 2
    finally:
        session.close()

    write_manifest(all_rows)
    write_summary(validations, all_rows)
    if all(bool(validations[year]["passed"]) for year in YEARS):
        print("\n" + "=" * 60)
        print("NIMROD 2021-2025 RAW DOWNLOAD COMPLETE")
        print("RAW ARCHIVES VERIFIED")
        print("READY FOR DYNAMIC RAINFALL PROCESSING")
        print("=" * 60)
        return 0
    return 1


def print_authentication_failure() -> None:
    print("CEDA TOKEN EXPIRED OR AUTHENTICATION FAILED.", file=sys.stderr)
    print("Replace ceda_token.txt with a new token and rerun the downloader.", file=sys.stderr)
    print("Existing completed files will be preserved and skipped.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
