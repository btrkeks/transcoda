#!/usr/bin/env python3
"""Inventory and download KernScores Humdrum files.

The KernScores site exposes browsable collection pages plus CGI download links.
This module keeps the ingest deliberately resumable and auditable:

1. Crawl browse pages into an inventory JSONL.
2. Download inventory rows with hashes and checkpoint rows.
3. Materialize downloaded files into the standard dataset pipeline tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import fire

BASE_URL = "https://kern.humdrum.org"
DEFAULT_OUTPUT_DIR = Path("data/external/kernscores-direct")
DEFAULT_MATERIALIZED_DIR = Path("data/interim/train/kernscores-direct/1_kern_conversions")
DEFAULT_USER_AGENT = (
    "TranscodaKernScoresIngest/0.1 "
    "(research dataset; permission confirmed with KernScores maintainer)"
)
SUCCESS_STATUSES = {"success", "success_existing"}


@dataclass(frozen=True)
class KernScoresLink:
    """A single downloadable KernScores Humdrum file."""

    source_path: str
    collection: str
    filename: str
    download_url: str
    discovered_from: str


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc


def normalize_collection(value: str | None) -> str:
    """Normalize a KernScores collection path to a safe relative POSIX path."""

    if value is None:
        return ""
    value = value.strip()
    if not value or value == "/":
        return ""

    value = value.replace("\\", "/").strip("/")
    value = posixpath.normpath(value)
    if value in {"", "."}:
        return ""
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe KernScores collection path: {value!r}")
    return "/".join(parts)


def safe_source_path(collection: str, filename: str) -> str:
    """Return a safe local relative path for a KernScores file."""

    collection = normalize_collection(collection)
    filename = filename.replace("\\", "/").strip("/")
    if not filename:
        raise ValueError("KernScores file link has an empty filename")

    raw_parts = []
    if collection:
        raw_parts.extend(collection.split("/"))
    raw_parts.extend(filename.split("/"))

    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Unsafe KernScores file path: {collection!r}, {filename!r}")

    return "/".join(raw_parts)


def _path_from_source_path(base_dir: Path, source_path: str) -> Path:
    parts = source_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe source path in manifest: {source_path!r}")
    return base_dir.joinpath(*parts)


def make_browse_url(collection: str) -> str:
    collection = normalize_collection(collection)
    location = "/" if not collection else f"/{collection}"
    return f"{BASE_URL}/browse?{urlencode({'l': location})}"


def make_download_url(collection: str, filename: str) -> str:
    collection = normalize_collection(collection)
    query = urlencode({"file": filename, "l": collection, "format": "kern"})
    return f"{BASE_URL}/cgi-bin/ksdata?{query}"


def _extract_hrefs(html: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(html)
    return parser.hrefs


def extract_page_links(html: str, page_url: str) -> tuple[set[str], list[KernScoresLink]]:
    """Extract child browse locations and downloadable kern links from a page."""

    child_locations: set[str] = set()
    kern_links: list[KernScoresLink] = []

    for href in _extract_hrefs(html):
        absolute_url = urljoin(page_url, href)
        parsed = urlparse(absolute_url)
        if parsed.netloc and parsed.netloc != "kern.humdrum.org":
            continue

        query = parse_qs(parsed.query, keep_blank_values=True)
        path = parsed.path.rstrip("/")

        if path in {"/browse", "/cgi-bin/browse"} and query.get("l"):
            try:
                child_locations.add(normalize_collection(query["l"][-1]))
            except ValueError:
                continue
            continue

        if path != "/cgi-bin/ksdata":
            continue

        formats = query.get("format", []) + query.get("f", [])
        if "kern" not in formats:
            continue
        filenames = query.get("file")
        locations = query.get("l") or query.get("location")
        if not filenames or not filenames[-1].strip() or not locations:
            continue

        collection = normalize_collection(locations[-1])
        filename = filenames[-1]
        source_path = safe_source_path(collection, filename)
        kern_links.append(
            KernScoresLink(
                source_path=source_path,
                collection=collection,
                filename=filename,
                download_url=make_download_url(collection, filename),
                discovered_from=page_url,
            )
        )

    return child_locations, kern_links


def _fetch_url(url: str, user_agent: str, timeout_seconds: float) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        return int(status), response.read()


class _RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._last_request_at = 0.0

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return
        now = time.monotonic()
        sleep_for = self.delay_seconds - (now - self._last_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request_at = time.monotonic()


def _parse_start_locations(start_locations: str | Iterable[str]) -> list[str]:
    if isinstance(start_locations, str):
        raw_locations = [item.strip() for item in start_locations.split(",")]
    else:
        raw_locations = [str(item).strip() for item in start_locations]
    locations = [normalize_collection(item) for item in raw_locations if item]
    return locations or [""]


def _inventory_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    inventory_dir = output_path / "inventory"
    return (
        inventory_dir / "kernscores_inventory.jsonl",
        inventory_dir / "crawl_pages.jsonl",
        inventory_dir / "inventory_summary.json",
    )


def _load_inventory_resume_state(
    inventory_jsonl: Path,
    page_log_jsonl: Path,
    start_locations: Iterable[str],
) -> tuple[set[str], set[str], deque[str]]:
    seen_files = {
        str(row["source_path"]) for row in _read_jsonl(inventory_jsonl) if row.get("source_path")
    }
    visited_locations: set[str] = set()
    queued_locations: set[str] = {normalize_collection(item) for item in start_locations}

    for row in _read_jsonl(page_log_jsonl):
        location = normalize_collection(row.get("location", ""))
        if row.get("status") == "success":
            visited_locations.add(location)
            for child in row.get("child_locations", []):
                child_location = normalize_collection(child)
                if child_location not in visited_locations:
                    queued_locations.add(child_location)

    queued = deque(sorted(queued_locations - visited_locations))
    return seen_files, visited_locations, queued


def inventory(
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    start_locations: str = "/",
    max_pages: int | None = None,
    delay_seconds: float = 0.25,
    timeout_seconds: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    resume: bool = True,
    quiet: bool = False,
) -> str:
    """Crawl KernScores browse pages and write a downloadable kern inventory.

    Args:
        output_dir: Base directory for inventory and later downloads.
        start_locations: Comma-separated KernScores locations to crawl.
        max_pages: Optional cap for this run, useful for smoke tests.
        delay_seconds: Delay between browse-page requests.
        timeout_seconds: Per-request timeout.
        user_agent: User-Agent sent to KernScores.
        resume: Continue from existing crawl logs instead of starting over.
        quiet: Suppress progress output.

    Returns:
        Path to the inventory JSONL.
    """

    output_path = Path(output_dir)
    inventory_jsonl, page_log_jsonl, summary_json = _inventory_paths(output_path)
    starts = _parse_start_locations(start_locations)

    if not resume:
        for path in (inventory_jsonl, page_log_jsonl, summary_json):
            path.unlink(missing_ok=True)

    seen_files, visited_locations, queue = _load_inventory_resume_state(
        inventory_jsonl, page_log_jsonl, starts
    )
    if not queue:
        queue = deque(sorted(set(starts) - visited_locations))

    limiter = _RateLimiter(delay_seconds)
    attempted_this_run = 0
    succeeded_this_run = 0
    new_files_this_run = 0

    while queue:
        if max_pages is not None and attempted_this_run >= max_pages:
            break

        location = normalize_collection(queue.popleft())
        if location in visited_locations:
            continue

        browse_url = make_browse_url(location)
        attempted_this_run += 1

        try:
            limiter.wait()
            status_code, body = _fetch_url(browse_url, user_agent, timeout_seconds)
            html = body.decode("utf-8", errors="replace")
            child_locations, links = extract_page_links(html, browse_url)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            _write_jsonl(
                page_log_jsonl,
                {
                    "timestamp": _utc_now(),
                    "status": "error",
                    "location": location,
                    "browse_url": browse_url,
                    "error": repr(exc),
                },
            )
            if not quiet:
                print(f"error crawling {browse_url}: {exc}", file=sys.stderr)
            continue

        page_new_files = 0
        for link in links:
            if link.source_path in seen_files:
                continue
            _write_jsonl(
                inventory_jsonl,
                {
                    **asdict(link),
                    "source": "kernscores",
                    "kind": "kern",
                    "discovered_at": _utc_now(),
                },
            )
            seen_files.add(link.source_path)
            page_new_files += 1

        child_locations.discard(location)
        for child_location in sorted(child_locations):
            if child_location not in visited_locations:
                queue.append(child_location)

        visited_locations.add(location)
        succeeded_this_run += 1
        new_files_this_run += page_new_files
        _write_jsonl(
            page_log_jsonl,
            {
                "timestamp": _utc_now(),
                "status": "success",
                "location": location,
                "browse_url": browse_url,
                "http_status": status_code,
                "bytes": len(body),
                "child_locations": sorted(child_locations),
                "kern_links": len(links),
                "new_kern_links": page_new_files,
            },
        )

        if not quiet and succeeded_this_run % 25 == 0:
            print(
                "crawled "
                f"{succeeded_this_run} pages this run; "
                f"{len(seen_files)} unique kern links total",
                file=sys.stderr,
            )

    summary = {
        "timestamp": _utc_now(),
        "output_dir": str(output_path),
        "inventory_jsonl": str(inventory_jsonl),
        "page_log_jsonl": str(page_log_jsonl),
        "start_locations": starts,
        "attempted_pages_this_run": attempted_this_run,
        "succeeded_pages_this_run": succeeded_this_run,
        "visited_pages_total": len(visited_locations),
        "new_kern_links_this_run": new_files_this_run,
        "unique_kern_links_total": len(seen_files),
        "queued_locations_remaining": len(set(queue) - visited_locations),
        "max_pages": max_pages,
        "delay_seconds": delay_seconds,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not quiet:
        print(
            "Inventory complete: "
            f"{summary['unique_kern_links_total']} unique kern links, "
            f"{summary['visited_pages_total']} pages visited. "
            f"Wrote {inventory_jsonl}",
            file=sys.stderr,
        )

    return str(inventory_jsonl)


def _download_manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "download_manifest.jsonl"


def _parse_extensions(file_extensions: str | None) -> set[str] | None:
    if file_extensions is None:
        return None
    value = file_extensions.strip()
    if not value or value.lower() in {"all", "*"}:
        return None
    extensions: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions or None


def _source_path_extension(source_path: str) -> str:
    name = source_path.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return f".{name.rsplit('.', 1)[-1].lower()}"


def _latest_successes(manifest_jsonl: Path) -> dict[str, dict]:
    successes: dict[str, dict] = {}
    for row in _read_jsonl(manifest_jsonl):
        source_path = row.get("source_path")
        if source_path and row.get("status") in SUCCESS_STATUSES:
            successes[str(source_path)] = row
    return successes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_once(
    url: str,
    user_agent: str,
    timeout_seconds: float,
    retries: int,
    delay_seconds: float,
) -> tuple[int, bytes]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _fetch_url(url, user_agent, timeout_seconds)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def download(
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    inventory_jsonl: str | None = None,
    limit: int | None = None,
    file_extensions: str | None = ".krn",
    delay_seconds: float = 0.5,
    timeout_seconds: float = 60.0,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    overwrite: bool = False,
    quiet: bool = False,
) -> str:
    """Download kern files from an inventory JSONL.

    Args:
        output_dir: Base directory containing inventory and receiving files.
        inventory_jsonl: Inventory path. Defaults to output_dir inventory path.
        limit: Optional cap for new rows processed in this run.
        file_extensions: Comma-separated source filename extensions to download,
            or "all" to download every inventory row.
        delay_seconds: Delay between download requests.
        timeout_seconds: Per-request timeout.
        user_agent: User-Agent sent to KernScores.
        retries: Retry count after the initial attempt.
        overwrite: Re-download files even when a successful file exists.
        quiet: Suppress progress output.

    Returns:
        Path to the download manifest JSONL.
    """

    output_path = Path(output_dir)
    if inventory_jsonl is None:
        inventory_path, _, _ = _inventory_paths(output_path)
    else:
        inventory_path = Path(inventory_jsonl)
    manifest_path = _download_manifest_path(output_path)
    files_dir = output_path / "files"
    allowed_extensions = _parse_extensions(file_extensions)

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory JSONL not found: {inventory_path}")

    successful_rows = _latest_successes(manifest_path)
    limiter = _RateLimiter(delay_seconds)

    processed_this_run = 0
    downloaded_this_run = 0
    existing_this_run = 0
    failed_this_run = 0
    skipped_extension_this_run = 0

    for row in _read_jsonl(inventory_path):
        source_path = safe_source_path(row.get("collection", ""), row.get("filename", ""))
        local_path = _path_from_source_path(files_dir, source_path)

        if (
            allowed_extensions is not None
            and _source_path_extension(source_path) not in allowed_extensions
        ):
            skipped_extension_this_run += 1
            continue

        if source_path in successful_rows and local_path.exists() and not overwrite:
            continue

        if limit is not None and processed_this_run >= limit:
            break

        processed_this_run += 1
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not overwrite:
            size = local_path.stat().st_size
            manifest_row = {
                "timestamp": _utc_now(),
                "status": "success_existing",
                "source_path": source_path,
                "local_path": str(local_path),
                "download_url": row.get("download_url"),
                "bytes": size,
                "sha256": _sha256_file(local_path),
                "contains_kern_spine": b"**kern" in local_path.read_bytes(),
            }
            _write_jsonl(manifest_path, manifest_row)
            successful_rows[source_path] = manifest_row
            existing_this_run += 1
            continue

        try:
            limiter.wait()
            http_status, body = _download_once(
                str(row.get("download_url")),
                user_agent,
                timeout_seconds,
                retries,
                delay_seconds,
            )
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            tmp_path.write_bytes(body)
            os.replace(tmp_path, local_path)

            manifest_row = {
                "timestamp": _utc_now(),
                "status": "success",
                "source": "kernscores",
                "kind": "kern",
                "source_path": source_path,
                "collection": row.get("collection"),
                "filename": row.get("filename"),
                "local_path": str(local_path),
                "download_url": row.get("download_url"),
                "http_status": http_status,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "contains_kern_spine": b"**kern" in body,
            }
            _write_jsonl(manifest_path, manifest_row)
            successful_rows[source_path] = manifest_row
            downloaded_this_run += 1
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            failed_this_run += 1
            _write_jsonl(
                manifest_path,
                {
                    "timestamp": _utc_now(),
                    "status": "error",
                    "source": "kernscores",
                    "kind": "kern",
                    "source_path": source_path,
                    "local_path": str(local_path),
                    "download_url": row.get("download_url"),
                    "error": repr(exc),
                },
            )
            if not quiet:
                print(f"error downloading {source_path}: {exc}", file=sys.stderr)

        if not quiet and processed_this_run % 100 == 0:
            print(
                f"processed {processed_this_run} rows this run; {downloaded_this_run} downloaded",
                file=sys.stderr,
            )

    summary_path = output_path / "download_summary.json"
    summary = {
        "timestamp": _utc_now(),
        "output_dir": str(output_path),
        "inventory_jsonl": str(inventory_path),
        "download_manifest_jsonl": str(manifest_path),
        "processed_this_run": processed_this_run,
        "downloaded_this_run": downloaded_this_run,
        "existing_this_run": existing_this_run,
        "failed_this_run": failed_this_run,
        "skipped_extension_this_run": skipped_extension_this_run,
        "successful_downloads_total": len(successful_rows),
        "limit": limit,
        "file_extensions": file_extensions,
        "delay_seconds": delay_seconds,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not quiet:
        print(
            "Download complete: "
            f"{downloaded_this_run} downloaded, "
            f"{existing_this_run} existing, "
            f"{failed_this_run} failed this run. "
            f"Wrote {manifest_path}",
            file=sys.stderr,
        )

    return str(manifest_path)


def _same_file(src_path: Path, dst_path: Path) -> bool:
    try:
        return src_path.samefile(dst_path)
    except (FileNotFoundError, OSError):
        return False


def _replace_with_copy(src_path: Path, dst_path: Path) -> None:
    if _same_file(src_path, dst_path):
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    shutil.copy2(src_path, dst_path)


def _replace_with_hardlink(src_path: Path, dst_path: Path) -> None:
    if _same_file(src_path, dst_path):
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.hardlink_to(src_path)


def _materialize_file(src_path: Path, dst_path: Path, link_mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        _replace_with_copy(src_path, dst_path)
        return
    if link_mode == "hardlink":
        _replace_with_hardlink(src_path, dst_path)
        return
    if link_mode != "auto":
        raise ValueError(f"Invalid link_mode {link_mode!r}. Expected one of: auto, copy, hardlink")
    try:
        _replace_with_hardlink(src_path, dst_path)
    except OSError:
        _replace_with_copy(src_path, dst_path)


def materialize(
    download_dir: str = str(DEFAULT_OUTPUT_DIR),
    output_dir: str = str(DEFAULT_MATERIALIZED_DIR),
    manifest_jsonl: str | None = None,
    require_kern_spine: bool = True,
    require_krn_extension: bool = True,
    link_mode: str = "auto",
    quiet: bool = False,
) -> str:
    """Materialize successful KernScores downloads into `1_kern_conversions/`.

    Args:
        download_dir: Base directory containing downloaded files.
        output_dir: Destination pipeline directory.
        manifest_jsonl: Download manifest path. Defaults to download_dir manifest.
        require_kern_spine: Skip rows whose download did not contain ``**kern``.
        require_krn_extension: Skip rows whose source path is not a ``.krn`` file.
        link_mode: auto, copy, or hardlink.
        quiet: Suppress progress output.

    Returns:
        Destination directory.
    """

    download_path = Path(download_dir)
    manifest_path = (
        Path(manifest_jsonl) if manifest_jsonl else _download_manifest_path(download_path)
    )
    output_path = Path(output_dir)

    if link_mode not in {"auto", "copy", "hardlink"}:
        raise ValueError(f"Invalid link_mode {link_mode!r}. Expected one of: auto, copy, hardlink")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Download manifest not found: {manifest_path}")

    latest_rows = _latest_successes(manifest_path)
    materialized = 0
    skipped_missing = 0
    skipped_non_kern = 0
    skipped_extension = 0

    for source_path, row in sorted(latest_rows.items()):
        if require_krn_extension and _source_path_extension(source_path) != ".krn":
            skipped_extension += 1
            continue
        if require_kern_spine and not row.get("contains_kern_spine", False):
            skipped_non_kern += 1
            continue
        src_path = Path(row.get("local_path", ""))
        if not src_path.exists():
            skipped_missing += 1
            continue
        dst_path = _path_from_source_path(output_path, source_path)
        _materialize_file(src_path, dst_path, link_mode)
        materialized += 1

    summary_path = output_path / "kernscores_materialization_summary.json"
    summary = {
        "timestamp": _utc_now(),
        "download_dir": str(download_path),
        "manifest_jsonl": str(manifest_path),
        "output_dir": str(output_path),
        "materialized": materialized,
        "skipped_missing": skipped_missing,
        "skipped_non_kern": skipped_non_kern,
        "skipped_extension": skipped_extension,
        "require_kern_spine": require_kern_spine,
        "require_krn_extension": require_krn_extension,
        "link_mode": link_mode,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not quiet:
        print(
            "Materialization complete: "
            f"{materialized} files to {output_path}; "
            f"{skipped_non_kern} non-kern rows skipped.",
            file=sys.stderr,
        )

    return str(output_path)


def scrape(
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    start_locations: str = "/",
    max_pages: int | None = None,
    download_limit: int | None = None,
    materialize_output_dir: str = str(DEFAULT_MATERIALIZED_DIR),
    delay_seconds: float = 0.5,
    timeout_seconds: float = 60.0,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    materialize_after_download: bool = True,
    quiet: bool = False,
) -> dict[str, str]:
    """Run inventory, download, and optional materialization in one command."""

    inventory_jsonl = inventory(
        output_dir=output_dir,
        start_locations=start_locations,
        max_pages=max_pages,
        delay_seconds=delay_seconds,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
        resume=True,
        quiet=quiet,
    )
    manifest_jsonl = download(
        output_dir=output_dir,
        inventory_jsonl=inventory_jsonl,
        limit=download_limit,
        delay_seconds=delay_seconds,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
        retries=retries,
        quiet=quiet,
    )
    result = {
        "inventory_jsonl": inventory_jsonl,
        "download_manifest_jsonl": manifest_jsonl,
    }
    if materialize_after_download:
        result["materialized_dir"] = materialize(
            download_dir=output_dir,
            output_dir=materialize_output_dir,
            manifest_jsonl=manifest_jsonl,
            quiet=quiet,
        )
    return result


def main() -> None:
    fire.Fire(
        {
            "inventory": inventory,
            "download": download,
            "materialize": materialize,
            "scrape": scrape,
        }
    )


if __name__ == "__main__":
    main()
