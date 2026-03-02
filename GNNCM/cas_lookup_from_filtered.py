import argparse
import csv
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILTERED_TXT = str(PROJECT_ROOT / "results" / "similarity" / "Filtered_results_0.95_1_million.txt")
DEFAULT_OUTPUT_CSV = str(PROJECT_ROOT / "results" / "similarity")

CAS_REGEX = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

class RateLimiter:
    def __init__(self, qps: float) -> None:
        self._qps = max(float(qps), 0.0)
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self._qps <= 0:
            return
        interval = 1.0 / self._qps
        with self._lock:
            now = time.perf_counter()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.perf_counter()
            self._next_time = now + interval

def load_cache(cache_path: Optional[str]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for smiles, value in payload.items():
        if isinstance(value, dict):
            cache[smiles] = (value.get("cas"), value.get("cid"))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            cache[smiles] = (value[0], value[1])
    return cache

def save_cache(cache_path: Optional[str], cache: Dict[str, Tuple[Optional[str], Optional[str]]]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {smiles: {"cas": cas, "cid": cid} for smiles, (cas, cid) in cache.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

@ dataclass

class MatchRecord:
    anchor_id: str
    anchor_smiles: str
    library_id: str
    library_smiles: str
    similarity: float

def parse_filtered_results(path: str) -> List[MatchRecord]:
    records: List[MatchRecord] = []
    anchor_id: Optional[str] = None
    anchor_smiles: Optional[str] = None

    anchor_pattern = re.compile(r"^Anchor molecule:\s*(.+?)\s*\|\s*(.+)$")
    match_pattern = re.compile(
        r"^\s*\d+\.\s*Library molecule:\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*Similarity:\s*([0-9.]+)"
    )

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            anchor_match = anchor_pattern.match (stripped)
            if anchor_match:
                anchor_id = anchor_match.group(1).strip()
                anchor_smiles = anchor_match.group(2).strip()
                continue

            match_match = match_pattern.match (stripped)
            if match_match and anchor_id and anchor_smiles:
                library_id = match_match.group(1).strip()
                library_smiles = match_match.group(2).strip()
                similarity = float(match_match.group(3))
                records.append(
                    MatchRecord(
                    anchor_id = anchor_id,
                    anchor_smiles = anchor_smiles,
                    library_id = library_id,
                    library_smiles = library_smiles,
                    similarity = similarity,
                )
                )

    return records

def find_cid_for_smiles(session: requests.Session, smiles: str, timeout: int=20) -> Optional[str]:
    url = f"{PUBCHEM_BASE}/compound/smiles/{requests.utils.quote(smiles)}/cids/TXT"
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        return None
    cid = response.text.strip().splitlines()[0]
    return cid if cid else None

def fetch_synonyms_by_cid(session: requests.Session, cid: str, timeout: int=20) -> List[str]:
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        return []
    payload = response.json()
    info_list = payload.get("InformationList", {}).get("Information", [])
    synonyms: List[str] = []
    for info in info_list:
        synonyms.extend(info.get("Synonym", []))
    return synonyms

def extract_cas_from_synonyms(synonyms: List[str]) -> Optional[str]:
    for synonym in synonyms:
        match = CAS_REGEX.search(synonym)
        if match:
            return match.group(0)
    return None

def resolve_cas_for_smiles(
    session: requests.Session,
    smiles: str,
    retries: int = 3,
    pause: float = 0.5,
    timeout: int = 20,
    limiter: Optional[RateLimiter] = None,
) -> Tuple[Optional[str], Optional[str]]:
    for attempt in range(1, retries + 1):
        try:
            if limiter:
                limiter.wait()
            cid = find_cid_for_smiles(session, smiles, timeout=timeout)
            if not cid:
                return None, None
            if limiter:
                limiter.wait()
            synonyms = fetch_synonyms_by_cid(session, cid, timeout=timeout)
            cas_number = extract_cas_from_synonyms(synonyms)
            return cas_number, cid
        except (requests.RequestException, ValueError, KeyError):
            if attempt == retries:
                return None, None
            backoff = pause * (2**(attempt - 1))
            time.sleep(backoff)
    return None, None

def resolve_many_smiles(
    smiles_list: List[str],
    retries: int,
    pause: float,
    timeout: int,
    workers: int,
    qps: float,
    cache: Dict[str, Tuple[Optional[str], Optional[str]]],
    cache_path: Optional[str],
    resume: bool,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    limiter = RateLimiter(qps)
    lock = threading.Lock()

    def worker(smiles: str) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
        if resume:
            existing = cache.get(smiles)
            if existing is not None:
                return smiles, existing

        with requests.Session() as session:
            cas_number, cid = resolve_cas_for_smiles(
                session,
                smiles,
                retries = retries,
                pause = pause,
                timeout = timeout,
                limiter = limiter,
            )
        with lock:
            cache[smiles] = (cas_number, cid)
            save_cache(cache_path, cache)
        return smiles, (cas_number, cid)

    if workers <= 1:
        for smiles in smiles_list:
            worker(smiles)
        return cache

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, smiles): smiles for smiles in smiles_list}
        done = 0
        total = len(smiles_list)
        for future in as_completed(futures):
            _ = future.result()
            done += 1
            if done % 25 == 0 or done == total:
                print(f"Completed {done} / {total} queries")
    return cache

def build_output_path(filtered_path: str, output_path: Optional[str]) -> str:
    if output_path:
        if output_path.lower().endswith(".csv"):
            return output_path
        if output_path.endswith(("/", "\\")):
            base = os.path.splitext(os.path.basename(filtered_path))[0]
            return os.path.join(output_path, f"{base}_cas.csv")
        if os.path.isdir(output_path):
            base = os.path.splitext(os.path.basename(filtered_path))[0]
            return os.path.join(output_path, f"{base}_cas.csv")
        return f"{output_path}.csv"

    directory = os.path.dirname(filtered_path)
    base = os.path.splitext(os.path.basename(filtered_path))[0]
    return os.path.join(directory, f"{base}_cas.csv")

def write_results(records: List[MatchRecord], cas_map: Dict[str, Tuple[Optional[str], Optional[str]]], output_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    header = [
        "anchor_id",
        "anchor_smiles",
        "library_id",
        "library_smiles",
        "similarity",
        "pubchem_cid",
        "cas_number",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in records:
            cas_number, cid = cas_map.get(record.library_smiles, (None, None))
            writer.writerow(
                [
                record.anchor_id,
                record.anchor_smiles,
                record.library_id,
                record.library_smiles,
                f"{record.similarity:  .4f}",
                cid or "",
                cas_number or "",
            ]
            )

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Look up CAS numbers for library molecules listed in a filtered similarity report."
    )
    parser.add_argument(
        "--filtered_txt",
        default = DEFAULT_FILTERED_TXT,
        help = "Filtered similarity report generated by smiles_similarity_search",
    )
    parser.add_argument(
        "--output_csv",
        default = DEFAULT_OUTPUT_CSV,
        help = "Output CSV path; by default a file is created next to the input report",
    )
    parser.add_argument(
        "--pause",
        type = float,
        default = 0.5,
        help = "Wait time in seconds before retrying a failed request",
    )
    parser.add_argument(
        "--retries",
        type = int,
        default = 3,
        help = "Maximum retry count for a single SMILES query",
    )
    parser.add_argument(
        "--timeout",
        type = int,
        default = 20,
        help = "HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--workers",
        type = int,
        default = 6,
        help = "Number of worker threads to use; values around 2-6 are usually safer",
    )
    parser.add_argument(
        "--qps",
        type = float,
        default = 6.0,
        help = "Global request rate limit in queries per second; use 0 to disable the limit",
    )
    parser.add_argument(
        "--cache_path",
        default = None,
        help = "Optional JSON cache file path for resume support and request deduplication",
    )
    parser.add_argument(
        "--resume",
        action = "store_true",
        help = "Resume from cache; skip requests for SMILES values that already have cached results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.filtered_txt):
        raise FileNotFoundError(f"Filtered report file not found: {args.filtered_txt}")

    print("Parsing filtered report...")
    records = parse_filtered_results(args.filtered_txt)
    if not records:
        print("No matching records were parsed. Exiting.")
        return
    print(f"Parsed {len(records)} matching records.")

    unique_smiles = {record.library_smiles for record in records}
    print(f"Unique library molecules to query: {len(unique_smiles)}")

    cas_map: Dict[str, Tuple[Optional[str], Optional[str]]] = load_cache(args.cache_path)
    if args.resume and cas_map:
        print(f"Loaded cached records: {len(cas_map)}")

    smiles_list = sorted(unique_smiles)
    pending_smiles = smiles_list
    if args.resume and cas_map:
        pending_smiles = [s for s in smiles_list if s not in cas_map]
        print(f"Resume mode enabled: querying {len(pending_smiles)} / {len(smiles_list)} molecules")

    if pending_smiles:
        resolve_many_smiles(
            pending_smiles,
            retries = args.retries,
            pause = args.pause,
            timeout = args.timeout,
            workers = args.workers,
            qps = args.qps,
            cache = cas_map,
            cache_path = args.cache_path,
            resume = args.resume,
        )

    output_csv = build_output_path(args.filtered_txt, args.output_csv)
    print(f"Writing results to {output_csv} ...")
    write_results(records, cas_map, output_csv)
    print("Done.")

if __name__ == "__main__":
    main()
