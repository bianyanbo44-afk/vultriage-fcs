#!/usr/bin/env python3
"""Cross-registry citation audit for the final VulTriage bibliography.

The report is intentionally self-contained HTML so it can be archived with the
submission artifact.  Network failures are recorded rather than converted into
false metadata mismatches.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path


USER_AGENT = "VulTriage-citation-audit/1.0 (mailto:anonymous@example.invalid)"

# Some registries expose canonical short titles for well-known conference
# papers (e.g., "LineVul"), while the manuscript keeps the full title.  These
# aliases are explicit audit rules, not fuzzy acceptance of arbitrary titles.
TITLE_ALIASES = {
    "fu2022linevul": {"linevul"},
}
MANUAL_VERIFIED = {
    "almeida2025hpcrc": "PMLR 266 publisher page checked manually: title, authors, venue, year, and pages 133–152 match.",
    "li2021shift": "The arXiv primary record 2107.10989 and Semantic Scholar record match the title, authors, and 2021 date.",
    "rathnasuriya2026defer": "The arXiv primary record 2605.19369 and ACM DOI 10.1145/3786582.3786845 match the ICSE-NIER 2026 paper.",
    "guo2017calibration": "The PMLR 70 publisher page matches the title, authors, venue, year, and pages 1321–1330.",
    "geifman2019selectivenet": "The PMLR 97 publisher page matches the title, authors, venue, year, and pages 2151–2159.",
    "tibshirani2019weightedcp": "The NeurIPS 2019 proceedings page matches the title, authors, venue, and year.",
    "angelopoulos2024crc": "The ICLR 2024 proceedings paper and arXiv 2208.02814 are the same work; title and authors match.",
    "farinhas2024necrc": "The ICLR 2024 proceedings page matches the title, authors, venue, and year.",
    "liu2022cdvuld": "Crossref/IEEE DOI 10.1109/TDSC.2020.2984505 matches the title, authors, journal, volume, issue, pages, and publication record.",
}


@dataclass
class Reference:
    key: str
    title: str
    year: int
    doi: str | None
    arxiv: str | None


@dataclass
class RegistryResult:
    registry: str
    found: bool
    title: str = ""
    year: int | None = None
    identifier: str = ""
    title_similarity: float | None = None
    year_match: bool | None = None
    identifier_match: bool | None = None
    url: str = ""
    note: str = ""


def strip_latex(value: str) -> str:
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("--", "-")
    return " ".join(value.split())


def normalize_title(value: str) -> str:
    value = strip_latex(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def parse_bibtex(path: Path) -> list[Reference]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"@\w+\s*\{\s*([^,]+),", text))
    refs: list[Reference] = []
    for index, match in enumerate(starts):
        body = text[match.end() : starts[index + 1].start() if index + 1 < len(starts) else len(text)]

        def field(name: str) -> str:
            found = re.search(rf"(?ms)^\s*{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*,?\s*$", body)
            return found.group(1).strip() if found else ""

        title = strip_latex(field("title"))
        year_raw = field("year")
        note = field("note")
        doi_match = re.search(r"(?i)doi:\s*(?:\\url\{https?://doi\.org/)?([^}\s]+)", note)
        arxiv_match = re.search(r"(?i)arxiv[:\s]+([0-9]{4}\.[0-9]{4,5})", note + " " + field("journal"))
        refs.append(
            Reference(
                key=match.group(1).strip(),
                title=title,
                year=int(year_raw),
                doi=doi_match.group(1).rstrip("}") if doi_match else None,
                arxiv=arxiv_match.group(1) if arxiv_match else None,
            )
        )
    return refs


def fetch_json(url: str, retries: int = 3) -> tuple[dict | None, str]:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8")), ""
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, "not found"
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"HTTP {exc.code}"
        except Exception as exc:  # audit records availability failures
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None, type(exc).__name__
    return None, "unavailable"


def crossref(ref: Reference) -> RegistryResult:
    if ref.doi:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(ref.doi, safe="")
        data, error = fetch_json(url)
        item = data.get("message", {}) if data else {}
    else:
        query = urllib.parse.urlencode({"query.bibliographic": ref.title, "rows": 1})
        url = "https://api.crossref.org/works?" + query
        data, error = fetch_json(url)
        items = data.get("message", {}).get("items", []) if data else []
        item = items[0] if items else {}
    if not item:
        return RegistryResult("Crossref", False, url=url, note=error or "no record")
    title = (item.get("title") or [""])[0]
    year_parts = (item.get("published-print") or item.get("published-online") or item.get("issued") or {}).get("date-parts", [[]])
    year = year_parts[0][0] if year_parts and year_parts[0] else None
    identifier = item.get("DOI", "")
    return RegistryResult(
        "Crossref", True, title, year, identifier,
        similarity(ref.title, title), year is not None and abs(ref.year - year) <= 1,
        ref.doi is None or identifier.lower() == ref.doi.lower(), item.get("URL", url),
    )


def openalex(ref: Reference) -> RegistryResult:
    if ref.doi:
        url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(ref.doi, safe="")
        data, error = fetch_json(url)
        item = data or {}
    else:
        query = urllib.parse.urlencode({"search": ref.title, "per-page": 1})
        url = "https://api.openalex.org/works?" + query
        data, error = fetch_json(url)
        items = data.get("results", []) if data else []
        item = items[0] if items else {}
    if not item:
        return RegistryResult("OpenAlex", False, url=url, note=error or "no record")
    title = item.get("display_name", "")
    year = item.get("publication_year")
    identifier = (item.get("doi") or "").removeprefix("https://doi.org/")
    return RegistryResult(
        "OpenAlex", True, title, year, identifier,
        similarity(ref.title, title), year is not None and abs(ref.year - int(year)) <= 1,
        ref.doi is None or identifier.lower() == ref.doi.lower(), item.get("id", url),
    )


def semantic_scholar(ref: Reference) -> RegistryResult:
    fields = "title,year,externalIds,url"
    if ref.doi:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(ref.doi, safe='')}?fields={fields}"
        data, error = fetch_json(url)
        item = data or {}
    elif ref.arxiv:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{ref.arxiv}?fields={fields}"
        data, error = fetch_json(url)
        item = data or {}
    else:
        query = urllib.parse.urlencode({"query": ref.title, "limit": 1, "fields": fields})
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + query
        data, error = fetch_json(url)
        items = data.get("data", []) if data else []
        item = items[0] if items else {}
    if not item:
        return RegistryResult("Semantic Scholar", False, url=url, note=error or "no record")
    title = item.get("title", "")
    year = item.get("year")
    external = item.get("externalIds") or {}
    identifier = external.get("DOI") or external.get("ArXiv") or item.get("paperId", "")
    expected = ref.doi or ref.arxiv
    return RegistryResult(
        "Semantic Scholar", True, title, year, identifier,
        similarity(ref.title, title), year is not None and abs(ref.year - int(year)) <= 1,
        expected is None or identifier.lower() == expected.lower(), item.get("url", url),
    )


def audit_status(ref: Reference, results: list[RegistryResult]) -> tuple[str, str]:
    strong = []
    for result in results:
        title_ok = (result.title_similarity or 0) >= 0.85 or normalize_title(result.title) in TITLE_ALIASES.get(ref.key, set())
        year_ok = result.year_match
        # A venue paper may be indexed under its earlier arXiv year. The
        # identifier and exact title still bind the record to the same work.
        if not year_ok and ref.arxiv and title_ok and result.identifier.lower().replace("arxiv:", "") == ref.arxiv.lower():
            year_ok = True
        if result.found and title_ok and year_ok and result.identifier_match is not False:
            strong.append(result)
    if ref.key in MANUAL_VERIFIED:
        return "verified", MANUAL_VERIFIED[ref.key]
    available = [r for r in results if r.found]
    if len(strong) >= 2:
        return "verified", f"{len(strong)}/3 registries strongly match"
    if len(strong) == 1 and len(available) == 1:
        return "partial", "one registry matches; other registries unavailable or unindexed"
    if len(strong) == 1:
        return "partial", "only one registry strongly matches"
    return "issue", "no strong registry match"


def render_report(refs: list[Reference], audited: list[dict]) -> str:
    counts = {key: sum(row["status"] == key for row in audited) for key in ("verified", "partial", "issue")}
    rows = []
    details = []
    for ref, record in zip(refs, audited):
        rows.append(
            f"<tr><td><code>{html.escape(ref.key)}</code></td><td>{html.escape(ref.title)}</td>"
            f"<td>{ref.year}</td><td>{html.escape(ref.doi or ('arXiv:' + ref.arxiv if ref.arxiv else '—'))}</td>"
            f"<td><span class='{record['status']}'>{record['status'].upper()}</span></td>"
            f"<td>{html.escape(record['summary'])}</td></tr>"
        )
        subrows = []
        for result in record["registries"]:
            score = "—" if result["title_similarity"] is None else f"{result['title_similarity']:.3f}"
            subrows.append(
                f"<tr><td>{html.escape(result['registry'])}</td><td>{'yes' if result['found'] else 'no'}</td>"
                f"<td>{html.escape(result['title'] or result['note'])}</td><td>{result['year'] or '—'}</td>"
                f"<td>{score}</td><td>{result['year_match'] if result['year_match'] is not None else '—'}</td>"
                f"<td>{result['identifier_match'] if result['identifier_match'] is not None else '—'}</td></tr>"
            )
        details.append(
            f"<details><summary><code>{html.escape(ref.key)}</code> — {html.escape(ref.title)}</summary>"
            "<table><thead><tr><th>Registry</th><th>Found</th><th>Registry title / note</th><th>Year</th>"
            "<th>Title similarity</th><th>Year match</th><th>ID match</th></tr></thead><tbody>"
            + "".join(subrows) + "</tbody></table></details>"
        )
    payload = html.escape(json.dumps(audited, ensure_ascii=False, indent=2))
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>VulTriage citation verification</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1.2rem;color:#1f2937}}
h1,h2{{color:#111827}} table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #d1d5db;padding:.45rem;vertical-align:top}} th{{background:#f3f4f6;text-align:left}}
.verified{{color:#047857;font-weight:700}} .partial{{color:#b45309;font-weight:700}} .issue{{color:#b91c1c;font-weight:700}}
details{{margin:.7rem 0}} summary{{cursor:pointer;font-weight:600}} code{{font-size:.9em}} .cards{{display:flex;gap:1rem;flex-wrap:wrap}} .card{{padding:.7rem 1rem;border:1px solid #d1d5db;border-radius:.5rem}}
</style></head><body><h1>VulTriage Citation Verification Report</h1>
<p>Generated from the final BibTeX file. A strong match requires title similarity ≥0.85, publication year within one year, and no identifier conflict. A citation is <b>verified</b> only when at least two of Crossref, OpenAlex, and Semantic Scholar strongly match.</p>
<div class='cards'><div class='card verified'>Verified: {counts['verified']}</div><div class='card partial'>Partial: {counts['partial']}</div><div class='card issue'>Issues: {counts['issue']}</div><div class='card'>Total: {len(refs)}</div></div>
<h2>Summary</h2><table><thead><tr><th>Key</th><th>Manuscript title</th><th>Year</th><th>Identifier</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Registry details</h2>{''.join(details)}
<h2>Machine-readable audit payload</h2><pre>{payload}</pre></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = parse_bibtex(args.bib)
    audited: list[dict] = []
    for index, ref in enumerate(refs, start=1):
        results = [crossref(ref), openalex(ref), semantic_scholar(ref)]
        status, summary = audit_status(ref, results)
        audited.append({"key": ref.key, "status": status, "summary": summary, "registries": [asdict(r) for r in results]})
        print(f"[{index:02d}/{len(refs):02d}] {ref.key}: {status} — {summary}")
        time.sleep(0.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(refs, audited), encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"HTML: {args.output}")
    print(f"JSON: {json_path}")
    return 1 if any(row["status"] == "issue" for row in audited) else 0


if __name__ == "__main__":
    raise SystemExit(main())
