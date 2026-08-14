"""Deterministic, disk-backed cross-dataset near-duplicate auditing.

The audit uses MinHash/LSH only to generate candidate pairs.  Every reported
pair is checked with the exact Jaccard similarity of the two lexical token
sets.  Function text and outcome labels are never stored in the audit index.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from vultriage.extension_data import canonicalize_function


MINHASH_PRIME = 4_294_967_291
MINHASH_SEED = "vultriage-extension-v2|minhash|sha256-v1"
DEFAULT_PERMUTATIONS = 128
DEFAULT_BANDS = 16
DEFAULT_ROWS_PER_BAND = 8

_MULTI_OPERATORS = tuple(
    sorted(
        {
            "%:%:",
            ">>=",
            "<<=",
            "<=>",
            "->*",
            "...",
            "::",
            ".*",
            "##",
            "->",
            "++",
            "--",
            "<<",
            ">>",
            "<=",
            ">=",
            "==",
            "!=",
            "&&",
            "||",
            "*=",
            "/=",
            "%=",
            "+=",
            "-=",
            "&=",
            "^=",
            "|=",
            "<:",
            ":>",
            "<%",
            "%>",
            "%:",
        },
        key=lambda value: (-len(value), value),
    )
)
_SINGLE_OPERATORS = frozenset("{}[]();:?.~!+-*/%^&|=<>,#\\")
_QUOTED_PREFIXES = ("u8R\"", "uR\"", "UR\"", "LR\"", "R\"", "u8\"", "u\"", "U\"", "L\"")
_CHAR_PREFIXES = ("u8'", "u'", "U'", "L'")


def _is_identifier_start(char: str) -> bool:
    return char == "_" or "A" <= char <= "Z" or "a" <= char <= "z"


def _is_identifier_continue(char: str) -> bool:
    return _is_identifier_start(char) or "0" <= char <= "9"


def _consume_quoted(text: str, quote_position: int, quote: str) -> int:
    position = quote_position + 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text[position] == quote:
            return position + 1
        else:
            position += 1
    return len(text)


def _consume_raw_string(text: str, prefix_position: int, prefix_length: int) -> int:
    delimiter_start = prefix_position + prefix_length
    opening = text.find("(", delimiter_start, delimiter_start + 17)
    if opening < 0:
        return len(text)
    delimiter = text[delimiter_start:opening]
    closing = text.find(")" + delimiter + '"', opening + 1)
    if closing < 0:
        return len(text)
    return closing + len(delimiter) + 2


def _consume_number(text: str, position: int) -> int:
    """Consume a deterministic C/C++ preprocessing-number approximation."""

    cursor = position + 1
    while cursor < len(text):
        char = text[cursor]
        previous = text[cursor - 1]
        if char.isascii() and (char.isalnum() or char in "_.'"):
            cursor += 1
        elif char in "+-" and previous in "eEpP":
            cursor += 1
        else:
            break
    return cursor


def cpp_lexical_tokens(text: str) -> tuple[str, ...]:
    """Return identifier, pp-number, and operator tokens from C/C++ source.

    Matching is case-sensitive and uses longest-match operator recognition.
    Whitespace, comments, string/character literals, and otherwise unrecognized
    bytes are omitted.  This deliberately avoids treating natural-language
    comments or literal contents as code similarity evidence.
    """

    source = canonicalize_function(text)
    tokens: list[str] = []
    position = 0
    while position < len(source):
        char = source[position]
        if char.isspace():
            position += 1
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position + 2)
            position = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            closing = source.find("*/", position + 2)
            position = len(source) if closing < 0 else closing + 2
            continue

        quoted = False
        for prefix in _QUOTED_PREFIXES:
            if source.startswith(prefix, position):
                if "R" in prefix:
                    position = _consume_raw_string(
                        source, position, len(prefix)
                    )
                else:
                    position = _consume_quoted(
                        source, position + len(prefix) - 1, '"'
                    )
                quoted = True
                break
        if quoted:
            continue
        if char == '"':
            position = _consume_quoted(source, position, '"')
            continue
        for prefix in _CHAR_PREFIXES:
            if source.startswith(prefix, position):
                position = _consume_quoted(
                    source, position + len(prefix) - 1, "'"
                )
                quoted = True
                break
        if quoted:
            continue
        if char == "'":
            position = _consume_quoted(source, position, "'")
            continue

        if _is_identifier_start(char):
            end = position + 1
            while end < len(source) and _is_identifier_continue(source[end]):
                end += 1
            tokens.append(source[position:end])
            position = end
            continue
        if "0" <= char <= "9":
            end = _consume_number(source, position)
            tokens.append(source[position:end])
            position = end
            continue

        operator = next(
            (item for item in _MULTI_OPERATORS if source.startswith(item, position)),
            None,
        )
        if operator is not None:
            tokens.append(operator)
            position += len(operator)
        elif char in _SINGLE_OPERATORS:
            tokens.append(char)
            position += 1
        else:
            position += 1
    return tuple(tokens)


def lexical_token_set(text: str) -> tuple[str, ...]:
    """Return the sorted unique lexical tokens used by set Jaccard."""

    return tuple(sorted(set(cpp_lexical_tokens(text))))


def serialize_token_set(tokens: Sequence[str]) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<I", len(tokens)))
    for token in tokens:
        encoded = token.encode("utf-8")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)
    return zlib.compress(bytes(payload), level=6)


def deserialize_token_set(payload: bytes) -> tuple[str, ...]:
    data = memoryview(zlib.decompress(payload))
    if len(data) < 4:
        raise ValueError("Truncated token-set payload")
    count = struct.unpack_from("<I", data, 0)[0]
    position = 4
    tokens: list[str] = []
    for _ in range(count):
        if position + 4 > len(data):
            raise ValueError("Truncated token length")
        length = struct.unpack_from("<I", data, position)[0]
        position += 4
        end = position + length
        if end > len(data):
            raise ValueError("Truncated token bytes")
        tokens.append(bytes(data[position:end]).decode("utf-8"))
        position = end
    if position != len(data):
        raise ValueError("Trailing bytes in token-set payload")
    return tuple(tokens)


def token_set_sha256(tokens: Sequence[str]) -> str:
    return hashlib.sha256(zlib.decompress(serialize_token_set(tokens))).hexdigest()


def minhash_coefficients(
    permutations: int = DEFAULT_PERMUTATIONS, seed: str = MINHASH_SEED
) -> tuple[np.ndarray, np.ndarray]:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    a = np.empty(permutations, dtype=np.uint64)
    b = np.empty(permutations, dtype=np.uint64)
    for index in range(permutations):
        digest = hashlib.sha256(f"{seed}|{index}".encode("utf-8")).digest()
        a[index] = 1 + int.from_bytes(digest[:8], "big") % (MINHASH_PRIME - 1)
        b[index] = int.from_bytes(digest[8:16], "big") % MINHASH_PRIME
    return a, b


def _token_minhash_value(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % MINHASH_PRIME


def minhash_signature(
    tokens: Sequence[str], coefficients: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    a, b = coefficients
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("MinHash coefficient arrays must be aligned vectors")
    if not tokens:
        return np.full(a.shape[0], MINHASH_PRIME, dtype="<u8")
    values = np.fromiter(
        (_token_minhash_value(token) for token in tokens),
        dtype=np.uint64,
        count=len(tokens),
    )
    result = np.full(a.shape[0], MINHASH_PRIME, dtype=np.uint64)
    for start in range(0, len(values), 4_096):
        chunk = values[start : start + 4_096]
        mixed = (a[:, None] * chunk[None, :] + b[:, None]) % MINHASH_PRIME
        result = np.minimum(result, mixed.min(axis=1))
    return np.asarray(result, dtype="<u8")


def signature_blob(signature: np.ndarray) -> bytes:
    return np.asarray(signature, dtype="<u8").tobytes(order="C")


def signature_from_blob(payload: bytes, permutations: int) -> np.ndarray:
    expected = permutations * np.dtype("<u8").itemsize
    if len(payload) != expected:
        raise ValueError(f"Signature has {len(payload)} bytes; expected {expected}")
    return np.frombuffer(payload, dtype="<u8")


def minhash_agreement(left: bytes, right: bytes, permutations: int) -> int:
    return int(
        np.count_nonzero(
            signature_from_blob(left, permutations)
            == signature_from_blob(right, permutations)
        )
    )


def lsh_band_keys(
    signature: np.ndarray, bands: int, rows_per_band: int
) -> tuple[bytes, ...]:
    if bands <= 0 or rows_per_band <= 0:
        raise ValueError("LSH bands and rows_per_band must be positive")
    if len(signature) != bands * rows_per_band:
        raise ValueError("LSH dimensions must consume the complete signature")
    keys: list[bytes] = []
    for band in range(bands):
        start = band * rows_per_band
        payload = np.asarray(
            signature[start : start + rows_per_band], dtype="<u8"
        ).tobytes(order="C")
        keys.append(
            hashlib.sha256(struct.pack("<I", band) + payload).digest()[:16]
        )
    return tuple(keys)


def exact_jaccard_counts(
    left: Sequence[str], right: Sequence[str]
) -> tuple[int, int, float]:
    left_set = set(left)
    right_set = set(right)
    intersection = len(left_set.intersection(right_set))
    union = len(left_set) + len(right_set) - intersection
    similarity = 1.0 if union == 0 else intersection / union
    return intersection, union, similarity


@dataclass(frozen=True)
class AuditDocument:
    dataset: str
    row_id: str
    source_file: str
    line_number: int
    project: str
    project_group: str
    exact_code_key: str
    tokens: tuple[str, ...]


class NearDuplicateIndex:
    """SQLite-backed MinHash LSH index and exact-candidate store."""

    _DOCUMENT_COMMIT_INTERVAL = 2_000
    _SCHEMA_COLUMNS = {
        "documents": (
            "doc_id",
            "dataset",
            "row_id",
            "source_file",
            "line_number",
            "project",
            "project_group",
            "exact_code_key",
            "token_count",
            "token_set_sha256",
            "token_blob",
            "signature_blob",
        ),
        "prime_bands": ("band", "band_key", "doc_id"),
        "target_bands": ("band", "band_key", "doc_id"),
        "candidates": ("target_doc_id", "prime_doc_id"),
        "flagged_pairs": (
            "target_doc_id",
            "prime_doc_id",
            "intersection_count",
            "union_count",
            "minhash_agreement",
        ),
    }

    def __init__(
        self,
        path: Path,
        *,
        permutations: int = DEFAULT_PERMUTATIONS,
        bands: int = DEFAULT_BANDS,
        rows_per_band: int = DEFAULT_ROWS_PER_BAND,
        seed: str = MINHASH_SEED,
    ):
        if path.exists():
            raise FileExistsError(f"Near-duplicate work database already exists: {path}")
        if permutations != bands * rows_per_band:
            raise ValueError("permutations must equal bands * rows_per_band")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.permutations = permutations
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.coefficients = minhash_coefficients(permutations, seed)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript(
            """
            CREATE TABLE documents (
                doc_id INTEGER PRIMARY KEY,
                dataset TEXT NOT NULL CHECK(dataset IN ('primevul', 'diversevul')),
                row_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                project TEXT NOT NULL,
                project_group TEXT NOT NULL,
                exact_code_key TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                token_set_sha256 TEXT NOT NULL,
                token_blob BLOB NOT NULL,
                signature_blob BLOB NOT NULL,
                UNIQUE(dataset, row_id)
            );
            CREATE TABLE seen_prime_keys (
                exact_code_key TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE prime_bands (
                band INTEGER NOT NULL,
                band_key BLOB NOT NULL,
                doc_id INTEGER NOT NULL,
                PRIMARY KEY(band, band_key, doc_id),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE target_bands (
                band INTEGER NOT NULL,
                band_key BLOB NOT NULL,
                doc_id INTEGER NOT NULL,
                PRIMARY KEY(band, band_key, doc_id),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            CREATE INDEX target_bands_doc_id ON target_bands(doc_id);
            CREATE TABLE candidates (
                target_doc_id INTEGER NOT NULL,
                prime_doc_id INTEGER NOT NULL,
                PRIMARY KEY(target_doc_id, prime_doc_id),
                FOREIGN KEY(target_doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY(prime_doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE flagged_pairs (
                target_doc_id INTEGER NOT NULL,
                prime_doc_id INTEGER NOT NULL,
                intersection_count INTEGER NOT NULL,
                union_count INTEGER NOT NULL,
                minhash_agreement INTEGER NOT NULL,
                PRIMARY KEY(target_doc_id, prime_doc_id),
                FOREIGN KEY(target_doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY(prime_doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            """
        )
        self._inserted_documents = 0

    @classmethod
    def open_existing(
        cls,
        path: Path,
        *,
        permutations: int = DEFAULT_PERMUTATIONS,
        bands: int = DEFAULT_BANDS,
        rows_per_band: int = DEFAULT_ROWS_PER_BAND,
        seed: str = MINHASH_SEED,
    ) -> "NearDuplicateIndex":
        """Open a previously completed index without rebuilding source data."""

        if not path.exists():
            raise FileNotFoundError(path)
        if permutations != bands * rows_per_band:
            raise ValueError("permutations must equal bands * rows_per_band")
        self = object.__new__(cls)
        self.path = path
        self.permutations = permutations
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.coefficients = minhash_coefficients(permutations, seed)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        required_tables = {"documents", "prime_bands", "target_bands", "candidates", "flagged_pairs"}
        observed_tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - observed_tables
        if missing:
            self.connection.close()
            raise ValueError(f"Existing near-duplicate index is missing tables: {sorted(missing)}")
        for table, expected_columns in self._SCHEMA_COLUMNS.items():
            observed_columns = tuple(
                str(row[1])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            )
            if observed_columns != expected_columns:
                self.connection.close()
                raise ValueError(
                    f"Existing near-duplicate index has unexpected {table} schema: "
                    f"{observed_columns!r}"
                )
        bad_signatures = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM documents
                WHERE token_count < 0 OR length(signature_blob) != ?
                """,
                (permutations * np.dtype("<u8").itemsize,),
            ).fetchone()[0]
        )
        if bad_signatures:
            self.connection.close()
            raise ValueError(f"Existing near-duplicate index has {bad_signatures} invalid signatures")
        expected_datasets = (("prime_bands", "primevul"), ("target_bands", "diversevul"))
        for table, dataset in expected_datasets:
            document_total = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset,)
                ).fetchone()[0]
            )
            band_total = int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            if band_total != document_total * bands:
                self.connection.close()
                raise ValueError(
                    f"Existing near-duplicate index has {band_total} {table} rows; "
                    f"expected {document_total * bands}"
                )
            incomplete = int(
                self.connection.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT doc_id FROM {table}
                        GROUP BY doc_id HAVING COUNT(*) != ?
                    )
                    """,
                    (bands,),
                ).fetchone()[0]
            )
            if incomplete:
                self.connection.close()
                raise ValueError(f"Existing near-duplicate index has {incomplete} incomplete {table} documents")
            cross_dataset = int(
                self.connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table} AS bands
                    JOIN documents AS documents ON documents.doc_id = bands.doc_id
                    WHERE documents.dataset != ?
                    """,
                    (dataset,),
                ).fetchone()[0]
            )
            if cross_dataset:
                self.connection.close()
                raise ValueError(f"Existing near-duplicate index has cross-dataset {table} rows")
        for table in ("candidates", "flagged_pairs"):
            cross_dataset = int(
                self.connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table} AS pairs
                    JOIN documents AS target ON target.doc_id = pairs.target_doc_id
                    JOIN documents AS prime ON prime.doc_id = pairs.prime_doc_id
                    WHERE target.dataset != 'diversevul' OR prime.dataset != 'primevul'
                    """
                ).fetchone()[0]
            )
            if cross_dataset:
                self.connection.close()
                raise ValueError(f"Existing near-duplicate index has cross-dataset {table} rows")
        self._inserted_documents = 0
        return self

    def __enter__(self) -> "NearDuplicateIndex":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def register_prime_key(self, exact_code_key: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen_prime_keys(exact_code_key) VALUES (?)",
            (exact_code_key,),
        )
        return cursor.rowcount == 1

    def add_document(self, document: AuditDocument) -> int:
        if document.dataset not in {"primevul", "diversevul"}:
            raise ValueError(f"Unknown dataset: {document.dataset}")
        signature = minhash_signature(document.tokens, self.coefficients)
        token_blob = serialize_token_set(document.tokens)
        cursor = self.connection.execute(
            """
            INSERT INTO documents(
                dataset, row_id, source_file, line_number, project,
                project_group, exact_code_key, token_count, token_set_sha256,
                token_blob, signature_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.dataset,
                document.row_id,
                document.source_file,
                document.line_number,
                document.project,
                document.project_group,
                document.exact_code_key,
                len(document.tokens),
                token_set_sha256(document.tokens),
                token_blob,
                signature_blob(signature),
            ),
        )
        doc_id = int(cursor.lastrowid)
        table = "prime_bands" if document.dataset == "primevul" else "target_bands"
        self.connection.executemany(
            f"INSERT INTO {table}(band, band_key, doc_id) VALUES (?, ?, ?)",
            (
                (band, key, doc_id)
                for band, key in enumerate(
                    lsh_band_keys(signature, self.bands, self.rows_per_band)
                )
            ),
        )
        self._inserted_documents += 1
        if self._inserted_documents % self._DOCUMENT_COMMIT_INTERVAL == 0:
            self.connection.commit()
        return doc_id

    def finish_documents(self) -> None:
        self.connection.commit()

    def document_count(self, dataset: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset,)
            ).fetchone()[0]
        )

    def token_count_summary(self, dataset: str) -> dict[str, float | int]:
        count, minimum, maximum, mean, empty = self.connection.execute(
            """
            SELECT COUNT(*), MIN(token_count), MAX(token_count), AVG(token_count),
                   SUM(token_count = 0)
            FROM documents WHERE dataset = ?
            """,
            (dataset,),
        ).fetchone()
        return {
            "documents": int(count),
            "minimum": int(minimum or 0),
            "maximum": int(maximum or 0),
            "mean": float(mean or 0.0),
            "empty": int(empty or 0),
        }

    def candidate_doc_id_bounds(self) -> tuple[int, int] | None:
        row = self.connection.execute(
            "SELECT MIN(doc_id), MAX(doc_id) FROM documents WHERE dataset='diversevul'"
        ).fetchone()
        if row[0] is None:
            return None
        return int(row[0]), int(row[1])

    def generate_candidates(
        self,
        *,
        threshold: float,
        batch_size: int = 2_000,
    ) -> Iterator[dict[str, int]]:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        bounds = self.candidate_doc_id_bounds()
        if bounds is None:
            return
        minimum, maximum = bounds
        previous_count = 0
        for start in range(minimum, maximum + 1, batch_size):
            end = min(maximum, start + batch_size - 1)
            self.connection.execute(
                """
                INSERT OR IGNORE INTO candidates(target_doc_id, prime_doc_id)
                SELECT tb.doc_id, pb.doc_id
                FROM target_bands AS tb
                JOIN prime_bands AS pb
                  ON pb.band = tb.band AND pb.band_key = tb.band_key
                JOIN documents AS target ON target.doc_id = tb.doc_id
                JOIN documents AS prime ON prime.doc_id = pb.doc_id
                WHERE tb.doc_id BETWEEN ? AND ?
                  AND prime.token_count >= ? * target.token_count
                  AND target.token_count >= ? * prime.token_count
                """,
                (start, end, threshold, threshold),
            )
            self.connection.commit()
            total = int(
                self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            )
            yield {
                "start_doc_id": start,
                "end_doc_id": end,
                "new_candidates": total - previous_count,
                "total_candidates": total,
            }
            previous_count = total

    def verify_candidates(
        self, threshold: float, *, cache_size: int = 8_192
    ) -> Iterator[dict[str, object]]:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        cursor = self.connection.execute(
            """
            SELECT candidate.target_doc_id, candidate.prime_doc_id,
                   target.token_blob, prime.token_blob,
                   target.signature_blob, prime.signature_blob
            FROM candidates AS candidate
            JOIN documents AS target ON target.doc_id = candidate.target_doc_id
            JOIN documents AS prime ON prime.doc_id = candidate.prime_doc_id
            ORDER BY candidate.target_doc_id, candidate.prime_doc_id
            """
        )
        cache: OrderedDict[int, tuple[str, ...]] = OrderedDict()
        current_target_id = -1
        current_target_tokens: tuple[str, ...] = ()
        checked = 0
        for (
            target_id,
            prime_id,
            target_blob,
            prime_blob,
            target_signature,
            prime_signature,
        ) in cursor:
            target_id = int(target_id)
            prime_id = int(prime_id)
            if target_id != current_target_id:
                current_target_tokens = deserialize_token_set(target_blob)
                current_target_id = target_id
            prime_tokens = cache.pop(prime_id, None)
            if prime_tokens is None:
                prime_tokens = deserialize_token_set(prime_blob)
            cache[prime_id] = prime_tokens
            if len(cache) > cache_size:
                cache.popitem(last=False)
            intersection, union, similarity = exact_jaccard_counts(
                current_target_tokens, prime_tokens
            )
            checked += 1
            if similarity + 1e-15 < threshold:
                continue
            agreement = minhash_agreement(
                target_signature, prime_signature, self.permutations
            )
            insert_cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO flagged_pairs(
                    target_doc_id, prime_doc_id, intersection_count,
                    union_count, minhash_agreement
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (target_id, prime_id, intersection, union, agreement),
            )
            if insert_cursor.rowcount == 0:
                existing = self.connection.execute(
                    """
                    SELECT intersection_count, union_count, minhash_agreement
                    FROM flagged_pairs
                    WHERE target_doc_id = ? AND prime_doc_id = ?
                    """,
                    (target_id, prime_id),
                ).fetchone()
                expected = (intersection, union, agreement)
                if existing is None or tuple(int(value) for value in existing) != expected:
                    raise ValueError(
                        "Existing flagged pair disagrees with exact verification: "
                        f"target_doc_id={target_id}, prime_doc_id={prime_id}"
                    )
            yield {
                "checked_candidates": checked,
                "target_doc_id": target_id,
                "prime_doc_id": prime_id,
                "intersection_count": intersection,
                "union_count": union,
                "exact_jaccard": similarity,
                "minhash_agreement": agreement,
            }
        self.connection.commit()

    def candidate_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def flagged_pair_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM flagged_pairs").fetchone()[0])

    def flagged_target_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT target_doc_id) FROM flagged_pairs"
            ).fetchone()[0]
        )

    def candidate_target_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT target_doc_id) FROM candidates"
            ).fetchone()[0]
        )

    def iter_flagged_pairs(self) -> Iterator[dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT target.row_id, target.project, target.project_group,
                   target.source_file, target.line_number, target.exact_code_key,
                   target.token_count, prime.row_id, prime.project,
                   prime.project_group, prime.source_file, prime.line_number,
                   prime.exact_code_key, prime.token_count,
                   flagged.intersection_count, flagged.union_count,
                   flagged.minhash_agreement
            FROM flagged_pairs AS flagged
            JOIN documents AS target ON target.doc_id = flagged.target_doc_id
            JOIN documents AS prime ON prime.doc_id = flagged.prime_doc_id
            ORDER BY target.row_id, prime.row_id
            """
        )
        fields = (
            "target_row_id",
            "target_project",
            "target_project_group",
            "target_source_file",
            "target_line_number",
            "target_exact_code_key",
            "target_token_count",
            "prime_row_id",
            "prime_project",
            "prime_project_group",
            "prime_source_file",
            "prime_line_number",
            "prime_exact_code_key",
            "prime_token_count",
            "intersection_count",
            "union_count",
            "minhash_agreement",
        )
        for row in cursor:
            result = dict(zip(fields, row))
            union = int(result["union_count"])
            result["exact_jaccard"] = (
                1.0 if union == 0 else int(result["intersection_count"]) / union
            )
            result["minhash_estimate"] = int(result["minhash_agreement"]) / self.permutations
            yield result

    def iter_flagged_targets(self) -> Iterator[dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT target.row_id, target.project, target.project_group,
                   target.source_file, target.line_number, target.exact_code_key,
                   target.token_count, COUNT(*),
                   MAX(CASE WHEN flagged.union_count = 0 THEN 1.0
                            ELSE 1.0 * flagged.intersection_count /
                                 flagged.union_count END)
            FROM flagged_pairs AS flagged
            JOIN documents AS target ON target.doc_id = flagged.target_doc_id
            GROUP BY flagged.target_doc_id
            ORDER BY target.row_id
            """
        )
        fields = (
            "target_row_id",
            "target_project",
            "target_project_group",
            "target_source_file",
            "target_line_number",
            "target_exact_code_key",
            "target_token_count",
            "flagged_prime_pair_count",
            "maximum_exact_jaccard",
        )
        for row in cursor:
            yield dict(zip(fields, row))

    def affected_projects(self) -> list[dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT target.project_group,
                   COUNT(DISTINCT flagged.target_doc_id), COUNT(*)
            FROM flagged_pairs AS flagged
            JOIN documents AS target ON target.doc_id = flagged.target_doc_id
            GROUP BY target.project_group
            ORDER BY target.project_group COLLATE NOCASE, target.project_group
            """
        )
        return [
            {
                "project_group": str(project),
                "affected_target_rows": int(rows),
                "flagged_pairs": int(pairs),
            }
            for project, rows, pairs in cursor
        ]


class PeakRSSMonitor:
    """Sample process RSS while a long-running audit is active."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSSMonitor":
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil is required for peak RSS recording") from exc
        process = psutil.Process()

        def sample() -> None:
            while not self._stop.is_set():
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(process.memory_info().rss)
                )
                self._stop.wait(self.interval_seconds)
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, int(process.memory_info().rss)
            )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5))


def lsh_candidate_probability(jaccard: float, bands: int, rows_per_band: int) -> float:
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError("jaccard must be in [0, 1]")
    return 1.0 - (1.0 - jaccard**rows_per_band) ** bands


def cardinality_can_reach_threshold(left: int, right: int, threshold: float) -> bool:
    if left < 0 or right < 0:
        raise ValueError("token-set cardinalities must be nonnegative")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if left == 0 or right == 0:
        return left == right
    return min(left, right) / max(left, right) + 1e-15 >= threshold
