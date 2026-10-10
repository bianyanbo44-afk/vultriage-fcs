"""Build a local aggregate-only v4 repository tree and two checked ZIP archives.

The published v3 research ZIP is an immutable base, not rebuilt from the current
manuscript. New evidence is selected only through each module's final public
manifest. No experiment, training, network, Git push or publication is run.

Examples:
    python scripts/build_trace_v4_public_release.py self-test
    python scripts/build_trace_v4_public_release.py inspect --allow-partial
    python scripts/build_trace_v4_public_release.py build --allow-partial --output .../preview-1
    python scripts/build_trace_v4_public_release.py build --output .../complete-1
    python scripts/build_trace_v4_public_release.py verify --tree .../repository --archive ...zip
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/packaging"
V3_ZIP = ROOT / "tmp/neurocomputing-strengthening/github-publication-v3/TRACE_reproducibility_artifact.zip"
V3_SHA = "366E697441884462C0B67C47E419FC3D8B669C012A7409CD17F91058FDCBC6A2"
V3_MANIFEST_SHA = "C835D1DF4C7C731F436C444405CBD6CEA02E3EA9F79564A7833621F9AB9B80FF"
V4_MANIFEST = "TRACE_V4_RELEASE_MANIFEST.json"
VERSION = "trace-validation-v4-2026-10-11"
ZIP_DATE = (2026, 10, 11, 0, 0, 0)
PRIVATE_PARTS = frozenset({".git", ".codex", "__pycache__", ".venv", "node_modules", "site_packages", "runtime_dependencies",
    "thirdpackage", "raw", "raw_data", "raw_datasets", "datasets", "vault", "vaults", "label_vault", "target_labels",
    "tokens", "token_cache", "token_caches", "cache", "caches", "checkpoints", "checkpoint", "weights", "predictions", "decisions",
    "credentials", "secrets", "author_portraits", "author_photos", "blind_review", "blind_review_v3", "mock_review", "mock_reviews",
    "attempt_1", "attempt_1_archived", "vendor", "vendors", "skills"})
SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".csv", ".tex", ".bbl", ".bib", ".cls", ".bst", ".pdf", ".svg", ".png", ".tif", ".tiff", ".ini", ".diff", ".sha256"})
RAW_FIELDS = frozenset({"row_id", "row_ids", "position", "positions", "target_position", "target_positions", "source_position", "source_positions",
    "target_probability", "target_probabilities", "source_probability", "source_probabilities", "arrival_eta", "service_batch", "prediction_batch",
    "raw_calibration_weights", "raw_target_weights", "domain_fold_ids", "target_labels", "labels", "fit_window_indices", "target_indices",
    "commit_id", "commit_ids", "cve", "cve_id", "cve_ids", "func", "function_text", "func_before", "text"})
RAW_CSV_FIELDS = RAW_FIELDS | {"label", "probability", "prediction", "action", "token_ids", "input_ids", "attention_mask", "raw_weight"}
CAPACITY_CODE = (
    "src/capacity_feedback_v4.py", "scripts/test_capacity_feedback_v4.py", "scripts/check_capacity_feedback_v4.py",
    "scripts/export_capacity_feedback_v4.py", "configs/capacity_feedback_v4.json", "protocol/capacity_feedback_v4.md")
RATIO_CODE = ("src/analyze_ratio_diagnostics_v4.py", "configs/ratio_diagnostics_v4.json", "tests/test_ratio_diagnostics_v4.py")
INDEPENDENT_CODE = (
    "src/independent_model_v4_1.py", "src/prepare_independent_model_v4_1.py", "src/encode_independent_model_v4_1.py",
    "src/train_independent_model_v4_1.py", "src/predict_independent_model_v4_1.py", "src/evaluate_independent_model_v4_1.py",
    "src/seal_joint_independent_model_v4_1.py", "src/reconstruct_independent_model_v4_1.py", "src/export_independent_model_v4_1.py",
    "src/audit_fits_independent_model_v4_1.py",
    "src/prepare_independent_model_v4.py", "src/audit_independent_model_v4.py", "configs/independent_model_v4_1.json",
    "protocol/independent_model_v4_1.md", "tests/test_independent_model_v4_1.py", "tests/test_reconstruct_independent_model_v4_1.py")
EXCLUDED_OLD_MODEL = frozenset({"src/independent_model_v4.py", "src/train_independent_model_v4.py", "src/predict_independent_model_v4.py",
    "src/encode_independent_model_v4.py", "src/evaluate_independent_model_v4.py", "configs/independent_model_v4.json", "protocol/independent_model_v4.md"})
FINAL_PUBLIC_MANIFEST_SHA = {
    "prom": "6A07CB04DB575385C5467D3680C9022E73EEB33361D2D3AECD5234AB8654248C",
    "figures": "56582E53297D8FA6A0316F5C388D16AE51BD744ACE4EF5F0FA54F6686797814A",
    "independent_model": "3DF0AB1B2BA5EAE8C09903DA75FA97E8FC9B62140D6A7101D6604ED2FF86BE37",
}
FIGURE_DIRECTORY = "paper/neurocomputing_revision/figures/v4"
FIGURE_PREFIX = "public_results/kbs-validation-v4/figure-data-v4/"
FIGURE_NAMES = ("fig_trace_workflow_v4", "fig_capacity_v4", "fig_feedback_v4")
FIGURE_SUFFIXES = (".alignment-layout.json", ".alignment.json", ".alignment.svg", ".geometry.json", ".source-preflight.json",
    ".pdf", ".svg", ".png", ".tiff", ".pdf-geometry.json", ".glyph-audit.json", ".collision-audit.json", ".collision-audit.pdf")
FIGURE_CSV = frozenset({"fig_capacity_v4_completion.csv", "fig_capacity_v4_cost_pairs.csv", "fig_feedback_v4_pairs.csv", "fig_feedback_v4_projects.csv"})
FIGURE_CODE = ("src/make_kbs_strengthening_figures_v4.py", "src/make_neural_validation_figures.py")
FIGURE_QA_MEMBERS = frozenset(prefix + figure + suffix
    for prefix in ("", FIGURE_DIRECTORY + "/", FIGURE_PREFIX)
    for figure in FIGURE_NAMES for suffix in FIGURE_SUFFIXES)
CAPACITY_ANALYSIS_SHA = "C8D2FE9E97918776570CA3E0C3A554061A742D7950FD0369985299525805611E"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def dump_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Repeated JSON key: " + key)
        result[key] = value
    return result


def read_json(data: bytes, label: str):
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs_unique)
    require(isinstance(value, dict), "Expected JSON object: " + label)
    return value


def sha_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value), "Invalid manifest SHA-256")
    return value.upper()


def safe_member(name: str):
    require(isinstance(name, str) and name and not any(c in name for c in ("\\", ":", "\0")), "Unsafe archive member")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and all(part not in {"", ".", ".."} for part in name.split("/")), "Member path traversal: " + name)
    normalized = {part.casefold().replace("-", "_") for part in path.parts}
    require(not normalized.intersection(PRIVATE_PARTS), "Private/runtime/archive member: " + name)
    require(not any(mark in name.casefold() for mark in (".env", "author_biograph", "competing_interests", "portrait", "mock_review", "blind_review", ".overlay.")), "Author/private member: " + name)
    require(name in FIGURE_QA_MEMBERS or not any(mark in name.casefold() for mark in (".alignment.", ".alignment-", ".collision.", ".collision-")), "Unselected figure QA member: " + name)
    licence_name = path.name.upper() in {"LICENSE", "LICENCE", "NOTICE", "COPYING"}
    require(path.suffix.casefold() in SUFFIXES or licence_name, "Non-public binary/file type: " + name)
    require(name not in EXCLUDED_OLD_MODEL, "Discarded model entrypoint: " + name)
    # Runtime data filenames remain forbidden even when their suffix is textual.
    if path.suffix.casefold() not in {".py", ".md"}:
        require(not re.search(r"(?:^|[_-])(?:row_ids|target_positions|target_rows|source_rows|source_train|source_calibration_rows|source_model_validation_rows|prediction_rows|row_predictions|label_events|batch_ledger|fit_ledger|token_ids)(?:$|[_-])", path.stem.casefold()), "Per-function member name: " + name)
    return name


def safe_source(base: Path, relative: str) -> Path:
    safe_member(relative)
    root = base.resolve(strict=True)
    require(not base.is_symlink(), "Source root cannot be a symlink")
    direct = base / relative
    cursor = direct
    while cursor != base:
        require(not cursor.is_symlink(), "Symlink source excluded: " + relative)
        cursor = cursor.parent
    resolved = direct.resolve(strict=True)
    require(resolved.is_file() and resolved.is_relative_to(root), "Source escapes declared manifest root")
    return resolved


def validate_data(name: str, data: bytes):
    suffix = PurePosixPath(name).suffix.casefold()
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Missing/duplicate aggregate CSV columns: " + name)
        offending = {field.casefold() for field in reader.fieldnames} & RAW_CSV_FIELDS
        if name == "paper/neurocomputing_revision/figures/data/fig_workflow_nodes.csv":
            # The pinned v3 diagram node labels are annotation, not outcomes.
            offending -= {"label", "text"}
        require(not offending, "Per-function CSV columns: " + name + " " + str(sorted(offending)))
    elif suffix == ".json":
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs_unique)
        require(isinstance(value, (dict, list)), "Expected aggregate JSON object/list: " + name)

        def scan(node, trail=""):
            if isinstance(node, dict):
                if {"row_id", "position", "func", "func_before", "function_text"}.intersection(k.casefold() for k in node):
                    require(not any(k.casefold() in {"func", "func_before", "function_text", "row_id"} and isinstance(v, str) and v for k, v in node.items()), "Raw row/function record in JSON: " + name)
                for key, child in node.items():
                    if key.casefold() in RAW_FIELDS and isinstance(child, (list, dict)) and child:
                        # Source class-count maps are aggregate evidence, not labels.
                        class_count_map = key.casefold() == "labels" and isinstance(child, dict) and set(child).issubset({"0", "1"}) and all(isinstance(v, int) and v >= 0 for v in child.values())
                        require(class_count_map, "Per-function JSON field: " + name + "/" + trail + key)
                    scan(child, trail + key + "/")
            elif isinstance(node, list):
                for child in node:
                    scan(child, trail)
            elif isinstance(node, str) and (name.startswith(("public_results/v4/", FIGURE_PREFIX)) or name.startswith("TRACE_V4_")):
                require(not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?:^|\s)/(?:Users|home|mnt)/|^\\\\", node),
                        "Absolute machine path in new public JSON: " + name + "/" + trail)
        scan(value)
    elif suffix == ".py":
        ast.parse(data.decode("utf-8-sig"), filename=name)


@dataclass
class Payload:
    files: dict[str, bytes]
    origins: dict[str, dict]

    def add(self, name: str, data: bytes, origin: dict):
        safe_member(name)
        validate_data(name, data)
        folded = {existing.casefold(): existing for existing in self.files}
        require(name.casefold() not in folded or folded[name.casefold()] == name, "Windows case-colliding payload paths")
        if name in self.files:
            require(self.files[name] == data, "Would overwrite a v3/shared source with different bytes: " + name)
            return
        self.files[name], self.origins[name] = data, origin


def collect_v3(path: Path) -> tuple[Payload, dict]:
    require(file_hash(path) == V3_SHA, "Published v3 base archive changed")
    files, origins = {}, {}
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "v3 archive CRC failure")
        require(len(archive.namelist()) == len(set(archive.namelist())), "Repeated v3 ZIP member")
        index_bytes = archive.read("ARCHIVE_MANIFEST.json")
        require(digest(index_bytes) == V3_MANIFEST_SHA, "v3 member manifest changed")
        index = read_json(index_bytes, "v3 archive manifest")
        expected = index["files_sha256"]
        require(set(archive.namelist()) == set(expected) | {"ARCHIVE_MANIFEST.json"}, "v3 archive inventory mismatch")
        payload = Payload(files, origins)
        for info in archive.infolist():
            require(not stat.S_ISLNK(info.external_attr >> 16), "v3 ZIP symlink member")
            name, data = info.filename, archive.read(info.filename)
            require(name == "ARCHIVE_MANIFEST.json" or digest(data) == sha_value(expected[name]), "v3 payload byte/hash mismatch")
            payload.add(name, data, {"scope": "published_v3_unchanged", "archive_sha256": V3_SHA, "sha256": digest(data)})
    return payload, {"archive_path": path.name, "archive_sha256": V3_SHA, "manifest_sha256": V3_MANIFEST_SHA,
                     "payload_members": len(expected), "all_original_payload_and_manifest_bytes_preserved": True}


def manifest_records(manifest: dict, module: str):
    if module == "capacity_feedback":
        require(manifest.get("status") == "COMPLETE_AGGREGATE_ONLY", "Capacity aggregate export incomplete")
        records = manifest["files"]
    elif module == "ratio_diagnostics":
        require(manifest.get("status") == "PASS", "Ratio diagnostics incomplete")
        records = [{"path": path, "sha256": sha} for path, sha in manifest["files_sha256"].items()]
    elif module == "prom":
        require(manifest.get("raw_row_data_exported") is False and manifest.get("full_official_repository_exported") is False, "PROM export does not enforce aggregate/small-source scope")
        records = manifest["included_files"]
    else:
        require(manifest.get("schema_version") == "trace-independent-model-v4.1-public-aggregates-1" and manifest.get("protocol_version") == "trace-independent-model-v4.1", "Only the corrected CodeT5 v4.1 aggregate release is eligible")
        require(manifest.get("status", "PASS").casefold() not in {"pending", "running", "failed", "incomplete"}, "Corrected CodeT5 aggregate export incomplete")
        if isinstance(manifest.get("files"), list):
            records = manifest["files"]
        elif isinstance(manifest.get("included_files"), list):
            records = manifest["included_files"]
        else:
            mapping = manifest.get("members", manifest.get("files_sha256", manifest.get("outputs_sha256")))
            require(isinstance(mapping, dict) and mapping, "Missing corrected CodeT5 public inventory")
            records = [{"path": path, "sha256": sha} for path, sha in mapping.items()]
    require(isinstance(records, list) and records, "Empty public evidence manifest")
    inventory = {}
    for record in records:
        path = record.get("path", record.get("file"))
        require(isinstance(path, str) and path not in inventory, "Repeated/invalid public member")
        safe_member(path)
        inventory[path] = sha_value(record["sha256"])
    return inventory


MODULES = {
    "capacity_feedback": ("outputs/kbs-validation-v4/capacity_feedback/public-aggregate", "capacity_feedback_v4_public_manifest.json"),
    "ratio_diagnostics": ("outputs/kbs-validation-v4/ratio_diagnostics", "manifest.json"),
    "prom": ("outputs/kbs-validation-v4/prom/public-evidence", "public_export_manifest.json"),
    "independent_model": ("outputs/kbs-validation-v4/independent_model/attempt-2/aggregate-export", "aggregate_manifest.json"),
    "figures": (FIGURE_DIRECTORY, "figure_manifest_v4.json")}


def add_code(payload: Payload, names: tuple[str, ...], module: str):
    result = []
    for name in names:
        path = safe_source(ROOT, name)
        data = path.read_bytes()
        payload.add(name, data, {"scope": module + "_reproduction_source", "source_path": name, "sha256": digest(data)})
        result.append({"path": name, "sha256": digest(data)})
    return result


def collect_module(payload: Payload, module: str) -> dict:
    directory, filename = MODULES[module]
    root = ROOT / directory
    manifest_path = root / filename
    require(manifest_path.exists(), "Pending final public manifest: " + str(manifest_path))
    manifest_bytes = manifest_path.read_bytes()
    if module in FINAL_PUBLIC_MANIFEST_SHA:
        require(digest(manifest_bytes) == FINAL_PUBLIC_MANIFEST_SHA[module], "Owner-finalized public manifest changed: " + module)
    manifest = read_json(manifest_bytes, module + " public manifest")
    inventory = manifest_records(manifest, module)
    prefix = "public_results/v4/" + module + "/"
    records = []
    for relative, sha in sorted(inventory.items()):
        path = safe_source(root, relative)
        data = path.read_bytes()
        require(digest(data) == sha, "Final module member changed: " + module + "/" + relative)
        target = prefix + relative
        payload.add(target, data, {"scope": module + "_public_manifest", "source_path": relative, "sha256": sha, "manifest_sha256": digest(manifest_bytes)})
        records.append({"source": relative, "destination": target, "sha256": sha})
        # Keep the manifest-bound packet intact, and expose its source at repo paths.
        if module == "prom" and (relative.startswith(("scripts/", "configs/", "protocol/", "third_party/")) or relative == "requirements-prom-official-v4.txt"):
            payload.add(relative, data, {"scope": "prom_repository_source_duplicate", "manifest_sha256": digest(manifest_bytes), "sha256": sha})
    payload.add(prefix + filename, manifest_bytes, {"scope": module + "_public_manifest_inventory", "sha256": digest(manifest_bytes)})
    for suffix in (".sha256",):
        companion = root / (Path(filename).stem + suffix)
        if companion.exists():
            data = companion.read_bytes()
            require(data.decode("ascii").strip().upper() == digest(manifest_bytes), "Public manifest hash sidecar mismatch")
            payload.add(prefix + companion.name, data, {"scope": module + "_public_manifest_sidecar", "sha256": digest(data)})
    code = ()
    if module == "capacity_feedback":
        code = CAPACITY_CODE
        bindings = read_json((root / "evidence_bindings.json").read_bytes(), "capacity evidence bindings")
        for record in bindings["code"]:
            require(file_hash(ROOT / record["path"]) == record["sha256"], "Capacity source/test/checker changed after verification")
    elif module == "ratio_diagnostics":
        code = RATIO_CODE
        require(file_hash(ROOT / code[0]) == manifest["script_sha256"] and file_hash(ROOT / code[1]) == manifest["config_sha256"], "Ratio execution source/config binding stale")
    elif module == "independent_model":
        code = INDEPENDENT_CODE
        whitelist = manifest.get("required_code_whitelist")
        require(isinstance(whitelist, dict) and set(whitelist) == set(INDEPENDENT_CODE), "Corrected model code whitelist differs from the owner-confirmed closure")
        for path, expected in whitelist.items():
            require(file_hash(ROOT / path) == sha_value(expected), "Corrected model source changed after aggregate export: " + path)
    code_records = add_code(payload, code, module) if code else []
    return {"status": "READY", "public_directory": directory, "manifest": filename, "manifest_sha256": digest(manifest_bytes),
            "manifest_payload_members": len(records), "member_mapping": records, "additional_repository_code": code_records}


def collect_figures(payload: Payload) -> dict:
    directory, filename = MODULES["figures"]
    root = ROOT / directory
    manifest_bytes = safe_source(root, filename).read_bytes()
    require(digest(manifest_bytes) == FINAL_PUBLIC_MANIFEST_SHA["figures"], "Owner-finalized figure manifest changed")
    manifest = read_json(manifest_bytes, "final figure evidence manifest")
    require(manifest.get("schema_version") == "trace-v4-capacity-feedback-figures-1", "Unknown figure evidence schema")
    require(manifest.get("capacity_analysis_manifest_sha256") == CAPACITY_ANALYSIS_SHA, "Figure analysis binding changed")
    public_capacity_manifest = "public_results/v4/capacity_feedback/capacity_feedback_v4_public_manifest.json"
    require(public_capacity_manifest in payload.files and manifest.get("capacity_public_aggregate_manifest_sha256") == digest(payload.files[public_capacity_manifest]),
            "Figure public-capacity aggregate manifest binding changed")
    require(set(manifest["figures"]) == set(FIGURE_NAMES), "Final three-figure roster changed")
    require(file_hash(safe_source(ROOT, FIGURE_CODE[0])) == manifest["source_script_sha256"], "Figure plotting source changed after QA")
    require(file_hash(safe_source(ROOT, FIGURE_CODE[1])) == manifest["helper_script_sha256"], "Figure helper source changed after QA")
    require(digest(payload.files[FIGURE_CODE[1]]) == manifest["helper_script_sha256"], "Figure helper differs from the immutable v3 source")
    contract_path = "figure_contract_v4.json"
    contract_bytes = safe_source(root, contract_path).read_bytes()
    require(digest(contract_bytes) == manifest["contract_sha256"], "Figure design contract changed after QA")
    contract = read_json(contract_bytes, "figure contract")
    require(set(contract["new_figures"]) == set(FIGURE_NAMES), "Figure contract roster changed")
    payload.add(directory + "/" + contract_path, contract_bytes, {"scope": "figure_renderer_contract_duplicate",
        "source_path": directory + "/" + contract_path, "sha256": digest(contract_bytes), "manifest_sha256": digest(manifest_bytes)})
    review_binding = manifest["visual_inspection"]
    require(review_binding.get("status") == "PASS" and review_binding.get("report") == "visual_review_v4.json", "Final visual inspection missing")
    review_bytes = safe_source(root, review_binding["report"]).read_bytes()
    require(digest(review_bytes) == review_binding["sha256"], "Visual inspection binding changed")
    review = read_json(review_bytes, "visual figure inspection")
    require(review.get("status") == "PASS" and review["source_script_sha256"] == manifest["source_script_sha256"], "Visual report did not inspect this plotting source")
    inventory = {contract_path: digest(contract_bytes), review_binding["report"]: digest(review_bytes)}
    csv_members = set()
    for figure in FIGURE_NAMES:
        entry = manifest["figures"][figure]
        require(entry["contract"] == contract["new_figures"][figure], "Figure design contract mismatch: " + figure)
        require(entry["source_script_sha256"] == manifest["source_script_sha256"] and abs(entry["width_mm"] - 160.0) < 0.01, "Figure source/physical-width binding changed")
        glyphs = entry["pdf_glyphs"]
        require(glyphs["verdict"] == "PASS" and glyphs["auditable"] and glyphs["minimum_found_pt"] >= 5 and glyphs["clipped_text_count"] == 0, "Figure glyph/clip gate failed")
        require(entry["alignment_verdict"] == ("NOT APPLICABLE" if figure == "fig_trace_workflow_v4" else "PASS"), "Figure alignment gate failed")
        require(entry["external_QA"]["collision"] == "PASS (strict)" and entry["external_QA"]["source_preflight"].startswith("PASS"), "Figure collision/source gate failed")
        require(entry["visual_review"] == review["figures"][figure] and entry["visual_review"]["status"] == "PASS", "Per-figure visual inspection differs")
        expected_names = {figure + suffix for suffix in FIGURE_SUFFIXES}
        require(set(entry["files_sha256"]) == expected_names, "Figure export/QA inventory changed: " + figure)
        for relative, sha in entry["files_sha256"].items():
            require(relative not in inventory, "Repeated figure member")
            inventory[relative] = sha_value(sha)
        for extension in ("pdf", "png"):
            require(entry["visual_review"][extension + "_sha256"] == inventory[figure + "." + extension], "Visual report refers to a different figure export")
        for relative, sha in entry["source_csv_sha256"].items():
            require(relative in FIGURE_CSV and relative not in csv_members and relative.startswith(figure + "_"), "Unselected/repeated figure source CSV")
            csv_members.add(relative)
            inventory["data/" + relative] = sha_value(sha)
    require(csv_members == FIGURE_CSV, "Final figure data inventory incomplete")
    records = []
    for relative, sha in sorted(inventory.items()):
        data = safe_source(root, relative).read_bytes()
        require(digest(data) == sha, "Final figure evidence changed: " + relative)
        target = FIGURE_PREFIX + relative
        payload.add(target, data, {"scope": "final_figure_evidence", "source_path": directory + "/" + relative,
                                 "sha256": sha, "manifest_sha256": digest(manifest_bytes)})
        records.append({"source": relative, "destination": target, "sha256": sha})
    payload.add(FIGURE_PREFIX + filename, manifest_bytes, {"scope": "final_figure_manifest_inventory", "sha256": digest(manifest_bytes)})
    code_records = add_code(payload, FIGURE_CODE, "figures")
    for record in code_records:
        name = record["path"]
        target = FIGURE_PREFIX + "source/" + PurePosixPath(name).name
        payload.add(target, payload.files[name], {"scope": "figure_manifest_bound_source_duplicate", "source_path": name,
                                                "sha256": record["sha256"], "manifest_sha256": digest(manifest_bytes)})
    readme = "\n".join([
        "# TRACE v4 figure evidence", "",
        "The three final figures and their four source CSVs retain the complete DiverseVul and MegaVul project rosters. The workflow is conceptual; the two quantitative figures use the fixed primary settings declared in `figure_contract_v4.json`.", "",
        "`figure_manifest_v4.json` binds the plotted CSVs, final PDF/SVG/PNG/TIFF exports, source preflight, physical geometry, glyph audit, alignment/collision reports and visual review. `visual_review_v4.json` documents inspection of the exact final PDF/PNG exports.", "",
        "The plotting source is available both in `source/` and at repository `src/make_kbs_strengthening_figures_v4.py`. Its unchanged v3 helper, `src/make_neural_validation_figures.py`, is included in `source/` for inspection. Both source hashes are recorded in the original figure manifest.", "",
        "The source reads the checked public aggregate packet through `--aggregate-dir`. Its manifest records both the original statistical analysis binding and the additional public-capacity-manifest binding. The complete four plotted CSVs also support direct data inspection.", "",
        "To render from the public repository tree, supply the external figure QA scripts directory:", "",
        "```", "python src/make_kbs_strengthening_figures_v4.py --aggregate-dir public_results/v4/capacity_feedback --audit-scripts PATH_TO_QA_SCRIPTS", "```", "",
        "The checked contract is also copied to `paper/neurocomputing_revision/figures/v4/figure_contract_v4.json`, the renderer's expected runtime path. The renderer writes new exports into that directory. The external `--audit-scripts` directory supplies `validate_figure.py`, `audit_panel_alignment.py`, `audit_pdf_text.py` and `audit_figure_collisions.py`; the skill/QA runtime is an explicit external dependency. Packaging verifies the saved export hashes and recorded QA gates without running experiments or rendering figures.", "",
    ])
    payload.add(FIGURE_PREFIX + "README.md", readme.encode("utf-8"), {"scope": "generated_figure_evidence_documentation"})
    return {"status": "READY", "public_directory": directory, "public_destination": FIGURE_PREFIX.rstrip("/"),
            "manifest": filename, "manifest_sha256": digest(manifest_bytes), "manifest_payload_members": len(records),
            "capacity_public_aggregate_manifest_sha256": manifest["capacity_public_aggregate_manifest_sha256"],
            "renderer_contract_repository_path": directory + "/" + contract_path,
            "member_mapping": records, "additional_repository_code": code_records, "source_csv_members": len(csv_members),
            "figures": list(FIGURE_NAMES), "saved_visual_geometry_alignment_glyph_and_collision_gates": "PASS",
            "figure_or_experiment_rerun": False}


def validate_licences(payload: Payload, modules: dict) -> dict:
    if modules.get("prom", {}).get("status") != "READY":
        return {"status": "PENDING_PROM", "v3_licence_assignment_changed": False}
    required = ("third_party/PROM/LICENSE", "third_party/PROM/src/LICENSE.txt")
    for name in required:
        require(name in payload.files and len(payload.files[name]) > 300, "PROM licence notice missing/truncated")
    root_notice = payload.files[required[0]].decode("utf-8-sig")
    src_notice = payload.files[required[1]].decode("utf-8-sig")
    require("Attribution 4.0 International" in root_notice and "MIT License" in src_notice and "Permission is hereby granted" in src_notice, "PROM original licence texts unexpected")
    source_files = [name for name in payload.files if name.startswith("third_party/PROM/src/prom/")]
    require(source_files and len(source_files) <= 200 and sum(len(payload.files[name]) for name in source_files) <= 5 * 1024 * 1024, "PROM source is missing or exceeds small-source boundary")
    require(all(name.endswith(".py") and "/tests/" not in name for name in source_files), "Unexpected PROM vendor/benchmark files")
    readme = payload.files.get("public_results/v4/prom/README.md", b"").decode("utf-8")
    require(all(author in readme for author in ("Huanting Wang", "Patrick Lenihan", "Zheng Wang")), "Upstream PROM attribution is missing")
    require("10.1145/3696443.3708959" in readme and "f16b52772123064551b6450f10b308971c4cdd39" in readme, "PROM source DOI/revision attribution incomplete")
    for name in ("third_party/PROM/prom_official_v4_binary_repair.diff", "third_party/PROM/prom_official_v4_pooled_count_optimization.diff"):
        require(name in payload.files and len(payload.files[name]) > 10, "PROM separate modification notice missing")
    return {"status": "PASS_ORIGINAL_NOTICES_RETAINED", "notices": [{"path": name, "sha256": digest(payload.files[name])} for name in required],
            "upstream_root_notice": "CC-BY-4.0", "upstream_src_notice": "MIT (original text retained)", "source_files": len(source_files),
            "source_bytes": sum(len(payload.files[name]) for name in source_files), "separate_repair_and_optimization_diffs": True,
            "new_software_or_data_licence_assigned": False}


def import_closure(payload: Payload) -> dict:
    local_stems = {PurePosixPath(name).stem for name in payload.files if name.startswith("src/") and name.endswith(".py")}
    missing = set()
    checked = 0
    for name, data in payload.files.items():
        if not name.endswith(".py") or not any(mark in name for mark in ("capacity_feedback_v4", "ratio_diagnostics_v4", "independent_model_v4", "prom_official_v4", "make_kbs_strengthening_figures_v4")):
            continue
        tree = ast.parse(data.decode("utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            candidates = ([node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else
                          [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in candidates:
                if module.startswith(("independent_model", "prepare_independent_model", "encode_independent_model", "train_independent_model", "predict_independent_model", "evaluate_independent_model", "reconstruct_independent_model", "audit_independent_model", "capacity_feedback", "analyze_ratio_diagnostics", "evaluate_neural_validation", "analyze_neurocomputing", "make_neural_validation_figures")):
                    if module not in local_stems:
                        missing.add((name, module))
                if module == "vultriage":
                    require("src/vultriage/__init__.py" in payload.files, "Shared v3 package missing")
        checked += 1
    require(not missing, "Missing local import dependencies: " + str(sorted(missing)))
    return {"new_code_files_ast_checked": checked, "missing_local_dependencies": [], "v3_shared_source_bytes_preserved": True}


def collect(allow_partial: bool, v3_path: Path):
    payload, v3 = collect_v3(v3_path)
    baseline_names = set(payload.files)
    modules = {}
    for module, (directory, filename) in MODULES.items():
        if not (ROOT / directory / filename).exists():
            modules[module] = {"status": "PENDING_FINAL_PUBLIC_MANIFEST", "public_directory": directory, "manifest": filename}
            require(allow_partial, "Final module export is not ready: " + module)
        else:
            modules[module] = collect_figures(payload) if module == "figures" else collect_module(payload, module)
    add_code(payload, ("scripts/build_trace_v4_public_release.py",), "packaging")
    general_protocol = ROOT / "protocol/kbs_strengthening_v4.md"
    if general_protocol.exists():
        add_code(payload, ("protocol/kbs_strengthening_v4.md",), "dated_v4_amendment")
    licences = validate_licences(payload, modules)
    imports = import_closure(payload)
    require(set(payload.files).issuperset(baseline_names), "v3 evidence disappeared")
    pending = sorted(name for name, record in modules.items() if record["status"] != "READY")
    return payload, baseline_names, {"status": "PARTIAL_NOT_RELEASEABLE" if pending else "COMPLETE_LOCAL_NOT_PUBLISHED", "pending_modules": pending,
        "v3": v3, "modules": modules, "licences": licences, "imports": imports,
        "safe_selection": "Published v3 bytes plus explicit final public manifests and named reproduction source only",
        "raw_rows_actions_events_predictions_weights_tokens_or_vaults_included": False,
        "discarded_independent_model_attempt_1_included": False, "new_or_old_experiments_rerun": False, "publication_performed": False}


def release_documentation(payload: Payload, information: dict):
    paragraphs = ["# TRACE v4 public research evidence", "", "TRACE evaluates risk, automatic-label allocation and cost under project shift. This increment adds analyst-capacity comparisons, delayed reviewed-label simulations, known-density diagnostics, the executed official PROM conformal component, and an independently trained CodeT5 encoder reference.", ""]
    interpretation = payload.files.get("public_results/v4/capacity_feedback/scientific_interpretation.md")
    if interpretation:
        paragraphs.extend([interpretation.decode("utf-8").strip(), ""])
    if information["pending_modules"]:
        paragraphs.extend(["This local preview is incomplete: final public evidence is still pending for " + ", ".join(information["pending_modules"]) + ". It is not a release candidate.", ""])
    paragraphs.extend(["The v3 public evidence, source, generated tables and figures retain their original bytes. New aggregate-only packets are under `public_results/v4/`; each preserves its own source manifest. The three final v4 figures, four source CSVs, plotting source/helper, contract, manifest and visual/geometry QA are under `public_results/kbs-validation-v4/figure-data-v4/`. Corrected CodeT5 uses attempt-2 and the v4.1 protocol. The older prepare/audit filenames are read-only helper dependencies, not the discarded training experiment.", "",
        "The full repository archive provides the complete checked staging tree; the increment archive supplies only additions to the published v3 research artifact. `TRACE_V4_RELEASE_MANIFEST.json` records every payload hash and the originating module manifests. The inherited `ARCHIVE_MANIFEST.json` describes the v3 subset only.", "",
        "Original v3 aggregate reconstruction remains available through its unchanged README and checkers. Capacity tables retain every fixed scenario and project, with ordinary cost undefined when work is unresolved. Ratio tables summarize independent simulation repetitions. PROM and CodeT5 provide aggregate reconstruction entrypoints; use the module README for its input paths and numerical-library versions. Full detector/row replay additionally requires independently acquired benchmark/model inputs described by the protocols.", "",
        "Packaging checks and archive verification are separate from experiments:", "", "```", "python scripts/build_trace_v4_public_release.py verify --tree .", "```", "",
        "No benchmark functions, per-function probabilities/actions/events, model weights, token caches, label vaults, author Word/photos, mock reviews or numerical vendor installations are supplied. PROM's original root and source licence notices, attribution and separate modification diffs are retained; this increment assigns no replacement software/data licence. Publication and manuscript submission are separate author actions.", ""])
    payload.add("TRACE_V4_PUBLIC_RELEASE.md", "\n".join(paragraphs).encode(), {"scope": "generated_v4_release_documentation"})
    metadata = {"schema_version": "trace-v4-public-release-metadata-1", "version": VERSION,
        "status": information["status"], "published": False, "pending_modules": information["pending_modules"],
        "known_previous_outcomes": True, "independent_model_selected_attempt": "attempt-2/v4.1",
        "datasets_analysed_separately": True, "analyst_cost_units": "illustrative", "real_feedback_arrival_measured": False,
        "v3_archive_sha256": V3_SHA, "v3_member_manifest_sha256": V3_MANIFEST_SHA,
        "module_manifest_hashes": {name: record.get("manifest_sha256") for name, record in information["modules"].items()},
        "licences": information["licences"], "no_new_software_or_data_licence_assigned": True}
    payload.add("TRACE_V4_PUBLIC_RELEASE_METADATA.json", dump_bytes(metadata), {"scope": "generated_v4_release_metadata"})


def archive_bytes(files: dict[str, bytes]) -> bytes:
    require(len({name.casefold() for name in files}) == len(files), "Archive Windows member collision")
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        for name, data in sorted(files.items()):
            safe_member(name)
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return memory.getvalue()


def verify_zip(path_or_bytes, *, expected: dict[str, str] | None = None) -> dict:
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with zipfile.ZipFile(source) as archive:
        require(archive.testzip() is None, "v4 ZIP CRC failure")
        names = archive.namelist()
        require(len(names) == len(set(names)) == len({name.casefold() for name in names}), "Duplicate/case-colliding ZIP members")
        manifest_data = archive.read(V4_MANIFEST)
        manifest = read_json(manifest_data, "v4 release manifest")
        if expected is None:
            expected = manifest["files_sha256"]
        require(set(names) == set(expected) | {V4_MANIFEST}, "v4 ZIP inventory differs from its manifest")
        for info in archive.infolist():
            safe_member(info.filename)
            require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlink member excluded")
            require(info.date_time == ZIP_DATE and info.create_system == 3 and info.external_attr == (stat.S_IFREG | 0o644) << 16
                    and info.compress_type == zipfile.ZIP_DEFLATED and not info.extra and not info.comment, "ZIP metadata is not deterministic")
            data = archive.read(info.filename)
            validate_data(info.filename, data)
            if info.filename != V4_MANIFEST:
                require(digest(data) == sha_value(expected[info.filename]), "ZIP member hash mismatch: " + info.filename)
        return {"status": "PASS", "payload_members": len(expected), "archive_manifest_sha256": digest(manifest_data),
                "archive_scope": manifest["archive_scope"], "paths_hashes_crc_permissions_and_private_exclusions_verified": True}


def write_new(path: Path, data: bytes):
    require(path.resolve().is_relative_to(REPORT_ROOT.resolve()), "Packaging output escaped delegated report directory")
    require(not path.is_symlink(), "Cannot write through symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def package_manifest(payload: Payload, information: dict, scope: str, names: set[str]):
    return {"schema_version": "trace-v4-manifest-bound-public-repository-1", "version": VERSION,
        "status": information["status"], "archive_scope": scope, "zip_timestamp": list(ZIP_DATE),
        "archive_kind": "increment" if scope.startswith("incremental additions only;") else "full",
        "sha256_scope": "All payload members; this manifest is not self-hashed", "files_sha256": {name: digest(payload.files[name]) for name in sorted(names)},
        "origins": {name: payload.origins[name] for name in sorted(names)},
        "v3_base": information["v3"], "modules": information["modules"], "pending_modules": information["pending_modules"],
        "licences": information["licences"], "checks": information["imports"],
        "aggregate_only_new_evidence": True, "discarded_model_attempt_1_excluded": True,
        "no_experiment_rerun": True, "no_publication_performed": True, "builder_sha256": file_hash(Path(__file__))}


def tree_inventory(manifest: dict, legacy: dict, legacy_sha: str) -> dict[str, str]:
    expected = dict(manifest["files_sha256"])
    kind = manifest.get("archive_kind", "increment" if manifest["archive_scope"].startswith("incremental additions only;") else "full")
    require(kind in {"full", "increment"}, "Unknown v4 archive kind")
    if kind == "increment":
        inherited = {**legacy["files_sha256"], "ARCHIVE_MANIFEST.json": legacy_sha}
        require(not set(expected).intersection(inherited), "Increment tries to overwrite original v3 members")
        expected = {**inherited, **expected}
    return expected


def verify_tree(tree: Path) -> dict:
    tree = tree.resolve(strict=True)
    require(tree.is_dir() and not tree.is_symlink(), "Repository tree must be a real directory")
    manifest_data = (tree / V4_MANIFEST).read_bytes()
    manifest = read_json(manifest_data, "staging manifest")
    legacy_bytes = (tree / "ARCHIVE_MANIFEST.json").read_bytes()
    require(digest(legacy_bytes) == V3_MANIFEST_SHA, "Inherited v3 subset index changed")
    legacy = read_json(legacy_bytes, "v3 subset manifest")
    expected = tree_inventory(manifest, legacy, digest(legacy_bytes))
    actual = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file()}
    require(actual == set(expected) | {V4_MANIFEST}, "Unexpected/missing staging members")
    for name, sha in expected.items():
        path = safe_source(tree, name)
        data = path.read_bytes()
        require(digest(data) == sha_value(sha), "Staging file changed: " + name)
        validate_data(name, data)
    # Verify every legacy v3 payload independently in its inherited subset manifest.
    for name, sha in legacy["files_sha256"].items():
        require(file_hash(safe_source(tree, name)) == sha, "A historical v3 evidence/source byte changed")
    return {"status": "PASS", "staging_members": len(expected) + 1, "preserved_v3_payload_members": len(legacy["files_sha256"]),
            "manifest_sha256": digest(manifest_data), "archive_kind": manifest.get("archive_kind", "full"),
            "private_file_exclusions_and_all_hashes_verified": True}


def verify_increment_application(baseline: dict[str, bytes], increment: bytes, full: dict[str, bytes]) -> dict:
    require(digest(baseline["ARCHIVE_MANIFEST.json"]) == V3_MANIFEST_SHA, "Increment base is not the preserved v3 inventory")
    legacy = read_json(baseline["ARCHIVE_MANIFEST.json"], "increment application base")
    require(set(baseline) == set(legacy["files_sha256"]) | {"ARCHIVE_MANIFEST.json"}, "Increment base inventory differs from v3")
    for name, sha in legacy["files_sha256"].items():
        require(digest(baseline[name]) == sha_value(sha), "Increment application base byte changed")
    verify_zip(increment)
    with zipfile.ZipFile(io.BytesIO(increment)) as archive:
        manifest = read_json(archive.read(V4_MANIFEST), "increment application manifest")
        require(manifest.get("archive_kind") == "increment" and manifest["v3_base"]["archive_sha256"] == V3_SHA, "Increment is not bound to the published v3 base")
        additions = {name: archive.read(name) for name in archive.namelist() if name != V4_MANIFEST}
    require(not set(additions).intersection(baseline), "Increment contains an overwrite of the historical artifact")
    applied = {**baseline, **additions}
    expected = tree_inventory(manifest, legacy, V3_MANIFEST_SHA)
    require(set(applied) == set(expected) == set(full), "Applying the increment gives a different complete inventory")
    for name, data in applied.items():
        require(data == full[name] and digest(data) == sha_value(expected[name]), "Increment application differs from the full artifact: " + name)
    return {"status": "PASS", "v3_archive_sha256": V3_SHA, "base_payload_and_inherited_manifest_members": len(baseline),
            "increment_payload_members": len(additions), "complete_payload_members": len(applied),
            "all_payload_bytes_equal_full_repository_archive": True, "no_original_member_overwritten": True,
            "increment_manifest_resolves_full_tree_verification": True}


def build(output: Path, allow_partial: bool, v3_path: Path):
    output = output.resolve()
    require(output.is_relative_to(REPORT_ROOT.resolve()) and output != REPORT_ROOT.resolve(), "Choose a new build directory beneath packaging/")
    require(not output.exists(), "Build output must be new; original reports/staging are immutable")
    payload, baseline_names, information = collect(allow_partial, v3_path)
    release_documentation(payload, information)
    all_names = set(payload.files)
    delta_names = all_names - baseline_names
    full_manifest = dump_bytes(package_manifest(payload, information, "full repository tree; original v3 plus v4 additions", all_names))
    delta_manifest = dump_bytes(package_manifest(payload, information, "incremental additions only; apply to the hash-matched v3 research artifact", delta_names))
    tree = output / "repository"
    for name, data in payload.files.items():
        write_new(tree / name, data)
    write_new(tree / V4_MANIFEST, full_manifest)
    prefix = "TRACE_v4_preview" if information["pending_modules"] else "TRACE_v4_public"
    full_bytes = archive_bytes({**payload.files, V4_MANIFEST: full_manifest})
    delta_bytes = archive_bytes({**{name: payload.files[name] for name in delta_names}, V4_MANIFEST: delta_manifest})
    full_name, delta_name = prefix + "_repository.zip", prefix + "_increment.zip"
    full_check, delta_check = verify_zip(full_bytes), verify_zip(delta_bytes)
    increment_check = verify_increment_application({name: payload.files[name] for name in baseline_names}, delta_bytes, payload.files)
    write_new(output / full_name, full_bytes)
    write_new(output / delta_name, delta_bytes)
    tree_check = verify_tree(tree)
    write_new(output / "staging_paths.nul", b"\0".join(name.encode() for name in sorted(all_names | {V4_MANIFEST})) + b"\0")
    write_new(output / "staging_members.sha256", "".join(digest(payload.files[name]) + "  " + name + "\n" for name in sorted(all_names)).encode())
    receipt = {"status": information["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": file_hash(Path(__file__)), "output": str(output), "staging_tree": str(tree),
        "members": len(all_names), "increment_members": len(delta_names), "v3_payload_members_preserved": len(baseline_names) - 1,
        "archives": [{"path": str(output / full_name), "sha256": digest(full_bytes), "bytes": len(full_bytes), **full_check},
                     {"path": str(output / delta_name), "sha256": digest(delta_bytes), "bytes": len(delta_bytes), **delta_check}],
        "staging_verification": tree_check, "input_bindings": information,
        "increment_application_verification": increment_check,
        "new_or_old_experiments_rerun": False, "github_or_journal_publication_performed": False,
        "source_manuscript_and_existing_experimental_artifacts_modified": False}
    write_new(output / "build_receipt.json", dump_bytes(receipt))
    return receipt


def self_test():
    checks = []
    bad_names = ["../outside.py", "/absolute.py", "C:/outside.py", "a\\b.py", "a/../b.py", "a//b.py", "datasets/data.csv",
        "predictions/example.json", "weights/model.pt", "tokens/array.npy", "vault/labels.json", "author_portraits/photo.png",
        "manuscript.docx", "mock_reviews/report.md", "runtime-dependencies/numpy/core.py", "vendor/scipy.py", "attempt-1/config.json",
        "src/train_independent_model_v4.py", "data/target_rows.csv", "events/label_events.csv",
        "private/fig_capacity_v4.alignment.svg", FIGURE_PREFIX + "custom.alignment.svg", FIGURE_PREFIX + "fig_capacity_v4.extra.collision-audit.json"]
    for name in bad_names:
        try:
            safe_member(name)
        except ValueError:
            checks.append("reject_path:" + name)
        else:
            raise AssertionError("Private/traversing member accepted: " + name)
    for data in (b"row_id,label,prediction\na,1,0.3\n", b"project,service_batch\na,2\n"):
        try:
            validate_data("public_results/v4/test/aggregate.csv", data)
        except ValueError:
            checks.append("reject_raw_csv")
        else:
            raise AssertionError("Raw per-function CSV accepted")
    for node in ({"target_positions": [0, 1]}, {"labels": [0, 1, 0]}, {"raw_target_weights": [1, 2]}, {"fit_window_indices": [1]}, {"rows": [{"row_id": "x", "func": "code"}]}):
        try:
            validate_data("public_results/v4/test/report.json", dump_bytes(node))
        except ValueError:
            checks.append("reject_raw_json")
        else:
            raise AssertionError("Raw row/vector JSON accepted")
    validate_data("public_results/v4/test/source_summary.json", dump_bytes({"labels": {"0": 100, "1": 12}, "source_rows_sha256": "A" * 64}))
    validate_data("public_results/v4/test/domain_summary.json", dump_bytes([{"project": "a", "source_n": 100, "target_n": 200, "domain_auc": .7}]))
    validate_data("public_results/v4/test/project.csv", b"project,n_vulnerable,n_safe,cost\na,30,200,0.25\n")
    checks.extend(["accept_aggregate_class_counts", "accept_top_level_project_diagnostic_list", "accept_project_denominators"])
    for name in ("fig_capacity_v4.alignment.json", FIGURE_PREFIX + "fig_feedback_v4.alignment.svg", FIGURE_PREFIX + "fig_trace_workflow_v4.collision-audit.json"):
        safe_member(name)
    validate_data(FIGURE_PREFIX + "fig_capacity_v4.geometry.json", dump_bytes({"figure_size_pt": [453.5, 257.9], "texts": [{"text": "Review capacity", "bbox_pt": [10, 20, 30, 40]}]}))
    checks.extend(["accept_exact_named_figure_qa_members", "accept_figure_annotation_geometry"])
    for value in ("F:/workspace/figure.pdf", "F:\\workspace\\figure.pdf", "/home/researcher/figure.pdf", "\\\\server\\private\\figure.pdf"):
        try:
            validate_data(FIGURE_PREFIX + "fig_capacity_v4.collision-audit.json", dump_bytes({"pdf": value}))
        except ValueError:
            checks.append("reject_public_absolute_machine_path")
        else:
            raise AssertionError("Absolute machine path entered public JSON")
    try:
        validate_data(FIGURE_PREFIX + "fig_capacity_v4.geometry.json", dump_bytes({"target_positions": [0, 1]}))
    except ValueError:
        checks.append("figure_qa_keeps_raw_vector_exclusion")
    else:
        raise AssertionError("Figure QA exception admitted per-function vectors")
    a = Payload({}, {})
    a.add("src/one.py", b"x=1\n", {})
    try:
        a.add("src/one.py", b"x=2\n", {})
    except ValueError:
        checks.append("reject_shared_source_overwrite")
    else:
        raise AssertionError("Scientific bytes overwritten")
    try:
        a.add("src/ONE.py", b"x=1\n", {})
    except ValueError:
        checks.append("reject_windows_case_collision")
    else:
        raise AssertionError("Case-colliding archive members accepted")
    tiny_files = {"README.md": b"Aggregate-only test fixture\n"}
    tiny_manifest = {"archive_scope": "synthetic test fixture", "files_sha256": {name: digest(data) for name, data in tiny_files.items()}}
    package = {**tiny_files, V4_MANIFEST: dump_bytes(tiny_manifest)}
    first, second = archive_bytes(package), archive_bytes(package)
    require(first == second, "ZIP output is nondeterministic")
    verify_zip(first)
    checks.extend(["deterministic_zip_bytes", "zip_crc_paths_hashes_permissions"])
    tampered = {**package, "README.md": b"changed"}
    try:
        verify_zip(archive_bytes(tampered))
    except ValueError:
        checks.append("reject_zip_hash_tamper")
    else:
        raise AssertionError("Tampered archive accepted")
    inherited = {"files_sha256": {"README.md": digest(b"Original public fixture\n")}}
    inherited_sha = digest(dump_bytes(inherited))
    increment_index = {"archive_kind": "increment", "archive_scope": "incremental additions only; synthetic fixture",
                       "files_sha256": {"public_results/v4/test/summary.json": digest(b"{}\n")}}
    combined = tree_inventory(increment_index, inherited, inherited_sha)
    require(set(combined) == {"README.md", "ARCHIVE_MANIFEST.json", "public_results/v4/test/summary.json"}
            and combined["README.md"] == inherited["files_sha256"]["README.md"], "Increment tree inventory failed to retain the v3 base")
    checks.append("increment_tree_inventory_resolves_original_plus_additions")
    for node in ({**increment_index, "files_sha256": {"README.md": digest(b"replacement")}},
                 {**increment_index, "archive_kind": "unknown"}):
        try:
            tree_inventory(node, inherited, inherited_sha)
        except ValueError:
            checks.append("reject_increment_base_overwrite_or_unknown_scope")
        else:
            raise AssertionError("Invalid incremental tree manifest accepted")
    return {"status": "PASS", "checks": checks, "check_count": len(checks), "builder_sha256": file_hash(Path(__file__)),
            "experiments_rerun": False, "fixtures": "in-memory only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "inspect", "build", "verify"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--v3-archive", type=Path, default=V3_ZIP)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "complete-1")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
    elif args.command == "inspect":
        payload, baseline, information = collect(args.allow_partial, args.v3_archive)
        result = {**information, "payload_members": len(payload.files), "baseline_members": len(baseline),
                  "source_snapshot_sha256": {name: digest(data) for name, data in sorted(payload.files.items())}}
    elif args.command == "build":
        result = build(args.output, args.allow_partial, args.v3_archive)
    else:
        require(args.tree or args.archive, "Specify --tree and/or --archive for read-only verification")
        result = {"tree": verify_tree(args.tree) if args.tree else None, "archive": verify_zip(args.archive) if args.archive else None}
    if args.report:
        write_new(args.report.resolve(), dump_bytes(result))
    shown = {k: v for k, v in result.items() if k not in ("source_snapshot_sha256", "input_bindings", "modules")}
    print(json.dumps(shown, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(json.dumps({"status": "FAILED_CLOSED", "reason": str(error)}, indent=2, ensure_ascii=False))
        raise SystemExit(1)
