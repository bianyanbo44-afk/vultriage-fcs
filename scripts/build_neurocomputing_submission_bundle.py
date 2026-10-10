"""Build deterministic, manifest-bound Neurocomputing submission archives.

Only explicit source files, aggregate releases and textual QA receipts are read.
The experiment tree is accessed solely at its frozen execution snapshot, never
at datasets, token caches, checkpoints, predictions, decisions or label vaults.
Use --describe-qa for the final QA contract, --print-source-sha256 for its source
map, or --validate-only to check all inputs without writing any archives.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/neurocomputing_revision"
OUT = PAPER / "submission"
QA = ROOT / "tmp/neurocomputing-strengthening/final_qa.json"
EXECUTION = ROOT / "outputs/neural-validation-v3/frozen_scripts/execution"
SOURCE_NAME = "TRACE_Neurocomputing_manuscript_source.zip"
CODE_NAME = "TRACE_supplementary_code_and_tables.zip"
FULL_NAME = "TRACE_Neurocomputing_submission.zip"
ZIP_DATE = (2026, 10, 10, 0, 0, 0)
MAIN_FIGURES = ("fig_workflow", "fig_neural_allocation", "fig_neural_costs")
SI_FIGURES = ("fig_matched_tradeoff", "fig_cost_sensitivity")
PDF_NAMES = ("main", "supplementary", "cover_letter")
TABLE_NAMES = ("neural_validation_main_tables.tex", "neural_validation_supplement_tables.tex")
ACTUAL_FITS = {
    "finetuned_codebert": [13, 37, 73],
    "hashing_sgd": [13, 37, 73],
    "frozen_codebert": [13],
}
LEGACY_PUBLIC_MANIFEST_SHA256 = "070101BC6E277567552107C2533DCCE9D5BB02856E3E07607CFDB155189C6454"
LEGACY_FIGURE_MANIFEST_SHA256 = "31FE628B84667CC04FFDA1499CB2DC22AD0D2697836D81CF483256CE44156D5E"
EXECUTION_MANIFEST_SHA256 = "3D2B8A7ED3FA772326FE42C86C0C8B89A6593B3587FFEE75DE1D7E96A4EF4EB8"
LEGACY_AGGREGATES = frozenset({
    "action_metrics_project.csv", "action_metrics_seed.csv",
    "action_paired_comparisons.csv", "action_summary.csv",
    "cost_paired_sensitivity.csv", "gate_audit.csv", "gate_project_scores.csv",
    "nominal_cost_summary.csv", "analysis_manifest.json",
    "controls/action_metrics_project.csv", "controls/action_metrics_seed.csv",
    "controls/action_summary.csv", "controls/cost_summary.csv",
    "controls/matched_cost_comparisons.csv", "controls/matched_paired_comparisons.csv",
    "controls/selection_invariants_seed.csv", "controls/source_calibration_fits.csv",
    "controls/analysis_manifest.json",
})
EXECUTED_FILES = frozenset({
    "src/prepare_neural_validation_v3.py", "src/prepare_megavul_validation_v3.py",
    "src/encode_neural_validation_v3.py", "src/train_neural_validation_v3.py",
    "src/predict_neural_validation_v3.py", "src/evaluate_neural_validation_v3.py",
    "src/vultriage/neural_audit.py", "src/vultriage/conformal.py",
    "src/vultriage/domain.py", "src/vultriage/data.py", "src/vultriage/metrics.py",
    "src/analyze_neurocomputing_controls.py", "src/analyze_neurocomputing_revision.py",
    "configs/neural_validation_v3.json", "configs/megavul_validation_v3.json",
})
PAPER_SOURCE_FILES = (
    "main.tex", "main.bbl", "supplementary.tex", "supplementary.bbl",
    "supplementary_tables.tex", *TABLE_NAMES, "cover_letter.tex",
    "highlights.txt", "README_NEUROCOMPUTING.md", "REVISION_NOTES.md", "references.bib",
    "elsarticle.cls", "elsarticle-num-names.bst",
    "neural_allocation_display.tex", "neural_sensitivity_display.tex",
    "author_biographies.docx", "declaration_of_competing_interests.docx",
    "reproducibility/validation_runtime_v3.json",
    "repository_deposit/README_DEPOSIT.md", "repository_deposit/DATA_DICTIONARY.md",
    "repository_deposit/repository_metadata_draft.json",
)
CURRENT_SOURCE_FILES = (
    "src/analyze_neurocomputing_revision.py", "src/analyze_neurocomputing_controls.py",
    "src/make_neurocomputing_audit_figures.py", "src/make_neural_validation_figures.py",
    "src/run_extension_predict.py", "src/prepare_feature_cache.py",
    "src/prepare_extension_manifest.py", "src/prepare_extension_inputs.py",
    "src/audit_near_duplicates_v2.py", "src/complete_near_duplicates_v2.py",
    "src/prepare_neural_validation_v3.py", "src/prepare_megavul_validation_v3.py",
    "src/encode_neural_validation_v3.py", "src/train_neural_validation_v3.py",
    "src/predict_neural_validation_v3.py", "src/evaluate_neural_validation_v3.py",
    "scripts/download_megavul_v3.py", "scripts/export_neural_validation_evidence.py",
    "scripts/audit_label_free_allocation_v3.py",
    "scripts/check_neural_aggregate_evidence.py", "scripts/check_neural_prediction_lineage.py",
    "scripts/check_neurocomputing_revision.py", "scripts/build_neurocomputing_submission_bundle.py",
    "scripts/build_neurocomputing_supplement_tables.py", "scripts/render_neurocomputing_revision.py",
    "scripts/build_neural_display_additions.py", "requirements-validation-v3-lock.txt",
    "requirements.txt", "requirements-neural.txt", "configs/preregistered_experiment.json",
    "configs/preregistered_extension_v2.json", "configs/neural_validation_v3.json",
    "configs/megavul_validation_v3.json", "protocol/neurocomputing_controls_v2.md",
    "protocol/neural_validation_v3.md", "protocol/near_duplicate_audit_v2.md",
    "protocol/extension_codebert_execution_amendment_v2.md",
    "tests/test_neuro_controls.py", "tests/test_neuro_operational_audit.py",
    "tests/test_neural_validation_v3.py", "tests/test_megavul_preparation.py",
)
LEGACY_PROVENANCE_FILES = (
    "evaluation-v2/evaluation_manifest.json", "evaluation-v2/fold_seed_metrics.csv",
    "contract-v1/contract_manifest.json", "contract-v1/qualification_summary.csv",
    "contract-v1/fixed_split_projects.csv", "gate-v1/gate_seal.json",
    "gate-v1/primevul_gate_development.csv",
)
ALLOWED_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".json", ".csv", ".tex", ".bbl", ".bib",
    ".cls", ".bst", ".pdf", ".svg", ".png", ".tif", ".tiff", ".ini", ".docx",
})
PRIVATE_PARTS = frozenset({
    ".git", ".codex", "skills", "__pycache__", "node_modules", "raw",
    "raw_data", "raw_datasets", "datasets", "vault", "vaults", "label_vault",
    "target_labels", "tokens", "token_cache", "token_caches", "cache", "caches",
    "checkpoints", "checkpoint", "weights", "predictions", "decisions",
    "credentials", "secrets",
})
AGGREGATE_DECISION_COUNT_MEMBERS = frozenset(
    f"{prefix}{dataset}/label_free_decision_counts.csv"
    for dataset in ("diversevul", "megavul")
    for prefix in ("", "analysis/neural_validation_v3/",
                   "paper/neurocomputing_revision/analysis/neural_validation_v3/",
                   "tmp/neurocomputing-strengthening/blind_review_v3/packet/paper/neurocomputing_revision/analysis/neural_validation_v3/")
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        require(name not in result, f"Repeated JSON key: {name}")
        result[name] = value
    return result


def read_json_bytes(data: bytes, label: str) -> dict:
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
    require(isinstance(value, dict), f"Expected JSON object: {label}")
    return value


def hash_value(value: object, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
            f"Invalid SHA-256: {label}")
    return value.upper()


def hash_map(value: object, label: str) -> dict[str, str]:
    require(isinstance(value, dict) and bool(value), f"Missing SHA-256 map: {label}")
    return {name: hash_value(expected, f"{label}/{name}") for name, expected in value.items()}


def safe_member(name: str, *, nested_zip: bool = False) -> str:
    require(isinstance(name, str) and bool(name), "Empty archive member")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and not any(part in {"", ".", ".."} for part in name.split("/"))
            and not any(char in name for char in ("\\", ":", "\0")),
            f"Unsafe archive member: {name}")
    require(name != "ARCHIVE_MANIFEST.json", "Reserved archive manifest member")
    normalized = {part.lower().replace("-", "_") for part in path.parts}
    require(not normalized.intersection(PRIVATE_PARTS), f"Private/runtime member excluded: {name}")
    lower = name.lower()
    require(path.name.lower() not in {"skill.md", ".env"}
            and not path.name.lower().startswith(".env.")
            and not any(mark in lower for mark in (".alignment.", ".collision.", ".overlay.")),
            f"Private/QA-art member excluded: {name}")
    allowed = ALLOWED_SUFFIXES | ({".zip"} if nested_zip else set())
    # This publisher-page capture is checked inside the immutable review packet.
    # It is never selected for either journal-source or code archive.
    if name == "tmp/neurocomputing-strengthening/blind_review_v3/packet/tmp/neurocomputing-review-v2/citation-metadata/codebert_acl.html":
        allowed = allowed | {".html"}
    require(path.suffix.lower() in allowed, f"Unsupported/nonaggregate member excluded: {name}")
    if path.suffix.lower() not in {".py", ".md"}:
        require(name in AGGREGATE_DECISION_COUNT_MEMBERS or not re.search(r"(?:^|[_-])(row_ids|running_examples|target_positions|target_rows|source_rows|"
                              r"source_train|source_calibration_rows|source_model_validation_rows|"
                              r"prediction_rows|row_predictions?|decisions?)(?:$|[_-])", path.stem.lower()),
                f"Row-level member excluded: {name}")
        prediction_name = re.search(r"(?:^|[_-])predictions?(?:$|[_-])", path.stem.lower())
        lineage_receipt = path.suffix.lower() == ".json" and re.search(
            r"(?:^|[_-])predictions?[_-]lineage(?:$|[_-])", path.stem.lower()) is not None
        require(not prediction_name or lineage_receipt,
                f"Prediction-row member excluded: {name}")
    return name


def local_path(base: Path, name: str) -> Path:
    safe_member(name)
    root = base.resolve()
    try:
        path = (base / name).resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"Required bundle artifact is missing: {(base / name).as_posix()}") from None
    require(path.is_relative_to(root) and path.is_relative_to(ROOT.resolve()) and path.is_file(),
            f"Source escapes its allowed directory or is not a file: {name}")
    return path


def file_bytes(base: Path, name: str, expected: str | None = None) -> bytes:
    data = local_path(base, name).read_bytes()
    if expected is not None:
        require(digest(data) == hash_value(expected, name), f"Stale/changed source bytes: {name}")
    return data


def add(entries: dict[str, bytes], name: str, data: bytes, *, nested_zip: bool = False) -> None:
    safe_member(name, nested_zip=nested_zip)
    require(name not in entries, f"Repeated archive member: {name}")
    entries[name] = data


def tex_without_comments(source: bytes) -> str:
    text = source.decode("utf-8-sig")
    lines = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char == "%":
                slashes = len(line[:index]) - len(line[:index].rstrip("\\"))
                if slashes % 2 == 0:
                    line = line[:index]
                    break
        lines.append(line)
    return "\n".join(lines)


def extract_title(source: bytes) -> str:
    """Read a balanced title argument, ignoring comments and formatting macros."""
    text = tex_without_comments(source)
    matches = list(re.finditer(r"\\title\s*(?:\[[^\]]*\]\s*)?\{", text))
    require(len(matches) == 1, "main.tex must contain one explicit title argument")
    start = matches[0].end()
    depth, end = 1, start
    while end < len(text) and depth:
        char = text[end]
        slashes = len(text[:end]) - len(text[:end].rstrip("\\"))
        if slashes % 2 == 0:
            depth += (char == "{") - (char == "}")
        end += 1
    require(depth == 0, "Unclosed title argument in main.tex")
    title = text[start:end - 1]
    title = re.sub(r"\\(?:textbf|textit|textsc|textrm|textnormal|emph)\s*", "", title)
    title = re.sub(r"\\(?:protect|normalfont)\b\s*", "", title)
    for command, replacement in (("LaTeX", "LaTeX"), ("TeX", "TeX"),
                                 ("textendash", "–"), ("textemdash", "—")):
        title = re.sub(r"\\" + command + r"\b\s*", replacement, title)
    title = re.sub(r"\\([&%_$#{}])", r"\1", title)
    title = title.replace("\\\\", " ").replace("~", " ")
    require("\\" not in title, "Unsupported title macro; supply an explicit readable title")
    title = re.sub(r"[{}]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    require(bool(title), "Empty manuscript title")
    return title


def main_source_dependencies(source: dict[str, bytes], paper_sources: dict[str, bytes]) -> None:
    """Require an editable main archive with its actual table/figure dependencies."""
    pending, checked, referenced_figures = ["main.tex"], set(), set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        text = tex_without_comments(source[name])
        for dependency in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", text):
            filename = dependency.strip()
            if not PurePosixPath(filename).suffix:
                filename += ".tex"
            safe_member(filename)
            require(filename in paper_sources, f"Unselected main TeX dependency: {filename}")
            source[filename] = paper_sources[filename]
            pending.append(filename)
        for dependency in re.findall(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}", text):
            filename = dependency.strip()
            if not PurePosixPath(filename).suffix:
                filename += ".pdf"
            if "/" not in filename:
                filename = "figures/" + filename
            safe_member(filename)
            require(filename in source, f"Unselected main figure dependency: {filename}")
            referenced_figures.add(filename)
    require(referenced_figures == {f"figures/{name}.pdf" for name in MAIN_FIGURES},
            "main.tex must reference exactly the three current main figures")


@dataclass
class Inputs:
    title: str
    source: dict[str, bytes]
    code: dict[str, bytes]
    paper_sources: dict[str, bytes]
    neural_manifest: dict
    neural_manifest_sha256: str
    execution_manifest_sha256: str

    def source_sha256(self) -> dict[str, str]:
        return {name: digest(data) for name, data in sorted(self.paper_sources.items())}


def collect_inputs() -> Inputs:
    paper_sources = {name: file_bytes(PAPER, name) for name in PAPER_SOURCE_FILES}
    code = {name: file_bytes(ROOT, name) for name in CURRENT_SOURCE_FILES}
    for path in sorted((ROOT / "src/vultriage").glob("*.py")):
        name = path.relative_to(ROOT).as_posix()
        require(name not in code, f"Duplicate current source: {name}")
        code[name] = file_bytes(ROOT, name)

    legacy_name = "analysis/public_analysis_manifest.json"
    legacy_bytes = file_bytes(PAPER, legacy_name, LEGACY_PUBLIC_MANIFEST_SHA256)
    legacy = read_json_bytes(legacy_bytes, legacy_name)
    legacy_hashes = hash_map(legacy.get("files_sha256"), legacy_name)
    require(set(legacy_hashes) == LEGACY_AGGREGATES, "Legacy release must retain exactly its 18 frozen members")
    require(legacy.get("upstream_manifests_unchanged") is True, "Legacy upstream manifests must remain unchanged")
    for name, expected in sorted(legacy_hashes.items()):
        paper_sources[f"analysis/{name}"] = file_bytes(PAPER / "analysis", name, expected)
    paper_sources[legacy_name] = legacy_bytes
    for name, expected in hash_map(legacy.get("analysis_source_sha256"), "legacy analysis sources").items():
        require(name in code and digest(code[name]) == expected, f"Frozen legacy analysis source changed: {name}")

    neural_base = PAPER / "analysis/neural_validation_v3"
    neural_bytes = file_bytes(neural_base, "public_analysis_manifest.json")
    neural = read_json_bytes(neural_bytes, "neural public release")
    require(neural.get("schema_version") == "trace-v3-aggregate-release-1"
            and neural.get("no_raw_datasets_predictions_weights_or_vaults") is True,
            "Missing aggregate-only neural release declaration")
    hashes = hash_map(neural.get("files_sha256"), "neural release")
    required_receipts = {f"training/{detector}_seed-{seed}.json"
                         for detector, seeds in ACTUAL_FITS.items() for seed in seeds}
    require({name for name in hashes if name.startswith("training/")} == required_receipts,
            "Neural release must include exactly the seven actual selected-fit receipts")
    datasets = neural.get("datasets")
    require(isinstance(datasets, dict) and set(datasets) == {"diversevul", "megavul"},
            "Neural release must distinguish both target datasets")
    for name, receipt in datasets.items():
        require(receipt.get("actual_seeds") == ACTUAL_FITS, f"Unexpected actual fits for {name}")
        hash_value(receipt.get("decision_seal_sha256"), f"{name} decision seal")
    actual_inventory = {p.relative_to(neural_base).as_posix()
                        for p in neural_base.rglob("*") if p.is_file()}
    require(actual_inventory == set(hashes) | {"public_analysis_manifest.json"},
            "Neural aggregate directory differs from its exact public manifest inventory")
    for name, expected in sorted(hashes.items()):
        paper_sources[f"analysis/neural_validation_v3/{name}"] = file_bytes(neural_base, name, expected)
    paper_sources["analysis/neural_validation_v3/public_analysis_manifest.json"] = neural_bytes
    table_hashes = hash_map(neural.get("table_source_sha256"), "neural table sources")
    require(set(table_hashes) == set(TABLE_NAMES), "Both generated neural TeX tables must be manifest-bound")
    for name, expected in table_hashes.items():
        require(digest(paper_sources[name]) == expected, f"Changed generated neural table: {name}")

    display_name = "display_additions/manifest.json"
    display_bytes = file_bytes(PAPER, display_name)
    display = read_json_bytes(display_bytes, "post-review compact displays")
    require(display.get("schema_version") == "trace-neural-display-additions-1"
            and display.get("original_release_or_generated_tables_modified") is False,
            "Missing separately bound post-review display declaration")
    require(hash_value(display.get("public_analysis_manifest_sha256"), "display release") == digest(neural_bytes)
            and hash_value(display.get("source_script_sha256"), "display source")
            == digest(code["scripts/build_neural_display_additions.py"]), "Compact display provenance is stale")
    display_hashes = hash_map(display.get("files_sha256"), "compact display files")
    require(set(display_hashes) == {"neural_allocation_display.tex", "neural_sensitivity_display.tex",
                                  "display_additions/action_fractions.csv",
                                  "display_additions/allocation_sensitivity.csv",
                                  "display_additions/cost_grid_summary.csv"}, "Unexpected compact display inventory")
    for name, expected in display_hashes.items():
        data = file_bytes(PAPER, name, expected)
        if name in paper_sources:
            require(paper_sources[name] == data, "Compact display source mismatch")
        else:
            paper_sources[name] = data
    paper_sources[display_name] = display_bytes

    execution_bytes = file_bytes(EXECUTION, "execution_manifest.json", EXECUTION_MANIFEST_SHA256)
    execution = read_json_bytes(execution_bytes, "frozen execution manifest")
    executed_hashes = hash_map(execution.get("files_sha256"), "frozen executed sources")
    require(set(executed_hashes) == EXECUTED_FILES, "Frozen execution manifest must retain its exact 15 source/config files")
    prefix = EXECUTION.relative_to(ROOT).as_posix()
    for name, expected in sorted(executed_hashes.items()):
        add(code, f"{prefix}/{name}", file_bytes(EXECUTION, name, expected))
    add(code, f"{prefix}/execution_manifest.json", execution_bytes)
    for name in ("configs/neural_validation_v3.json", "configs/megavul_validation_v3.json"):
        require(digest(code[name]) == executed_hashes[name], f"Current protocol differs from its executed snapshot: {name}")
    for detector, seeds in ACTUAL_FITS.items():
        script = ("src/train_neural_validation_v3.py" if detector == "finetuned_codebert"
                  else "src/predict_neural_validation_v3.py")
        for seed in seeds:
            name = f"analysis/neural_validation_v3/training/{detector}_seed-{seed}.json"
            receipt = read_json_bytes(paper_sources[name], name)
            require(receipt.get("seed") == seed and receipt.get("target_labels_accessed") is False,
                    f"Exported source-fit receipt identity/access declaration differs: {name}")
            require(hash_value(receipt.get("config_sha256"), name + " config")
                    == executed_hashes["configs/neural_validation_v3.json"],
                    f"Exported source-fit receipt is bound to another protocol: {name}")
            require(hash_value(receipt.get("script_sha256"), name + " script") == executed_hashes[script],
                    f"Exported source-fit receipt differs from the executed script snapshot: {name}")

    def figure_files(manifest: dict, names: tuple[str, ...], *, main: bool) -> None:
        figures = manifest.get("figures")
        require(isinstance(figures, dict) and set(names).issubset(figures), "Missing figure manifest entries")
        if main:
            require(set(figures) == set(names), "New figure manifest must bind exactly the three main figures")
            require(hash_value(manifest.get("public_analysis_manifest_sha256"), "new figure input manifest")
                    == digest(neural_bytes), "New figure manifest is stale against the neural aggregate release")
            require(hash_value(manifest.get("source_script_sha256"), "new figure source")
                    == digest(code["src/make_neural_validation_figures.py"]), "New figure generator binding is stale")
        for name in names:
            if main:
                require(manifest.get("schema_version") == "trace-neural-figures-v3",
                        "Unexpected current figure manifest schema")
                bound_files = hash_map(figures[name].get("files_sha256"), f"figure/{name} files")
                require(all(PurePosixPath(filename).name == filename
                            and filename.startswith(name + ".") for filename in bound_files),
                        f"Figure inventory contains another figure or a path: {name}")
                figure_hashes = {suffix: bound_files[f"{name}.{suffix}"]
                                 for suffix in ("pdf", "svg", "png", "tif", "tiff")
                                 if f"{name}.{suffix}" in bound_files}
            else:
                figure_hashes = hash_map(figures[name].get("sha256"), f"figure/{name}")
            require({"pdf", "svg", "png"}.issubset(figure_hashes), f"Missing figure formats: {name}")
            for suffix in ("pdf", "svg", "png"):
                filename = f"figures/{name}.{suffix}"
                paper_sources[filename] = file_bytes(PAPER, filename, figure_hashes[suffix])
            for suffix in ("tif", "tiff"):
                filename = f"figures/{name}.{suffix}"
                if suffix in figure_hashes and (PAPER / filename).is_file():
                    paper_sources[filename] = file_bytes(PAPER, filename, figure_hashes[suffix])

    for name, figures, main in (
        ("figures/figure_manifest_v2.json", SI_FIGURES, False),
        ("figures/figure_manifest_v3.json", MAIN_FIGURES, True),
    ):
        expected = LEGACY_FIGURE_MANIFEST_SHA256 if not main else None
        data = file_bytes(PAPER, name, expected)
        paper_sources[name] = data
        figure_files(read_json_bytes(data, name), figures, main=main)
    for path in sorted((PAPER / "figures/data").glob("*.csv")):
        name = path.relative_to(PAPER).as_posix()
        paper_sources[name] = file_bytes(PAPER, name)
    for name in LEGACY_PROVENANCE_FILES:
        archive_name = f"public_results/extension-v2/{name}"
        add(code, archive_name, file_bytes(ROOT, archive_name))
    for name, data in sorted(paper_sources.items()):
        if name not in {"author_biographies.docx", "declaration_of_competing_interests.docx"}:
            add(code, f"paper/neurocomputing_revision/{name}", data)

    source_names = ("main.tex", "main.bbl", "references.bib", "elsarticle.cls",
                    "elsarticle-num-names.bst", "neural_validation_main_tables.tex", "neural_allocation_display.tex")
    source = {name: paper_sources[name] for name in source_names}
    for name in MAIN_FIGURES:
        filename = f"figures/{name}.pdf"
        source[filename] = paper_sources[filename]
    main_source_dependencies(source, paper_sources)
    return Inputs(extract_title(paper_sources["main.tex"]), source, code, paper_sources,
                  neural, digest(neural_bytes), digest(execution_bytes))


def qa_contract() -> dict:
    """Concrete final_qa.json schema; report paths and packet members are repo-relative."""
    h = "<64-character SHA-256>"
    return {
        "status": "PASS", "submission_readiness": "ready_with_author_checks",
        "source_sha256": {"<every supplied paper-relative source member>": h},
        "pdfs": {name: {"sha256": h, "pages": "<positive integer measured from this PDF>"} for name in PDF_NAMES},
        "aggregate_reconstruction": {
            "status": "PASS", "report": "<repo-relative JSON report>", "sha256": h,
            "public_analysis_manifest_sha256": h,
        },
        "joint_lineage": {"status": "PASS", "report": "<repo-relative post-seal JSON report>", "sha256": h},
        "visual_audit": {
            "status": "PASS", "report": "<repo-relative Markdown/JSON report>", "sha256": h,
            "pdf_sha256": {name: h for name in PDF_NAMES},
        },
        "blind_review": {
            "status": "PASS", "unresolved_blocking_findings": 0,
            "reviews": [{"report": "<distinct frozen report>", "sha256": h,
                         "packet_manifest_sha256": h} for _ in range(3)],
            "packet_manifest": {"report": "<repo-relative JSON manifest>", "sha256": h},
            "resolution": {"status": "PASS", "report": "<repo-relative JSON report>", "sha256": h},
        },
        "_report_requirements": {
            "aggregate_reconstruction": "PASS; current public_analysis_manifest_sha256 and checker_script_sha256; both Generated table bytes check categories",
            "joint_lineage": "PASS; target_labels_accessed=false; both current config hashes; current checker hash; actual_shared_fits; 7 selected objects; all 14 dataset/fit evidence entries bound to exported training receipts, checkpoints and source freeze; both public-release-bound seals",
            "visual_audit": "Contains all three final PDF hashes; its pdf_sha256 map equals pdfs",
            "packet_manifest": {"files_sha256": {"<frozen repo-relative packet member>": h}},
            "resolution": {
                "status": "PASS", "unresolved_blocking_findings": 0,
                "packet_manifest_sha256": h, "review_report_sha256": [h, h, h],
                "source_sha256": "<exact final QA source_sha256 map>",
                "pdf_sha256": {name: h for name in PDF_NAMES},
            },
            "freshness": "All source/PDF/report/packet hashes are checked; the frozen packet may precede final revision, while its resolution must bind final sources/PDFs.",
        },
    }


def verify_qa(inputs: Inputs, qa_path: Path = QA) -> tuple[dict, dict[str, bytes], dict[str, bytes]]:
    """Validate receipts against the same byte snapshot that will be archived."""
    qa_resolved = qa_path.resolve(strict=True)
    require(qa_resolved.is_relative_to(ROOT.resolve()) and qa_resolved.is_file(), "Final QA must be a local repository file")
    qa_bytes = qa_resolved.read_bytes()
    qa = read_json_bytes(qa_bytes, "final QA")
    require(qa.get("status") == "PASS"
            and qa.get("submission_readiness") == "ready_with_author_checks",
            "Final QA must pass with ready_with_author_checks")
    expected_sources = inputs.source_sha256()
    supplied_sources = hash_map(qa.get("source_sha256"), "final paper sources")
    require(set(supplied_sources) == set(expected_sources),
            "Final QA source inventory is incomplete or different; use --print-source-sha256")
    for name, expected in expected_sources.items():
        require(supplied_sources[name] == expected, f"Final source QA is stale: {name}")
    if "title" in qa:
        require(qa["title"] == inputs.title, "Final QA title differs from main.tex")

    pdfs = qa.get("pdfs")
    require(isinstance(pdfs, dict) and set(pdfs) == set(PDF_NAMES), "Final QA must bind all three document PDFs")
    try:
        import pymupdf as fitz
    except ImportError as error:
        raise RuntimeError("Install PyMuPDF to verify final PDF page counts") from error
    pdf_bytes, pdf_hashes, pdf_records = {}, {}, {}
    for name in PDF_NAMES:
        record = pdfs[name]
        pages = record.get("pages")
        require(isinstance(pages, int) and not isinstance(pages, bool) and pages > 0,
                f"Invalid verified PDF page count: {name}")
        data = file_bytes(PAPER, f"{name}.pdf", hash_value(record.get("sha256"), f"{name} PDF"))
        with fitz.open(stream=data, filetype="pdf") as document:
            require(not document.is_encrypted and len(document) == pages, f"Final PDF page-count QA is stale: {name}")
        pdf_bytes[f"{name}.pdf"] = data
        pdf_hashes[name] = digest(data)
        pdf_records[name] = {"sha256": digest(data), "pages": pages}

    verification = {"verification/final_qa.json": qa_bytes}

    def artifact(record: object, label: str, *, passed: bool = True, json_report: bool = True) -> tuple[dict | None, bytes]:
        require(isinstance(record, dict), f"Missing QA artifact: {label}")
        if passed:
            require(record.get("status") == "PASS", f"QA artifact did not pass: {label}")
        name = record.get("report")
        require(isinstance(name, str) and PurePosixPath(name).suffix.lower() in {".json", ".md", ".txt"},
                f"QA receipt must be a textual report: {label}")
        data = file_bytes(ROOT, name, hash_value(record.get("sha256"), f"{label} report"))
        suffix = PurePosixPath(name).suffix.lower()
        add(verification, f"verification/{label}{suffix}", data)
        parsed = read_json_bytes(data, label) if json_report else None
        return parsed, data

    aggregate, _ = artifact(qa.get("aggregate_reconstruction"), "aggregate_reconstruction")
    require(aggregate.get("status") == "PASS"
            and hash_value(aggregate.get("public_analysis_manifest_sha256"), "aggregate reconstruction manifest")
            == inputs.neural_manifest_sha256, "Aggregate reconstruction is stale or not PASS")
    require(hash_value(qa["aggregate_reconstruction"].get("public_analysis_manifest_sha256"), "QA aggregate manifest")
            == inputs.neural_manifest_sha256, "Final QA aggregate manifest binding is stale")
    require(hash_value(aggregate.get("checker_script_sha256"), "aggregate checker")
            == digest(inputs.code["scripts/check_neural_aggregate_evidence.py"]), "Aggregate checker source binding is stale")
    require(not aggregate.get("skipped_checks"), "Final aggregate reconstruction has skipped checks")
    categories = aggregate.get("check_categories", [])
    require({f"Generated table bytes {name}" for name in TABLE_NAMES}.issubset(set(categories)),
            "Final aggregate checker must run with --check-table-bytes")

    lineage, _ = artifact(qa.get("joint_lineage"), "joint_lineage")
    require(lineage.get("status") == "PASS" and lineage.get("target_labels_accessed") is False,
            "Joint lineage report must pass without target-label access")
    for field, filename in (("config_sha256", "configs/neural_validation_v3.json"),
                            ("megavul_config_sha256", "configs/megavul_validation_v3.json")):
        require(hash_value(lineage.get(field), field) == digest(inputs.code[filename]), f"Joint lineage protocol is stale: {filename}")
    require(hash_value(lineage.get("checker_script_sha256"), "lineage checker")
            == digest(inputs.code["scripts/check_neural_prediction_lineage.py"]), "Lineage checker source binding is stale")
    require(lineage.get("actual_shared_fits") == ACTUAL_FITS
            and lineage.get("independent_fitted_object_count") == 7
            and lineage.get("target_dataset_prediction_copies") == 14,
            "Joint lineage must preserve the seven selected source objects shared across two datasets")
    seals = lineage.get("seals")
    require(isinstance(seals, list) and len(seals) == 2
            and {seal.get("dataset") for seal in seals} == {"diversevul", "megavul"},
            "Final lineage report must be post-seal and bind both datasets")
    for seal in seals:
        expected = inputs.neural_manifest["datasets"][seal["dataset"]]["decision_seal_sha256"]
        require(hash_value(seal.get("sha256"), "lineage seal") == hash_value(expected, "released dataset seal"),
                f"Joint lineage and public dataset seal differ: {seal['dataset']}")
    evidence = lineage.get("evidence")
    identities = {(dataset, detector, seed) for dataset in ("diversevul", "megavul")
                  for detector, seeds in ACTUAL_FITS.items() for seed in seeds}
    require(isinstance(evidence, list) and len(evidence) == 14
            and {(row.get("dataset"), row.get("detector"), row.get("seed")) for row in evidence} == identities,
            "Joint lineage must contain all fourteen dataset/selected-fit identities")
    for row in evidence:
        receipt_name = f"training/{row['detector']}_seed-{row['seed']}.json"
        receipt_bytes = inputs.paper_sources[f"analysis/neural_validation_v3/{receipt_name}"]
        receipt = read_json_bytes(receipt_bytes, receipt_name)
        require(hash_value(row.get("training_manifest_sha256"), "lineage fit receipt") == digest(receipt_bytes)
                and hash_value(row.get("checkpoint_sha256"), "lineage checkpoint")
                == hash_value(receipt.get("checkpoint_sha256"), receipt_name + " checkpoint"),
                f"Exported fit receipt differs from post-seal lineage: {receipt_name}")
        source_field = ("source_token_manifest_sha256" if row["detector"] == "finetuned_codebert"
                        else "source_rows_sha256")
        require(hash_value(receipt.get(source_field), receipt_name + " source")
                == hash_value(lineage.get(source_field), "lineage source"),
                f"Exported fit receipt is bound to another source freeze: {receipt_name}")

    visual_record = qa.get("visual_audit")
    _, visual_bytes = artifact(visual_record, "visual_audit", json_report=False)
    require(hash_map(visual_record.get("pdf_sha256"), "visual PDF bindings") == pdf_hashes,
            "Visual audit PDF binding is stale")
    visual_text = visual_bytes.decode("utf-8-sig").upper()
    require(all(expected in visual_text for expected in pdf_hashes.values()),
            "Visual audit report must contain all final PDF hashes")
    if PurePosixPath(visual_record["report"]).suffix.lower() == ".json":
        require(read_json_bytes(visual_bytes, "visual audit").get("status") == "PASS", "Visual JSON report did not pass")

    blind = qa.get("blind_review")
    require(isinstance(blind, dict) and blind.get("status") == "PASS"
            and type(blind.get("unresolved_blocking_findings")) is int
            and blind["unresolved_blocking_findings"] == 0, "Blocking blind-review findings remain unresolved")
    packet, packet_bytes = artifact(blind.get("packet_manifest"), "blind_review_packet_manifest", passed=False)
    packet_sha = digest(packet_bytes)
    packet_hashes = hash_map(packet.get("files_sha256"), "frozen review packet")
    # The packet is frozen evidence. It need not equal the revised manuscript.
    for name, expected in packet_hashes.items():
        file_bytes(ROOT, name, expected)
    reviews = blind.get("reviews")
    require(isinstance(reviews, list) and len(reviews) == 3, "Exactly three frozen blind reviews are required")
    review_hashes, review_paths = [], []
    for index, review in enumerate(reviews, 1):
        _, data = artifact(review, f"blind_review_{index}", passed=False, json_report=False)
        require(hash_value(review.get("packet_manifest_sha256"), f"review {index} packet") == packet_sha,
                f"Blind review {index} is bound to another packet")
        review_hashes.append(digest(data))
        review_paths.append(review["report"])
    require(len(set(review_hashes)) == len(set(review_paths)) == 3, "Blind reviews must be three distinct frozen reports")
    resolution, _ = artifact(blind.get("resolution"), "blind_review_resolution")
    require(resolution.get("status") == "PASS"
            and type(resolution.get("unresolved_blocking_findings")) is int
            and resolution["unresolved_blocking_findings"] == 0, "Review resolution did not clear all blocking findings")
    require(hash_value(resolution.get("packet_manifest_sha256"), "resolution packet") == packet_sha,
            "Resolution is bound to another review packet")
    resolved_reviews = resolution.get("review_report_sha256")
    require(isinstance(resolved_reviews, list) and len(resolved_reviews) == 3
            and {hash_value(h, "resolved review") for h in resolved_reviews} == set(review_hashes),
            "Resolution does not bind all three frozen review reports")
    require(hash_map(resolution.get("source_sha256"), "resolution final sources") == expected_sources,
            "Blind-review resolution is stale against the final sources")
    require(hash_map(resolution.get("pdf_sha256"), "resolution final PDFs") == pdf_hashes,
            "Blind-review resolution is stale against the final PDFs")

    if "synthesis" in blind:
        artifact(blind["synthesis"], "blind_review_synthesis", passed=False, json_report=False)
    for label, record in qa.get("editorial_audits", {}).items():
        require(re.fullmatch(r"[a-z][a-z0-9_]*", label) is not None, "Unsafe editorial audit label")
        artifact(record, "author_" + label, passed=False,
                 json_report=PurePosixPath(record["report"]).suffix.lower() == ".json")

    summary = {
        "status": "PASS", "title": inputs.title, "submission_readiness": qa["submission_readiness"],
        "final_qa_sha256": digest(qa_bytes), "source_sha256": expected_sources, "pdfs": pdf_records,
        "release_manifests": {
            "legacy_public_analysis_manifest_sha256": LEGACY_PUBLIC_MANIFEST_SHA256,
            "legacy_figure_manifest_sha256": LEGACY_FIGURE_MANIFEST_SHA256,
            "neural_public_analysis_manifest_sha256": inputs.neural_manifest_sha256,
            "frozen_execution_manifest_sha256": inputs.execution_manifest_sha256,
        },
        "checks": {
            "aggregate_reconstruction": qa["aggregate_reconstruction"],
            "joint_lineage": qa["joint_lineage"], "visual_audit": visual_record,
            "blind_review": blind,
        },
        "archived_verification_files_sha256": {name: digest(data) for name, data in sorted(verification.items())},
        "limitations": [
            "Aggregate reconstruction verifies released bytes and arithmetic; it does not retrain detectors or replay raw-row selection.",
            "Hash receipts attest local byte identity and checked lineage, not external time attestation or every historical leakage channel.",
            "The frozen blind-review packet can precede final revision; its resolution is bound to final source and PDF bytes.",
            "Author declarations, affiliations, related submissions and final journal portal fields remain author checks.",
        ],
    }
    return summary, verification, pdf_bytes


CODE_README = """# TRACE supplementary code and aggregate evidence

This package accompanies:

**TITLE_PLACEHOLDER**

TRACE means **Triage Risk, Allocation and Cost Evaluation**. VulTriage is the
frozen weighted policy under audit. The verified final documents have
MAIN_PAGES_PLACEHOLDER manuscript pages, SI_PAGES_PLACEHOLDER supplementary
pages and COVER_PAGES_PLACEHOLDER cover-letter pages. These counts come from
the final PDF QA receipt.

## Supplied evidence

The original 18 aggregate members remain byte-identical under
paper/neurocomputing_revision/analysis/public_analysis_manifest.json. The new
analysis/neural_validation_v3 subtree contains exactly the members declared
by its separate public_analysis_manifest.json, including source-fit receipts,
dataset receipts, domain diagnostics and project/seed aggregate evidence.
The two generated neural TeX tables are bound by that manifest.

The main figures are fig_workflow, fig_neural_allocation and fig_neural_costs;
figures/figure_manifest_v3.json binds their bytes, input release and generator.
The unchanged fig_matched_tradeoff and fig_cost_sensitivity figures support
the legacy supplementary analysis. Their v2 manifest is retained as historical
provenance; its old workflow entry does not bind the current main workflow.
PDF/SVG/PNG exports and aggregate figure data are supplied.

Current pipeline source is supplied at repository-relative paths. Exact
executed source/config snapshots are separately retained under
outputs/neural-validation-v3/frozen_scripts/execution, with the original
15-file execution_manifest.json. A current utility and its frozen executed
version are distinct provenance objects. The execution snapshot and training
receipts are local hash evidence, not public preregistration or external time
attestations.

The separate display_additions manifest binds a primary assigned-action table
and compact summaries of every frozen allocation and free-volume cost grid.
These expose existing evidence without changing the 48-member neural release
or either original generated neural table. Run
python scripts/build_neural_display_additions.py --check to reconstruct their
bytes. requirements-validation-v3-lock.txt and the runtime sidecar distinguish
the dedicated validation environment from the historical control environment;
the sidecar was collected after execution and is not a historical attestation.

No raw datasets, token/embedding caches, weights, checkpoints, label vaults,
per-function predictions or decisions, private skills, transient TeX files,
or figure QA alignment/overlay art are supplied. Historical manifests may
refer to excluded files. The public manifests define the released subset.
Each ZIP has its own exact inventory and payload hashes in ARCHIVE_MANIFEST.json.

## Aggregate-only reconstruction

Use Python 3.12. Run from the extracted code-package root:

~~~text
python -m pip install -r requirements-audit.txt
python scripts/check_neurocomputing_revision.py --aggregate-only --skip-pdf
python scripts/check_neural_aggregate_evidence.py --analysis paper/neurocomputing_revision/analysis/neural_validation_v3 --output neural_aggregate_reconstruction.json --check-table-bytes
python scripts/build_neural_display_additions.py --check
python scripts/build_neurocomputing_supplement_tables.py
~~~

The legacy checker verifies its original aggregate cohort, arithmetic, paired
comparisons, source-fit declarations and table transcription. It explicitly
reports skipped raw-input replay and PDF compilation. The separate neural
checker independently reconstructs actual-fit averages, paired medians,
direction counts and all 10,000-draw project bootstrap intervals from the
released aggregates, including failed/undefined endpoints, and verifies both
generated neural table hashes. Neither command retrains a detector or opens
the excluded raw-row archives.

The legacy SI table builder can regenerate its aggregate-derived tables.
The new table sources are supplied with verified byte hashes; their exporter
requires the original local evaluation and decision seals and is not an
aggregate-only table regeneration command.

For figure regeneration, see the CLI help for the supplied
src/make_neurocomputing_audit_figures.py and src/make_neural_validation_figures.py.
Font and plotting-library changes may alter figure bytes while preserving
the supplied aggregate values. The meaningful synthetic policy/input tests
are:

~~~text
python -m pytest tests/test_neuro_controls.py tests/test_neuro_operational_audit.py tests/test_neural_validation_v3.py tests/test_megavul_preparation.py -q
~~~

The packaged pytest.ini adds src to the import path. These tests exercise
policy arithmetic, fallback/endpoint behavior and cohort preparation rules;
they do not constitute a rerun of the detector experiments.

## Full experimental replay

Full experiments require independently obtained raw PrimeVul and DiverseVul
releases, the recorded MegaVul mirror bytes, the frozen historical manifests
and clone-screen index, source-role and target-cohort reconstruction, and the
pinned microsoft/codebert-base revision
3b0952feddeffad0063f274080e3c23d75e7eb39. None of those raw inputs or model
weights are bundled. The MegaVul download is a third-party mirror; its hash
receipt does not establish byte equivalence to an independently verified
official release. The original v1/v2 configs retain the project alias maps.

requirements-experiment.txt adds the recorded Torch 2.6.0 and Transformers
4.51.3 versions. Install the appropriate CUDA-enabled Torch build for the
host; fine-tuning requires a CUDA GPU with bfloat16 support. Exact source
snapshots, configs and receipts describe the local executed pipeline, but
hardware/library differences can change fits. Use the preparation,
encoding, training, prediction and evaluation entry points with their CLI
help and supplied protocol. Do not replace missing historical inputs with
new splits while claiming exact replay.

For the original frozen CodeBERT analysis, one head per project was copied
to five seed addresses. Those addresses are not five independent fits.
The new validation uses three actual fine-tuned CodeBERT fits, three actual
hashing-SGD fits and one selected frozen-encoder logistic head, all shared
across DiverseVul and MegaVul. These are seven selected fitted objects.
Reference hyperparameter grids fit multiple candidates; seven does not
mean only seven optimizer calls. Report the collections separately with
projects as the within-dataset inference units; their overlapping project
identities do not create a pooled set of independent projects.

Original repository: https://github.com/bianyanbo44-afk/vultriage-fcs

This local package contains the revision state. Unpushed local changes are
not asserted to be available at that public URL.

## Editable documents and final QA

From paper/neurocomputing_revision:

~~~text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
~~~

The code ZIP contains editable sources. The complete submission ZIP supplies
the final manuscript, supplementary and cover-letter PDFs, highlights,
author-facing README, manuscript-source ZIP and code ZIP.

verification/verification_summary.json and the scientific textual receipts
record final source/PDF bindings, aggregate reconstruction, post-seal joint
lineage and visual audit. The three mock review reports, packet manifest,
resolution and full author QA JSON are supplied only in the complete local
handoff archive under author_qa, rather than in this supplementary archive.
Local report paths can refer to evidence outside this aggregate release.
Final QA is not a claim that every historical leakage channel was excluded.
Author declarations and journal portal fields remain author checks.
"""

AUDIT_REQUIREMENTS = """numpy>=1.26
pandas>=2.2
scipy>=1.12
scikit-learn>=1.4
matplotlib>=3.8
psutil>=5.9
ijson>=3.3
Pillow>=10
PyMuPDF>=1.24
pytest>=8
"""
EXPERIMENT_REQUIREMENTS = """-r requirements-audit.txt
# Select the platform-appropriate CUDA wheel; fine-tuning requires CUDA bf16.
torch==2.6.0
transformers==4.51.3
"""


def rendered_readme(title: str, pdfs: dict) -> bytes:
    text = CODE_README.replace("TITLE_PLACEHOLDER", title)
    for placeholder, name in (("MAIN_PAGES_PLACEHOLDER", "main"),
                              ("SI_PAGES_PLACEHOLDER", "supplementary"),
                              ("COVER_PAGES_PLACEHOLDER", "cover_letter")):
        text = text.replace(placeholder, str(pdfs[name]["pages"]))
    return text.encode("utf-8")


def zip_bytes(entries: dict[str, bytes], title: str, scope: str, filename: str) -> tuple[bytes, dict, bytes]:
    require("ARCHIVE_MANIFEST.json" not in entries, "Payload cannot supply its own archive manifest")
    for name in entries:
        safe_member(name, nested_zip=name in {SOURCE_NAME, CODE_NAME})
    hashes = {name: digest(data) for name, data in sorted(entries.items())}
    manifest = {
        "schema_version": "trace-deterministic-submission-1", "title": title, "scope": scope,
        "zip_timestamp": list(ZIP_DATE), "sha256_scope": "All payload members; manifest not self-hashed",
        "files_sha256": hashes,
    }
    manifest_bytes = json_bytes(manifest)
    payload = dict(entries)
    payload["ARCHIVE_MANIFEST.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    archive_bytes = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        require(archive.testzip() is None, f"Archive CRC failure: {filename}")
        require(archive.namelist() == sorted(payload), f"Archive inventory mismatch: {filename}")
        for name, data in payload.items():
            require(digest(archive.read(name)) == digest(data), f"Archive member SHA mismatch: {name}")
        for info in archive.infolist():
            require(info.date_time == ZIP_DATE and info.create_system == 3
                    and info.external_attr == (stat.S_IFREG | 0o644) << 16
                    and info.compress_type == zipfile.ZIP_DEFLATED
                    and not info.extra and not info.comment, f"Archive metadata mismatch: {info.filename}")
    record = {
        "filename": filename, "bytes": len(archive_bytes), "sha256": digest(archive_bytes),
        "payload_members": len(entries), "archive_manifest_sha256": digest(manifest_bytes), "scope": scope,
    }
    return archive_bytes, record, manifest_bytes


def build(inputs: Inputs, summary: dict, verification: dict[str, bytes], pdfs: dict[str, bytes]) -> dict[str, bytes]:
    source, code = dict(inputs.source), dict(inputs.code)
    add(source, "README.txt", (
        inputs.title + "\n\nEntry point: main.tex\n"
        "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex\n"
        "This archive contains all stated main-manuscript source dependencies.\n"
        "Checked PDFs, supplementary sources and code are supplied in the companion packages.\n"
    ).encode("utf-8"))
    add(code, "README.md", rendered_readme(inputs.title, summary["pdfs"]))
    add(code, "requirements-audit.txt", AUDIT_REQUIREMENTS.encode("utf-8"))
    add(code, "requirements-experiment.txt", EXPERIMENT_REQUIREMENTS.encode("utf-8"))
    add(code, "pytest.ini", b"[pytest]\npythonpath = src\n")
    for name, data in sorted(verification.items()):
        if not name.startswith(("verification/blind_review", "verification/author_")) and name != "verification/final_qa.json":
            add(code, name, data)
    add(code, "verification/verification_summary.json", json_bytes(summary))
    source_zip, source_record, source_manifest = zip_bytes(
        source, inputs.title, "Editable main manuscript; root entry main.tex", SOURCE_NAME)
    code_zip, code_record, code_manifest = zip_bytes(
        code, inputs.title, "Current and frozen executed source, manifest-bound aggregate evidence and checked sources; no raw-row archives", CODE_NAME)
    submission_manifest = json_bytes({
        "title": inputs.title, "readiness": summary["submission_readiness"],
        "final_qa_sha256": summary["final_qa_sha256"], "pdfs": summary["pdfs"],
        "release_manifests": summary["release_manifests"], "archives": [source_record, code_record],
        "note": "The complete handoff archive checksum is recorded separately in bundle_build_checks.json to avoid a checksum cycle.",
    })
    full = dict(pdfs)
    for name in ("highlights.txt", "README_NEUROCOMPUTING.md", "REVISION_NOTES.md", "author_biographies.docx",
                 "declaration_of_competing_interests.docx", "repository_deposit/README_DEPOSIT.md",
                 "repository_deposit/repository_metadata_draft.json", "repository_deposit/DATA_DICTIONARY.md"):
        add(full, name, inputs.paper_sources[name])
    for name, data in sorted(verification.items()):
        if name.startswith(("verification/blind_review", "verification/author_")) or name == "verification/final_qa.json":
            add(full, "author_qa/" + name, data)
    add(full, SOURCE_NAME, source_zip, nested_zip=True)
    add(full, CODE_NAME, code_zip, nested_zip=True)
    add(full, "submission_manifest.json", submission_manifest)
    full_zip, full_record, full_manifest = zip_bytes(
        full, inputs.title, "Complete local handoff; journal upload roles remain author checks", FULL_NAME)
    records = [source_record, code_record, full_record]
    return {
        SOURCE_NAME: source_zip, CODE_NAME: code_zip, FULL_NAME: full_zip,
        f"{Path(SOURCE_NAME).stem}.ARCHIVE_MANIFEST.json": source_manifest,
        f"{Path(CODE_NAME).stem}.ARCHIVE_MANIFEST.json": code_manifest,
        f"{Path(FULL_NAME).stem}.ARCHIVE_MANIFEST.json": full_manifest,
        "submission_manifest.json": submission_manifest,
        "bundle_build_checks.json": json_bytes({"status": "PASS", "title": inputs.title, "archives": records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Validate complete final QA and input inventories; write nothing")
    mode.add_argument("--print-source-sha256", action="store_true", help="Print exact source map needed by final QA; write nothing")
    mode.add_argument("--describe-qa", action="store_true", help="Print final QA/report schema without reading release outcomes")
    parser.add_argument("--qa", type=Path, default=QA, help="Final QA JSON; defaults to tmp/neurocomputing-strengthening/final_qa.json")
    args = parser.parse_args()
    if args.describe_qa:
        print(json_bytes(qa_contract()).decode("utf-8"), end="")
        return
    inputs = collect_inputs()
    if args.print_source_sha256:
        print(json_bytes(inputs.source_sha256()).decode("utf-8"), end="")
        return
    summary, verification, pdfs = verify_qa(inputs, args.qa)
    if args.validate_only:
        print(json_bytes({
            "status": "PASS", "mode": "validate-only", "title": inputs.title,
            "source_members": len(inputs.paper_sources), "pdfs": summary["pdfs"],
            "final_qa_sha256": summary["final_qa_sha256"], "release_manifests": summary["release_manifests"],
            "archives_written": False,
        }).decode("utf-8"), end="")
        return
    outputs = build(inputs, summary, verification, pdfs)
    OUT.mkdir(parents=True, exist_ok=True)
    require(OUT.resolve().is_relative_to(ROOT.resolve()), "Submission output escapes the repository")
    for name, data in outputs.items():
        path = OUT / name
        temporary = path.with_name(f".{name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        require(digest(path.read_bytes()) == digest(data), f"Written output SHA mismatch: {name}")
    print(outputs["bundle_build_checks.json"].decode("utf-8"), end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(json_bytes({"status": "FAIL", "reason": str(error)}).decode("utf-8"), end="")
        raise SystemExit(1) from None
