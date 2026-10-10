"""Meaningful boundary checks for the sealed MegaVul cohort preparation."""

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import prepare_megavul_validation_v3 as prep
from vultriage.data import sha256
from vultriage.extension_data import exact_code_key
from vultriage.near_duplicate import AuditDocument, NearDuplicateIndex, lexical_token_set


def record(text, index, label=False, project="org/wanted", commit=None):
    return dict(repo_name=project, commit_hash=commit or f"{index + 1000:040x}",
                commit_date=1700000000 + index, is_vul=label,
                func_before=text if label else None,
                func=f"int patch_{index}(void) {{ return patch_marker_{index}; }}" if label else text)


def fixture_args(tmp_path, keep=200):
    frozen_inputs = tmp_path / "inputs"
    frozen_inputs.mkdir()
    prime_text = " ".join(f"historical_prime_token_{i}" for i in range(100))
    diverse_text = "int historical_diverse(void) { return historical_marker; }"
    diverse = tmp_path / "diverse.json"
    diverse.write_text(json.dumps(dict(func=diverse_text, commit_id="b" * 40, project="old")) + "\n", encoding="utf-8")
    clone_index = tmp_path / "clone.sqlite"
    with NearDuplicateIndex(clone_index) as index:
        index.add_document(AuditDocument(
            dataset="primevul", row_id="historical-prime", source_file="old.jsonl",
            line_number=1, project="old", project_group="old",
            exact_code_key=exact_code_key(prime_text), tokens=lexical_token_set(prime_text),
        ))
        index.finish_documents()
    np.savez_compressed(frozen_inputs / "historical_exclusions.npz",
                        prime_keys=np.asarray([exact_code_key(prime_text)]),
                        diverse_keys=np.asarray([exact_code_key(diverse_text)]),
                        prime_commits=np.asarray(["a" * 40 + "?w=1", "c" * 8]),
                        diverse_commits=np.asarray(["b" * 40]))
    rows = [record(f"int keep_{i}(void) {{ return keep_marker_{i}; }}", i, i < 20,
                   project="org/ALIAS") for i in range(keep)]
    # Each historical screen has a distinct trigger, avoiding order masking.
    rows.extend([
        record(prime_text, 10000, True),
        record(diverse_text, 10001, True),
        record("/* formatting difference */ " + prime_text, 10002, True),
        record("/* new comment */ " + diverse_text, 10003, True),
        record("int prime_commit_duplicate(void) { return prime_committer; }", 10004, True, commit="a" * 40),
        record("int diverse_commit_duplicate(void) { return diverse_committer; }", 10005, True, commit="b" * 40),
        record(prime_text + " additional_lexical_token", 10006, True),
        record("int abbreviated_commit_duplicate(void) { return abbreviated_marker; }", 10007, True, commit="c" * 40),
    ])
    conflict = "int globally_conflicting(void) { return conflict_marker; }"
    rows.append(record(conflict, 10008, True))
    # A conflict outside D62 must still quarantine its earlier target record.
    rows.append(record(conflict, 10009, False, project="not-a-candidate"))
    rows.append(record(rows[0]["func_before"], 10010, True))
    rows.extend(record(f"int low_{i}(void) {{ return low_marker_{i}; }}", i + 20000,
                       i < 20, project="owner/low") for i in range(199))
    megavul = tmp_path / "megavul.json"
    megavul.write_text(json.dumps(rows), encoding="utf-8")
    source_config = tmp_path / "source-config.json"
    source_config.write_text('{"protocol_version":"frozen-source"}\n', encoding="utf-8")
    config = tmp_path / "megavul-config.json"
    config.write_text(json.dumps(dict(
        protocol_version="trace-second-source-validation-v3", eligibility=prep.FROZEN_ELIGIBILITY,
        sha256=sha256(megavul), bytes=megavul.stat().st_size,
        mirror="fixture-mirror", mirror_revision="fixed-fixture-revision",
        repository_identity_allowlist={
            "core": ["dovecot/core"], "server": ["mariadb/server"],
            "wanted": ["org/alias", "org/wanted"],
        },
    )), encoding="utf-8")
    candidates = ["wanted", "low", "core", "server"] + [f"missing-{i}" for i in range(58)]
    frozen = dict(
        candidate_projects=candidates, selected_projects=["low"],
        source_target_projects_disjoint=True, source_roles_commit_disjoint=True,
        source_counts={role: {"0": 20, "1": 20} for role in ["train", "model_validation", "calibration"]},
        source_audit={}, source_train_natural=40, source_train_sampled=40,
        protocol_version="frozen-source", input_hashes=dict(
            config=sha256(source_config), diversevul=sha256(diverse), clone_index=sha256(clone_index)),
        output_hashes={"historical_exclusions.npz": sha256(frozen_inputs / "historical_exclusions.npz")},
    )
    (frozen_inputs / "preparation_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    return argparse.Namespace(config=config, source_config=source_config, megavul=megavul,
                              frozen_inputs=frozen_inputs, diversevul=diverse,
                              clone_index=clone_index, output=frozen_inputs / "megavul",
                              resume_incomplete=False)


def test_row_mapping_uses_before_for_vulnerable_and_never_leaks_field_choice():
    before = "int vulnerable(void) { return unchecked_value; }"
    raw = record(before, 0, True)
    mapped, reason = prep.map_megavul_row(raw, 0, Path("megavul.json"), {})
    assert reason is None
    assert mapped["text"] == before
    assert mapped["text"] != raw["func"]
    assert mapped["label"] == 1
    assert mapped["source_pointer"] == "/0"
    assert "source_code_field" not in mapped
    negative, reason = prep.map_megavul_row(record(before, 1), 1, Path("megavul.json"), {})
    assert reason is None and negative["text"] == before and negative["label"] == 0
    raw["func_before"] = None
    mapped, reason = prep.map_megavul_row(raw, 2, Path("megavul.json"), {})
    assert mapped is None and reason == "missing_vulnerable_before"


def test_project_rule_is_exact_basename_and_frozen_alias_only():
    lookup = {"alias": "wanted"}
    assert prep.megavul_project("Org/ALIAS", lookup) == "wanted"
    assert prep.megavul_project("Org/ALIAS-extra", lookup) == "alias-extra"
    assert prep.megavul_project("wanted-copy", lookup) == "wanted-copy"


@pytest.mark.parametrize("name,url,group,allowed,reason", [
    ("LibreOffice/core", "https://github.com/LibreOffice/core/commit/abc", "core", False,
     "repository_identity_not_allowlisted"),
    ("libreoffice/core", None, "core", False, "repository_identity_not_allowlisted"),
    ("ONLYOFFICE/core", "https://github.com/ONLYOFFICE/core/commit/abc", "core", False,
     "repository_identity_not_allowlisted"),
    ("unrelated/core", None, "core", False, "repository_identity_not_allowlisted"),
    ("DoveCot/CORE", "https://github.com/DOVECOT/Core/commit/abc", "core", True, None),
    ("core", "https://github.com/dovecot/core/commit/abc", "core", True, None),
    ("core", None, "core", False, "repository_identity_unresolved"),
    ("core", "https://github.com/LibreOffice/core/commit/abc", "core", False,
     "repository_identity_not_allowlisted"),
    ("dovecot/core", "https://github.com/LibreOffice/core/commit/abc", "core", False,
     "repository_identity_not_allowlisted"),
    ("ONLYOFFICE/core", "https://github.com/dovecot/core/commit/abc", "core", False,
     "repository_identity_not_allowlisted"),
    ("MariaDB/SERVER", None, "server", True, None),
    ("server", "https://github.com/mariadb/server/commit/abc", "server", True, None),
    ("mysql/server", "https://github.com/mysql/server/commit/abc", "server", False,
     "repository_identity_not_allowlisted"),
    ("unrelated/server", None, "server", False, "repository_identity_not_allowlisted"),
    ("server", "https://example.com/mariadb/server/commit/abc", "server", False,
     "repository_identity_unresolved"),
])
def test_exact_repository_identity_prevents_basename_collisions(name, url, group, allowed, reason):
    allowlist = {"core": frozenset({"dovecot/core"}), "server": frozenset({"mariadb/server"})}
    identity = prep.repository_identity(name, url, {}, allowlist)
    assert identity.candidate_group == group
    assert identity.allowed is allowed
    assert identity.exclusion_reason == reason


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/DoveCot/CORE/commit/abc?w=1#diff", ("dovecot/core", "github.com")),
    ("https://gitlab.com/LibTiff/LibTiff/-/commit/abc", ("libtiff/libtiff", "gitlab.com")),
    ("https://gitlab.com/group/subgroup/repo/-/commit/abc", ("group/subgroup/repo", "gitlab.com")),
    ("https://cgit.freedesktop.org/systemd/systemd/commit/?id=abc", ("systemd/systemd", "cgit.freedesktop.org")),
    ("https://example.com/unrelated/core/commit/abc", (None, "example.com")),
    (None, (None, None)),
])
def test_commit_namespace_parser_uses_complete_repository_path(url, expected):
    assert prep.commit_repository_namespace(url) == expected


@pytest.mark.parametrize("name,url,group,allowed", [
    ("libtiff", "https://gitlab.com/libtiff/libtiff/-/commit/abc", "libtiff", True),
    ("vadz/libtiff", "https://gitlab.com/libtiff/libtiff/-/commit/abc", "libtiff", True),
    ("rzkn/libtiff", "https://gitlab.com/rzkn/libtiff/-/commit/abc", "libtiff", True),
    ("zherczeg/jerryscript", "https://github.com/jerryscript-project/jerryscript/commit/abc", "jerryscript", True),
    ("systemd", "http://cgit.freedesktop.org/systemd/systemd/commit/?id=abc", "systemd", True),
    ("unverified/libtiff", "https://gitlab.com/libtiff/libtiff/-/commit/abc", "libtiff", False),
    ("vadz/libtiff", "https://github.com/unverified/libtiff/commit/abc", "libtiff", False),
])
def test_only_explicitly_approved_repository_mirrors_share_a_family(name, url, group, allowed):
    allowlist = {
        "libtiff": frozenset({"libtiff/libtiff", "vadz/libtiff", "rzkn/libtiff"}),
        "jerryscript": frozenset({"jerryscript-project/jerryscript", "zherczeg/jerryscript"}),
        "systemd": frozenset({"systemd/systemd"}),
    }
    identity = prep.repository_identity(name, url, {}, allowlist)
    assert identity.candidate_group == group and identity.allowed is allowed
    assert identity.basis == "commit_url_namespace"


def test_repository_allowlist_must_explicitly_freeze_core_and_server():
    candidates = ["core", "server", "libtiff"]
    with pytest.raises(ValueError, match="explicit frozen"):
        prep.repository_allowlist({}, candidates)
    with pytest.raises(ValueError, match="Frozen core/server"):
        prep.repository_allowlist({"repository_identity_allowlist": {
            "core": ["LibreOffice/core"], "server": ["mariadb/server"],
        }}, candidates)
    with pytest.raises(ValueError, match="Invalid repository identity allowlist group"):
        prep.repository_allowlist({"repository_identity_allowlist": {
            "core": ["dovecot/core"], "server": ["mariadb/server"], "unfrozen": ["owner/unfrozen"],
        }}, candidates)
    result = prep.repository_allowlist({"repository_identity_allowlist": {
        "core": ["DOVECOT/Core"], "server": ["MariaDB/Server"],
        "libtiff": ["libtiff/libtiff", "vadz/libtiff", "rzkn/libtiff"],
    }}, candidates)
    assert result["core"] == frozenset({"dovecot/core"})
    assert result["server"] == frozenset({"mariadb/server"})


def test_complete_array_screening_seals_labels_and_keeps_readonly_source(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "aliases", lambda: {"alias": "wanted"})
    args = fixture_args(tmp_path)
    original_index_hash = sha256(args.clone_index)
    original_source_hash = sha256(args.source_config)
    result = prep.prepare(args)
    assert result["selected_projects"] == ["wanted"]  # Not the historical selected subset.
    assert result["target_rows"] == 200 and result["target_positive"] == 20
    assert len(result["candidate_projects"]) == 62
    excluded = result["exclusion_counts"]["wanted"]
    assert excluded["prime_exact_duplicate"] == 1
    assert excluded["diverse_exact_duplicate"] == 1
    assert excluded["prime_lexical_duplicate"] == 1
    assert excluded["diverse_lexical_duplicate"] == 1
    assert excluded["prime_commit_overlap"] == 2
    assert excluded["diverse_commit_overlap"] == 1
    assert excluded["source_near_clone"] == 1
    assert excluded["label_conflict"] == 1
    assert sha256(args.clone_index) == original_index_hash
    assert sha256(args.source_config) == original_source_hash
    rows = [json.loads(line) for line in (args.output / "target_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(set(row) & {"label", "target", "is_vul", "func_before", "source_code_field"} for row in rows)
    assert all(row["repository_identity"] == "org/alias" and row["repository_identity_allowed"] for row in rows)
    assert all(row["repository_identity_basis"] == "qualified_repository_name" for row in rows)
    assert all(row["repository_host"] is None and "repository_identity_exclusion_reason" not in row for row in rows)
    with np.load(args.output / "target_label_vault.npz", allow_pickle=False) as vault:
        assert vault["labels"].tolist() == [1] * 20 + [0] * 180
        assert vault["row_ids"].tolist() == [row["row_id"] for row in rows]
        assert vault["positions"].tolist() == [row["position"] for row in rows] == list(range(200))
    with (args.output / "candidate_projects.csv").open(encoding="utf-8", newline="") as stream:
        audit = {row["project_group"]: row for row in csv.DictReader(stream)}
    assert len(audit) == 62 and audit["low"]["selected"] == "False"
    assert audit["low"]["screened_safe"] == "179"
    assert audit["missing-0"]["screened_safe"] == "0"
    assert audit["wanted"]["within_megavul_exact_duplicate"] == "1"
    assert json.loads(audit["wanted"]["matched_repositories"]) == ["org/ALIAS", "org/wanted"]
    assert json.loads(audit["wanted"]["observed_repositories"]) == ["org/ALIAS", "org/wanted"]
    assert audit["core"]["repository_identity_excluded_rows"] == "0"
    assert json.loads(audit["core"]["repository_identity_excluded_repositories"]) == {}
    assert result["repository_identity_allowlist"]["core"] == ["dovecot/core"]
    assert result["repository_identity_mappings"]["org/ALIAS"]["repository_identity"] == "org/alias"
    assert result["checks"]["repository_identity_allowlist_applied"]
    assert result["provenance"]["official_byte_equivalence"] == "unknown"
    assert not result["target_outputs_used_for_selection"]
    assert not result["projects_pooled_across_datasets"]
    assert not result["chronological_holdout_claim"]
    # A sealed result is immutable even when the interrupted-run switch is set.
    args.resume_incomplete = True
    with pytest.raises(FileExistsError, match="immutable"):
        prep.prepare(args)


def test_zero_qualifying_projects_are_reported_without_relaxing_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "aliases", lambda: {"alias": "wanted"})
    args = fixture_args(tmp_path, keep=199)
    result = prep.prepare(args)
    assert result["selected_projects"] == [] and result["target_rows"] == 0
    assert (args.output / "target_rows.jsonl").read_text(encoding="utf-8") == ""
    with np.load(args.output / "target_label_vault.npz", allow_pickle=False) as vault:
        assert vault["labels"].dtype == np.int8 and vault["labels"].size == 0
        assert vault["row_ids"].dtype.kind == "U" and vault["positions"].size == 0
    assert result["eligibility"] == prep.FROZEN_ELIGIBILITY


def test_transactional_cache_resumes_without_recomputing_and_complete_cache_is_readonly(tmp_path, monkeypatch):
    source = tmp_path / "diverse.json"
    source.write_text("".join(json.dumps(dict(func=f"int example_{i}(void) {{ return value_{i}; }}",
                                              commit_id=f"{i:040x}", target=i % 2)) + "\n"
                              for i in range(5)), encoding="utf-8")
    output = tmp_path / "inputs" / "cache-receipts"
    output.mkdir(parents=True)
    cache = tmp_path / "shared-cache.sqlite"
    real_tokens = prep.lexical_token_set
    visited = []

    def interrupted(text):
        visited.append(text)
        if len(visited) == 4:
            raise RuntimeError("simulate interrupted cache build")
        return real_tokens(text)

    monkeypatch.setattr(prep, "lexical_token_set", interrupted)
    with pytest.raises(RuntimeError, match="interrupted cache"):
        prep.diverse_lexical_history(source, output, sha256(source), None, cache, checkpoint_rows=2)
    with sqlite3.connect(cache) as connection:
        progress = json.loads(connection.execute("SELECT value FROM metadata WHERE key='progress'").fetchone()[0])
        assert progress["processed_rows"] == 2 and not progress["complete"]
        assert connection.execute("SELECT COUNT(*) FROM exact_keys").fetchone()[0] == 2
    visited.clear()

    def counted(text):
        visited.append(text)
        return real_tokens(text)

    monkeypatch.setattr(prep, "lexical_token_set", counted)
    keys, receipt = prep.diverse_lexical_history(source, output, sha256(source), None, cache, checkpoint_rows=2)
    assert len(visited) == 3 and len(keys) == 5
    assert receipt["resumed_from_row"] == 2 and receipt["cache_complete"]
    completed_hash = sha256(cache)
    visited.clear()
    keys, receipt = prep.diverse_lexical_history(source, output, sha256(source), None, cache, checkpoint_rows=2)
    assert not visited and receipt["complete_cache_reused_read_only"]
    assert sha256(cache) == completed_hash


def test_missing_locator_does_not_hide_a_global_conflicting_label(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "aliases", lambda: {"alias": "wanted"})
    args = fixture_args(tmp_path)
    raw = json.loads(args.megavul.read_text(encoding="utf-8"))
    contradiction = record(raw[0]["func_before"], 30000, False)
    contradiction["repo_name"] = None
    raw.append(contradiction)
    args.megavul.write_text(json.dumps(raw), encoding="utf-8")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.update(sha256=sha256(args.megavul), bytes=args.megavul.stat().st_size)
    args.config.write_text(json.dumps(cfg), encoding="utf-8")
    result = prep.prepare(args)
    assert result["selected_projects"] == []  # 19 positives remain, below the frozen rule.
    assert result["raw_counts"]["missing_project_or_commit"] == 1
    assert result["within_megavul_conflicting_exact_keys"] == 2


def test_identity_rejected_row_still_quarantines_global_label_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "aliases", lambda: {"alias": "wanted"})
    args = fixture_args(tmp_path)
    raw = json.loads(args.megavul.read_text(encoding="utf-8"))
    raw.append(record(raw[0]["func_before"], 30000, False, project="LibreOffice/core"))
    args.megavul.write_text(json.dumps(raw), encoding="utf-8")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.update(sha256=sha256(args.megavul), bytes=args.megavul.stat().st_size)
    args.config.write_text(json.dumps(cfg), encoding="utf-8")
    result = prep.prepare(args)
    assert result["selected_projects"] == []  # Identity exclusion must not hide the 20th positive's contradiction.
    assert result["screened_counts"]["wanted"]["1"] == 19
    assert result["within_megavul_conflicting_exact_keys"] == 2
    assert result["raw_counts"]["repository_identity_not_allowlisted"] == 1
    assert result["repository_identity_exclusion_counts"]["core"] == {"repository_identity_not_allowlisted": 1}
    assert not result["repository_identity_mappings"]["LibreOffice/core"]["allowed"]
    with (args.output / "candidate_projects.csv").open(encoding="utf-8", newline="") as stream:
        audit = {row["project_group"]: row for row in csv.DictReader(stream)}
    assert json.loads(audit["core"]["observed_repositories"]) == ["LibreOffice/core"]
    assert json.loads(audit["core"]["matched_repositories"]) == []
    assert json.loads(audit["core"]["repository_identity_excluded_repositories"]) == {"LibreOffice/core": 1}


def test_unrelated_products_cannot_supply_core_or_server_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "aliases", lambda: {"alias": "wanted"})
    args = fixture_args(tmp_path)
    raw = json.loads(args.megavul.read_text(encoding="utf-8"))
    for project_index, project in enumerate(["dovecot/core", "LibreOffice/core", "ONLYOFFICE/core",
                                            "mariadb/server", "mysql/server"]):
        size = 199 if project in {"dovecot/core", "mariadb/server"} else 200
        raw.extend(record(f"int identity_{project_index}_{i}(void) {{ return identity_marker_{project_index}_{i}; }}",
                          40000 + project_index * 1000 + i, i < 20, project=project)
                   for i in range(size))
    args.megavul.write_text(json.dumps(raw), encoding="utf-8")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.update(sha256=sha256(args.megavul), bytes=args.megavul.stat().st_size)
    args.config.write_text(json.dumps(cfg), encoding="utf-8")
    result = prep.prepare(args)
    assert result["selected_projects"] == ["wanted"]
    assert len(result["candidate_projects"]) == 62
    assert result["screened_counts"]["core"] == result["screened_counts"]["server"] == {"1": 20, "0": 179}
    assert result["repository_identity_exclusion_counts"]["core"] == {"repository_identity_not_allowlisted": 400}
    assert result["repository_identity_exclusion_counts"]["server"] == {"repository_identity_not_allowlisted": 200}
    with (args.output / "candidate_projects.csv").open(encoding="utf-8", newline="") as stream:
        audit = {row["project_group"]: row for row in csv.DictReader(stream)}
    for group, repository in [("core", "dovecot/core"), ("server", "mariadb/server")]:
        assert audit[group]["selected"] == "False"
        assert audit[group]["eligibility_failures"] == "minimum_total"
        assert json.loads(audit[group]["matched_repositories"]) == [repository]


def test_hard_killed_cache_transaction_recovers_its_committed_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "raw.json"
    source.write_text("".join(json.dumps(dict(func=f"int row_{i}(void) {{ return marker_{i}; }}")) + "\n"
                              for i in range(5)), encoding="utf-8")
    output = tmp_path / "inputs" / "receipts"
    output.mkdir(parents=True)
    cache = tmp_path / "recover.sqlite"
    real_tokens = prep.lexical_token_set
    calls = []

    def stop_after_checkpoint(text):
        calls.append(text)
        if len(calls) == 3:
            raise RuntimeError("checkpoint fixture interruption")
        return real_tokens(text)

    monkeypatch.setattr(prep, "lexical_token_set", stop_after_checkpoint)
    with pytest.raises(RuntimeError):
        prep.diverse_lexical_history(source, output, sha256(source), None, cache, checkpoint_rows=2)
    # Force dirty pages to disk, then exit without close/rollback. This exercises
    # the same owned-cache recovery needed after the observed silent worker exit.
    program = """
import json,os,sqlite3,sys
c=sqlite3.connect(sys.argv[1])
c.execute('PRAGMA cache_size=1')
c.execute('BEGIN IMMEDIATE')
for i in range(5000):
    c.execute('INSERT OR IGNORE INTO exact_keys VALUES(?)',(f'{i:064x}',))
c.execute("UPDATE metadata SET value=? WHERE key='progress'",
          (json.dumps({'processed_rows':9999,'complete':False,'counts':{}}),))
os._exit(9)
"""
    killed = subprocess.run([sys.executable, "-c", program, str(cache)], capture_output=True, check=False)
    assert killed.returncode == 9 and cache.with_name(cache.name + "-journal").exists()
    monkeypatch.setattr(prep, "lexical_token_set", real_tokens)
    keys, receipt = prep.diverse_lexical_history(source, output, sha256(source), None, cache, checkpoint_rows=2)
    assert len(keys) == 5 and receipt["resumed_from_row"] == 2
    assert receipt["counts"]["raw_rows"] == 5 and receipt["exact_keys"] == 5
