"""Meaningful boundary and full-encoder update checks for the new family."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import T5Config, T5EncoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from independent_model_v4_1 import CodeT5Detector, encoder_digests, length_batches
from evaluate_independent_model_v4_1 import _verify_training_manifest
from prepare_independent_model_v4_1 import read_rows, role_receipts
from prepare_independent_model_v4_1 import corrections
from audit_independent_model_v4 import actual_timestamp, extract_metadata, strict_temporal_rule


def tiny_detector(seed=13):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    config = T5Config(vocab_size=48, d_model=32, d_ff=64, num_layers=2, num_heads=4, d_kv=8, dropout_rate=0.0)
    return CodeT5Detector(encoder=T5EncoderModel(config))


def test_instantiated_encoder_only_receives_real_updates_in_every_block():
    model = tiny_detector()
    assert not any("decoder" in name for name, _ in model.named_parameters())
    assert all(p.requires_grad for p in model.parameters())
    initial = encoder_digests(model)
    batch = dict(input_ids=torch.tensor([[1, 4, 9, 2], [1, 6, 12, 2]]),
                 attention_mask=torch.ones((2, 4), dtype=torch.int64),
                 special_tokens_mask=torch.tensor([[True, False, False, True]] * 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits, pooled = model(**batch)
    assert pooled.shape == (2, 32) and logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0) for p in model.parameters())
    optimizer.step()
    final = encoder_digests(model)
    query_names = [n for n in initial if n.endswith("SelfAttention.q.weight")]
    assert len(query_names) == 2 and all(initial[n] != final[n] for n in query_names)


def test_all_special_pooling_is_rejected():
    model = tiny_detector()
    with pytest.raises(ValueError, match="nonspecial"):
        model(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.int64), torch.ones((1, 2), dtype=torch.bool))


def actual_manifest():
    return dict(schema_version="trace-independent-model-v4.1-actual-fit-1", seed=13, config_sha256="A" * 64,
                target_labels_accessed=False, actual_independent_fit=True, model_class="T5EncoderModel",
                model_id="Salesforce/codet5-small", model_revision="b1ee9570c289f21b5922b9c768a1ce12957bf968",
                unused_pretrained_decoder_instantiated=False, unused_pretrained_decoder_trainable_parameters=0,
                encoder_query_relative_updates={f"block.{i}.q": 0.01 for i in range(6)},
                encoder_initial_tensor_hashes={"q": "before"}, encoder_final_tensor_hashes={"q": "after"},
                changed_encoder_tensor_count=1, encoder_parameters=10000, classifier_parameters=1026,
                trainable_parameters=11026, total_parameters=11026, optimizer_steps=1464,
                source_role_hashes={"train": {"rows": 15594}},
                initial_gradient_evidence={"q": {"received_gradient": True, "finite": True, "nonzero": True}})


def test_old_family_or_unused_decoder_cannot_pass_actual_fit_validation():
    original = actual_manifest()
    _verify_training_manifest(original, "finetuned_codet5_small", 13, "A" * 64)
    for field, value in [("model_class", "RobertaModel"), ("unused_pretrained_decoder_instantiated", True),
                         ("actual_independent_fit", False), ("optimizer_steps", 1), ("classifier_parameters", 1538)]:
        bad = copy.deepcopy(original)
        bad[field] = value
        with pytest.raises(ValueError):
            _verify_training_manifest(bad, "finetuned_codet5_small", 13, "A" * 64)
    bad = copy.deepcopy(original)
    bad["encoder_query_relative_updates"]["block.5.q"] = 0.0
    with pytest.raises(ValueError, match="updating"):
        _verify_training_manifest(bad, "finetuned_codet5_small", 13, "A" * 64)


def test_roles_hash_binds_code_labels_assignment_and_order():
    rows = [dict(position=i, row_id=str(i), commit_id=f"{i + 1:040x}", role=role, label=i % 2, text=f"code {i}")
            for i, role in enumerate(["train", "train", "model_validation", "calibration"])]
    expected = role_receipts(rows)
    for field, value in [("text", "changed code"), ("label", 1), ("role", "calibration")]:
        changed = copy.deepcopy(rows)
        changed[0][field] = value
        assert role_receipts(changed) != expected
    changed = copy.deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    assert role_receipts(changed) != expected


def test_target_label_field_rejected_before_tokenization(tmp_path):
    row = dict(position=0, row_id="one", project_group="target", commit_id="a" * 40, label=1, text="code")
    path = tmp_path / "target_rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        read_rows(path, source=False)


def test_seeded_batches_cover_training_once_and_vary_across_seeds():
    cache = type("Cache", (), {"lengths": np.arange(257) % 512 + 1})()
    positions = np.arange(257)
    a, b = length_batches(cache, positions, 4, 13), length_batches(cache, positions, 4, 37)
    assert np.array_equal(np.sort(np.concatenate(a)), positions)
    assert np.array_equal(np.sort(np.concatenate(b)), positions)
    assert not np.array_equal(np.concatenate(a), np.concatenate(b))


def test_strict_temporal_subset_requires_every_source_commit_date():
    source = [dict(commit_id="a" * 40, dates=[10]), dict(commit_id="b" * 40, dates=[20])]
    target = [dict(row_id="same-day", dates=[20]), dict(row_id="later", dates=[21]), dict(row_id="missing", dates=[])]
    complete = strict_temporal_rule(source, target)
    assert complete["available"] is True and complete["cutoff_timestamp"] == 20
    assert complete["eligible_row_ids"] == ["later"]
    incomplete = copy.deepcopy(source)
    incomplete[0]["dates"] = []
    refused = strict_temporal_rule(incomplete, target)
    assert refused["available"] is False and refused["cutoff_timestamp"] is None and refused["eligible_row_ids"] == []
    conflicting = copy.deepcopy(source)
    conflicting[0]["dates"] = [10, 11]
    assert strict_temporal_rule(conflicting, target)["available"] is False


def test_cve_message_mentions_and_publication_dates_do_not_fabricate_metadata():
    raw = dict(message="Fix CVE-2020-12345", cwe="CWE-79", publish_date="2020-01-01T00:00:00+00:00",
               commit_id="a" * 40)
    meta = extract_metadata(raw, provenance="test")
    assert meta["cve_ids"] == [] and meta["dates"] == []
    assert meta["cve_message_mentions"] == ["CVE-2020-12345"]
    assert actual_timestamp("2020-01-01") is None and actual_timestamp(True) is None
    assert actual_timestamp("2020-01-01T00:00:00+00:00") == 1577836800


def test_family_filter_closes_complete_commit_and_original_event_sets_remain_conservative():
    def row(position, role, commit, project):
        return dict(position=position, row_id=str(position), role=role, commit_id=commit * 40,
                    project=project, project_group=project, label=position % 2)
    source = [row(0, "train", "a", "abrt"), row(1, "train", "a", "other-alias"),
              row(2, "train", "b", "unrelated"), row(3, "model_validation", "c", "valid"),
              row(4, "calibration", "d", "cal"), row(5, "model_validation", "e", "unknown")]
    metadata = {r["row_id"]: dict(repositories=[], cve_ids=[]) for r in source}
    metadata["0"]["cve_ids"] = ["CVE-2020-10001"]
    metadata["3"]["cve_ids"] = ["CVE-2020-10001"]
    metadata["4"]["cve_ids"] = ["CVE-2020-10001"]
    kept, removed, details = corrections(source, metadata)
    assert [r["row_id"] for r in kept] == ["2", "5"]
    assert [r["original_position"] for r in kept] == [2, 5]
    assert [r["position"] for r in kept] == [0, 1]
    assert {r["stage"] for r in removed} == {"known_family_complete_commit", "validation_CVE_present_in_original_train", "calibration_CVE_present_in_original_train_or_validation"}
    assert "a" * 40 in details["family_commits"]
    # A family-filtered original train commit still excludes its known event from later roles.
    assert "c" * 40 in details["validation_event_commits"] and "d" * 40 in details["calibration_event_commits"]
