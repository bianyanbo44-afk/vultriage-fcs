import numpy as np

from vultriage.near_duplicate import (
    AuditDocument,
    NearDuplicateIndex,
    cardinality_can_reach_threshold,
    cpp_lexical_tokens,
    deserialize_token_set,
    exact_jaccard_counts,
    lexical_token_set,
    lsh_band_keys,
    lsh_candidate_probability,
    minhash_coefficients,
    minhash_signature,
    serialize_token_set,
)


def document(dataset: str, row_id: str, text: str) -> AuditDocument:
    return AuditDocument(
        dataset=dataset,
        row_id=row_id,
        source_file=f"{dataset}.jsonl",
        line_number=1,
        project=dataset,
        project_group=dataset,
        exact_code_key=(row_id * 64)[:64],
        tokens=lexical_token_set(text),
    )


def test_cpp_tokenizer_uses_longest_match_and_omits_comments_and_literals():
    source = r'''
        // natural language and fake_call()
        const char *s = "ignored >= words";
        char c = '\'';
        value >>= 2;
        ptr->*member;
        x %:%: y;
        auto raw = R"tag(ignored && content)tag";
    '''
    assert cpp_lexical_tokens(source) == (
        "const", "char", "*", "s", "=", ";", "char", "c", "=", ";",
        "value", ">>=", "2", ";", "ptr", "->*", "member", ";", "x",
        "%:%:", "y", ";", "auto", "raw", "=", ";",
    )


def test_token_set_is_unique_case_sensitive_and_round_trips():
    tokens = lexical_token_set("int X = x + x; int X = 0x10;")
    assert tokens.count("x") == 1
    assert "X" in tokens and "x" in tokens and "0x10" in tokens
    assert deserialize_token_set(serialize_token_set(tokens)) == tokens


def test_minhash_is_deterministic_and_uses_all_128_permutations():
    coefficients_a = minhash_coefficients(128)
    coefficients_b = minhash_coefficients(128)
    assert np.array_equal(coefficients_a[0], coefficients_b[0])
    assert np.array_equal(coefficients_a[1], coefficients_b[1])
    signature_a = minhash_signature(("a", "b", "c"), coefficients_a)
    signature_b = minhash_signature(("c", "b", "a"), coefficients_b)
    assert signature_a.shape == (128,)
    assert np.array_equal(signature_a, signature_b)
    assert len(lsh_band_keys(signature_a, bands=16, rows_per_band=8)) == 16


def test_exact_jaccard_and_cardinality_upper_bound_at_frozen_threshold():
    intersection, union, similarity = exact_jaccard_counts(
        tuple(str(i) for i in range(10)), tuple(str(i) for i in range(9))
    )
    assert (intersection, union, similarity) == (9, 10, 0.9)
    assert cardinality_can_reach_threshold(9, 10, 0.9)
    assert not cardinality_can_reach_threshold(8, 10, 0.9)
    assert lsh_candidate_probability(0.9, 16, 8) > 0.9998


def test_sqlite_lsh_deduplicates_multi_band_candidates_and_exactly_flags(tmp_path):
    path = tmp_path / "near.sqlite"
    base = " ".join(f"token{i}" for i in range(10))
    near = " ".join(f"token{i}" for i in range(9))
    far = "totally different lexical content"
    with NearDuplicateIndex(path) as index:
        index.register_prime_key("p1")
        assert not index.register_prime_key("p1")
        index.add_document(document("primevul", "prime", base))
        index.add_document(document("diversevul", "target-near", near))
        index.add_document(document("diversevul", "target-far", far))
        index.finish_documents()
        list(index.generate_candidates(threshold=0.9, batch_size=1))
        flagged = list(index.verify_candidates(0.9))
        rerun_flagged = list(index.verify_candidates(0.9))

        assert index.candidate_count() == 1
        assert index.candidate_target_count() == 1
        assert index.flagged_pair_count() == 1
        assert index.flagged_target_count() == 1
        assert len(flagged) == 1
        assert len(rerun_flagged) == 1
        pair = next(index.iter_flagged_pairs())
        assert pair["target_row_id"] == "target-near"
        assert pair["exact_jaccard"] == 0.9
        assert next(index.iter_flagged_targets())["flagged_prime_pair_count"] == 1


def test_exact_threshold_rejects_candidate_below_point_nine(tmp_path):
    path = tmp_path / "threshold.sqlite"
    shared = tuple(f"token{i}" for i in range(89))
    left_tokens = shared + tuple(f"left{i}" for i in range(5))
    right_tokens = shared + tuple(f"right{i}" for i in range(5))
    left = document("primevul", "prime", "")
    right = document("diversevul", "target", "")
    left = AuditDocument(**{**left.__dict__, "tokens": tuple(sorted(left_tokens))})
    right = AuditDocument(**{**right.__dict__, "tokens": tuple(sorted(right_tokens))})
    with NearDuplicateIndex(path) as index:
        prime_id = index.add_document(left)
        target_id = index.add_document(right)
        index.connection.execute(
            "INSERT INTO candidates(target_doc_id, prime_doc_id) VALUES (?, ?)",
            (target_id, prime_id),
        )
        assert list(index.verify_candidates(0.9)) == []
        assert index.flagged_pair_count() == 0
