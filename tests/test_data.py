from vultriage.data import alias_lookup, canonical_project, source_role, stable_bucket


CONFIG = {
    "split_salt": "vultriage-v1",
    "source_partition": {
        "train_end": 70,
        "model_validation_end": 80,
        "calibration_end": 100,
    },
    "target_groups": {
        "linux": ["linux", "linux-2.6"],
        "qemu": ["qemu", "qemu-kvm"],
    },
}


def test_alias_mapping_is_explicit_and_stable():
    lookup = alias_lookup(CONFIG)
    assert canonical_project("linux-2.6", lookup) == "linux"
    assert canonical_project("unknown", lookup) == "unknown"


def test_stable_bucket_and_source_role_are_deterministic():
    first = stable_bucket("abc123", "vultriage-v1")
    second = stable_bucket("abc123", "vultriage-v1")
    assert first == second
    assert 0 <= first < 100
    assert source_role("abc123", CONFIG) in {
        "train",
        "model_validation",
        "calibration",
    }

