from pathlib import Path

from verify_citations import parse_bibtex


def test_parse_bibtex_prefers_standard_doi_field(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    bibliography.write_text(
        """@article{standard,
  title = {A Standard DOI Entry},
  year = {2025},
  doi = {https://doi.org/10.1000/example.1}
}
""",
        encoding="utf-8",
    )

    [reference] = parse_bibtex(bibliography)

    assert reference.doi == "10.1000/example.1"


def test_parse_bibtex_keeps_note_doi_fallback(tmp_path: Path) -> None:
    bibliography = tmp_path / "references.bib"
    bibliography.write_text(
        """@inproceedings{legacy,
  title = {A Legacy DOI Entry},
  year = {2024},
  note = {doi: 10.1000/example.2}
}
""",
        encoding="utf-8",
    )

    [reference] = parse_bibtex(bibliography)

    assert reference.doi == "10.1000/example.2"
