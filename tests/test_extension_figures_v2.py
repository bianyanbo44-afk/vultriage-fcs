import matplotlib.pyplot as plt

from make_extension_v2_figures import configure_style, export


def test_figure_export_writes_submission_formats(tmp_path):
    configure_style()
    stem = tmp_path / "fig1_workflow"
    figure, axis = plt.subplots(figsize=(0.5, 0.4))
    axis.plot([0, 1], [0, 1])

    export(figure, stem)

    for suffix in (".pdf", ".png", ".svg", ".tiff"):
        path = stem.with_suffix(suffix)
        assert path.is_file()
        assert path.stat().st_size > 0
