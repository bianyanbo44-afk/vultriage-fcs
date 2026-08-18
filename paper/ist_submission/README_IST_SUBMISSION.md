# IST submission package

Target journal: *Information and Software Technology* (Elsevier).

## Upload files

- `VulTriage_IST_manuscript.pdf` - compiled initial-submission manuscript.
- `highlights.txt` - four journal highlights.
- `cover_letter.en.md` - cover letter text to paste into the submission system or convert to the requested format.
- `Declaration_of_Interest.docx` - separate Elsevier declaration-of-interest upload (kept in the local submission package, not this public snapshot).

The source bundle (`main.tex`, `references.bib`, `figures/`, `elsarticle.cls`, and `elsarticle-num-names.bst`) is retained for editorial requests and reproducibility. A main-manuscript Word file is not included because the LaTeX/PDF route is sufficient for the initial submission unless the portal explicitly requests DOCX.

The declaration file is an administrative upload and is intentionally excluded
from this public reproducibility snapshot.

## Build

From this directory:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses Elsevier's `elsarticle` class and numeric references. The compiled PDF was visually checked after rendering representative pages, including the title/abstract page, project table, figures, declarations, and references.

## Final author checks

Before dispatch, enter the corresponding author in the portal as given name
`Yanbo`, family name `Bian`, and flag only Yanbo Bian as corresponding. Both
authors should confirm the author order, Yanbo Bian's corresponding-author
email, and the CRediT statement. The public artifact link
in the manuscript is:

https://github.com/bianyanbo44-afk/vultriage-fcs

The manuscript reports a frozen, prediction-sealed external-evaluation protocol.
It does not claim public preregistration, a distribution-free target-risk
guarantee, detector-general validity, or a universal speedup.

The companion public artifact is maintained at
<https://github.com/bianyanbo44-afk/vultriage-fcs>. The repository name is a
historical project identifier; its current README and
`paper/ist_submission/` source are aligned with this IST submission. The
submission package remains the authoritative copy of the IST manuscript and
its cover-letter materials.
