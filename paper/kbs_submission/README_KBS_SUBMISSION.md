# KBS submission package

Target journal: *Knowledge-Based Systems* (Elsevier).

## Upload files

- `VulTriage_KBS_manuscript.pdf` - compiled manuscript for the KBS submission.
- `highlights.txt` - four journal highlights.
- `cover_letter.en.md` - cover letter text to paste into the submission system.
- The separate Elsevier declaration-of-interest DOCX is kept in the local
  portal package and is intentionally omitted from this public directory.
- `main.tex`, `references.bib`, `figures/`, `elsarticle.cls`, and
  `elsarticle-num-names.bst` - editable source and figure package.

The clean upload archive should contain only the files listed above and should
not include the older IST staging directories that remain elsewhere in this
working folder.

## Build

From this directory:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses Elsevier's `elsarticle` class and numeric references. The
compiled PDF was checked for a clean build, figure references, declarations,
and bibliography completeness.

## Scope and evidence boundary

The KBS version frames VulTriage as a knowledge-guided selective decision-
support layer. The explicit knowledge set contains the asymmetric risk
budgets, evidence-sufficiency constraints, severe-risk threshold, and
fail-closed action semantics. It is an auditable prototype policy, not an
expert-elicited ontology or a general-purpose knowledge graph.

The reported evidence remains detector-conditional. Weighting improves the
hashing-SGD risk-alignment outcome at the primary operating point, while the
same overall improvement is not observed for frozen CodeBERT. The support gate
has external discrimination for hashing and not for CodeBERT. The manuscript
does not claim universal target-risk guarantees, detector-general validity,
SOTA detector performance, or a speed/energy advantage.

## Mandatory author and submission checks

Before uploading to KBS:

1. Confirm that any earlier submission of this manuscript is formally withdrawn
   or closed. The cover letter's originality statement must be true at the time
   of KBS submission; do not submit the same manuscript simultaneously to two
   journals.
2. Enter given name `Yanbo`, family name `Bian`, and flag only Yanbo Bian as
   corresponding author (`24722081@bjtu.edu.cn`).
3. Confirm the author order, CRediT statement, AI-use disclosure, funding,
   competing-interest statement, and the separate declaration-of-interest
   upload with both authors.
4. Verify the portal's current KBS article-type, file-size, and data/code
   questions before final dispatch.

The public reproducibility artifact is:

https://github.com/bianyanbo44-afk/vultriage-fcs

It contains the frozen protocol, public metadata, evaluation and analysis
code, summary tables, figure source data, and validation manifests. It excludes
raw source datasets, target-label vaults, private indexes, embeddings, feature
caches, and per-function prediction or decision archives.
