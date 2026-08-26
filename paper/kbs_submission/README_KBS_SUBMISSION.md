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

The KBS version presents VulTriage as a proof-carrying detector--shift knowledge
contract. A qualification registry binds automation authority to a detector,
source domain, target domain, and policy version. Four three-valued predicates
govern budget validity, evidence completeness, gate support, and singleton
eligibility; the fifth rule fails closed and requires a verifiable certificate.

Hashing-SGD registers at Q=0.757 and confirms at Q=0.917. Both deterministic
12-project halves qualify in 84.94% of 10,000 project partitions. On the fixed
confirmation half, all six non-severe projects are eligible and five of six
severe projects are review-only. The same registry withholds CodeBERT authority.
The empirical foundation remains the hashing-SGD risk-alignment gain across all
24 external projects and 202 of 216 frozen project--budget cells.

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
