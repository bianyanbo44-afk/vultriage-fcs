# Frontiers of Computer Science: submission requirements (v2)

**Checked:** 2026-08-14 (China Standard Time)  
**Scope:** venue fit and manuscript-package requirements for the VulTriage
research article. This note is a checklist, not a substitute for the portal's
current fields or an editorial decision.

## Verified official sources

- [Aims and scope](https://link.springer.com/journal/11704/aims-and-scope),
  accessed 2026-08-14. The journal explicitly covers software, artificial
  intelligence, information security, and interdisciplinary computer science;
  the proposed vulnerability-detection/project-shift study is in scope when
  presented as a software-security and deployment-methods contribution.
- [Submission guidelines](https://link.springer.com/journal/11704/submission-guidelines),
  accessed 2026-08-14. The page links the official [Instructions for
  Authors (2025 PDF)](https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/11704_Instructions%20for%20Athours_2025.pdf),
  SHA-256 `7487E98DE1FD07329C64B6FC1636E6C0E07594A7FEBC0C9B53BB0AD46F1ECB2E`.
- [How to publish with us](https://link.springer.com/journal/11704/how-to-publish-with-us),
  accessed 2026-08-14. FCS is hybrid: subscription publication has no APC;
  the page displayed an OA APC of GBP 2,390 / USD 3,350 / EUR 2,740 and CC BY
  for OA articles. Fees and available institutional agreements can change.
- [Ethics & disclosures](https://link.springer.com/journal/11704/ethics-and-disclosures),
  accessed 2026-08-14. The journal states a competing-interest policy, COPE
  membership, research-integrity standards, and single-anonymous peer review.
- [Submit manuscript](https://mc.manuscriptcentral.com/hepfcs), linked from
  the journal page and the author PDF.

## Hard manuscript requirements

The official PDF describes a regular **Research Article** (not a Letter or
Review) and requires: title; all authors and affiliations; one-paragraph
abstract of no more than 300 words; up to eight semicolon-separated keywords;
main text; acknowledgements; competing interests; references; figure captions;
and tables. Nomenclature and appendices are conditional. No formal total
paper-length limit is stated, although editors may request condensation.

Formatting and presentation checks:

- Use Word or LaTeX. The initial portal upload is a PDF; source files are
  requested after acceptance. The current FCS class/template is therefore
  appropriate, but the final source and PDF must be synchronized.
- Use the citation-sequence system. In-text numbers must follow first
  appearance, agree exactly with the reference list, and use full journal
  names. Include DOI links where useful and verify every bibliographic record.
- Expand abbreviations at first use and keep them consistent. Identify and
  style mathematical symbols; write `Equation/Equations` at sentence starts
  and `Eq./Eqs.` elsewhere. Avoid essential footnotes.
- Use three-line tables (three horizontal rules, normally no vertical rules).
- Number figures in order of citation, give every figure a title/caption, and
  label subfigures `(a)`, `(b)`, etc. Embed figures in the manuscript. The PDF
  specifies 300 dpi for colour images, 600 dpi for monochrome and line art,
  and asks for TIFF/EPS/Corel-Draw originals after acceptance. Keep the current
  vector PDF/SVG plus high-resolution TIFF exports until production requests
  the original format.
- Provide author biographies of no more than 120 words each and photographs
  if requested by the production workflow; do not fabricate biographies or
  photos.

## Required declarations and portal checks

- Include acknowledgements/funding, competing interests, author contributions,
  and an ethics/data-use statement appropriate to the study. For this project,
  state no specific funding, no competing interests, public defensive-use
  datasets, no human participants, and the data/code availability boundary.
- Enter **all authors'** affiliations and contact details in the submission
  system, not only the corresponding author's email. Confirm author order,
  corresponding-author flags, institutional names, and the verified Wang
  contact before the final click; the guide says all authors must confirm the
  submission before it is sent for review.
- Select Research Article in the portal. The cover letter should explain fit,
  originality, and that the work is not under simultaneous consideration.
- Upload the three-page PPT Highlights file separately if the portal requests
  it. The author PDF says Highlights are displayed online as supplementary
  material and do not appear in the article PDF; do not substitute the DOCX
  compatibility artifact.
- Keep public-repository, data-availability, and exact-result hashes stable,
  but do not upload raw DiverseVul, the label vault, private SQLite files, or
  large embedding/prediction caches unless the journal specifically requests
  them and redistribution is permitted.

## v2 package actions before submission

1. Rebuild the manuscript, figures, captions, references, PDF, DOCX, cover
   letter, Highlights PPTX, citation report, and checksum manifest from the
   sealed v2 evaluation outputs. Remove every stale E1-only number or hash.
2. Ensure the abstract remains one paragraph and under 300 words after the
   DiverseVul/CodeBERT results are inserted; keep the positive conclusion
   bounded and distinguish support qualification from target-risk validation.
3. Re-run citation-order, DOI, figure-resolution, table-rule, abbreviation,
   declaration, and author-metadata checks. Verify that every figure/table is
   cited and that the PDF and editable source contain identical content.
4. At upload, retain the portal receipt/manuscript dispatch number. After
   acceptance, return proofs within the journal's stated three-day window and
   provide requested original figure/source files and the selected copyright
   or OA statement.
