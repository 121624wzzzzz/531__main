# Supplementary Material — AGD (Anonymous EMNLP 2026 Submission)

This folder supports **EMNLP 2026 originality / thin-slicing compliance** for a concurrent companion submission on vocabulary-boundary adaptation.

## Contents

| File | Purpose |
|------|---------|
| `companion_submission_statement.md` | Formal note to reviewers/area chairs about the concurrent anonymous companion paper |
| `related_work_companion_alora.tex` | Paste-ready **Related Work** paragraph citing the companion A-LoRA submission |
| `related_work_companion_alora.md` | Markdown version of the same text |
| `anonymous_bibliography_entries.bib` | Anonymous `\bibitem` / BibTeX entries for both concurrent papers |

## How to use at submission time

1. **Main paper PDF**: upload `../main.pdf` as the primary submission.
2. **Code**: upload the zipped release (`AGD_code_EMNLP2026_anonymous.zip`) or link if the venue allows anonymous code.
3. **This supplementary folder**: upload as **anonymous supplementary material** together with the main PDF, or merge into a single supplementary PDF if the portal requires one file.
4. **Related Work**: insert the paragraph from `related_work_companion_alora.tex` (or `.md`) into Section 2 of the main paper **before final submission**.
5. **Bibliography**: add the entries from `anonymous_bibliography_entries.bib` using labels `Anonymous2026a` (this paper) and `Anonymous2026b` (companion A-LoRA paper).

## Anonymity checklist

- Do **not** upload internal lab paths, private repos, or author-identifying experiment logs.
- Use only the anonymized code release in `../` (no upstream project paths in README).
- Cite the companion work as **Anonymous (2026b)** until deanonymization.
