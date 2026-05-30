# EMNLP 2026 Anonymous Submission Pack

Two concurrent anonymous submissions on vocabulary-boundary asymmetry, with matched code releases and cross-citation supplementary material.

## Package layout

```
getcode/
├── SUBMISSION_PACK_README.md          ← this file
├── AGD_code_EMNLP2026_anonymous.zip   ← Paper 1 code + supplementary
├── ALoRA_code_EMNLP2026_anonymous.zip ← Paper 2 code + supplementary
├── 1/                                 ← AGD (unzipped source)
│   ├── main.pdf
│   ├── README.md
│   ├── src/ scripts/ analysis/
│   └── supplementary/                 ← upload with main PDF
└── 2/                                 ← A-LoRA (unzipped source)
    ├── main.pdf
    ├── README.md
    ├── src/ scripts/ analysis/
    └── supplementary/
```

## Paper mapping

| Label | Paper | Folder | Zip archive |
|-------|-------|--------|-------------|
| **Anonymous (2026a)** | Efficient Untying … **AGD** | `1/` | `AGD_code_EMNLP2026_anonymous.zip` |
| **Anonymous (2026b)** | Rethinking LoRA Placement … **A-LoRA** | `2/` | `ALoRA_code_EMNLP2026_anonymous.zip` |

## EMNLP 2026 originality checklist

- [ ] **Related Work updated in both PDFs** using `supplementary/related_work_companion_*.tex`
- [ ] **Anonymous bib entries added** from `supplementary/anonymous_bibliography_entries.bib`
- [ ] **Companion statement uploaded** (`supplementary/companion_submission_statement.md`) as anonymous supplementary material for **each** submission
- [ ] **Code zip uploaded** (or linked if venue permits anonymous code hosting)
- [ ] **No author-identifying paths** in uploaded materials (internal `im_exp/` / `get_useful/` paths removed from README)
- [ ] **Difference table** in companion statement makes clear: pretrain AGD vs. post-train A-LoRA

## What each paper must cite about the other

### AGD paper → cite A-LoRA (2026b)

- Concurrent post-training placement study
- Hidden-space affine adapter; conditional placement with hidden LoRA
- **Difference:** AGD = architectural pretrain untying; A-LoRA = PEFT on frozen weights

### A-LoRA paper → cite AGD (2026a)

- Concurrent pretraining-time untying via bias + hidden LR
- Cross-model output-dominated tying diagnosis
- **Difference:** A-LoRA = adapter placement & mergeability; AGD = pretrain architecture

## Suggested upload bundle per submission

1. Main paper PDF (`main.pdf`)
2. Anonymous supplementary PDF or ZIP containing:
   - `supplementary/companion_submission_statement.md`
   - `supplementary/related_work_companion_*.md` (backup of LaTeX snippet)
   - `supplementary/anonymous_bibliography_entries.bib`
3. Code ZIP (`AGD_…zip` or `ALoRA_…zip`)

## Rebuild zip archives

From this directory:

```bash
cd /path/to/getcode
rm -rf 1/src/agd/__pycache__ 2/src/alora/__pycache__
zip -r AGD_code_EMNLP2026_anonymous.zip 1 -x "1/src/agd/__pycache__/*" "1/**/__pycache__/*"
zip -r ALoRA_code_EMNLP2026_anonymous.zip 2 -x "2/src/alora/__pycache__/*" "2/**/__pycache__/*"
```

## After acceptance

Replace `Anonymous (2026a/b)` with real citations, add explicit cross-links in both camera-ready versions, and publish deanonymized code if required by the venue.
