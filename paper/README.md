# Paper

Deliberately empty for now.

## Before acceptance

Keep the repository private and leave the manuscript out of it. A draft under
review sitting in a repository — even a private one — is one misconfigured
setting away from being public, and some venues treat that as a problem.

## After acceptance

Add here:

- `paper.pdf` — the accepted version, **if** the venue's policy allows it.
  IEEE generally permits the accepted manuscript (post-peer-review, pre-copyedit)
  rather than the published version of record. Check the copyright form you
  signed before uploading anything.
- `figures/` — the exact PNGs used in the camera-ready, so the figures in the
  paper and the figures the code produces can be compared directly.
- `bibtex.bib` — the citation entry with its final DOI.

Then:

1. Fill in the DOI, pages and repository URL in `CITATION.cff`.
2. Check that the repository URL printed in Section IV of the manuscript matches
   the public repository name exactly.
3. Confirm `artifacts/selected_features.json` and
   `artifacts/selected_features_ablation.json` still hold the manuscript-run
   subsets — see `docs/REPRODUCIBILITY.md`.
4. Flip the repository to public.

The discrepancies that earlier revisions of `docs/PAPER_MAPPING.md` listed as open
are resolved in the current manuscript and recorded there under *Resolved
discrepancies*. Nothing in that list is outstanding.
