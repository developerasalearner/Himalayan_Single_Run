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
2. Replace `artifacts/selected_features.json` with the real subsets from the
   manuscript run, if that has not been done already — see
   `docs/REPRODUCIBILITY.md`.
3. Resolve the two open discrepancies recorded in `docs/PAPER_MAPPING.md`
   (Equation 4's stated objective, and the archived notebook's missing
   `termreason` exclusion).
4. Flip the repository to public.
