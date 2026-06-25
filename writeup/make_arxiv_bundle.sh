#!/usr/bin/env bash
# Build a clean, arXiv-ready source bundle (single directory + tar.gz) from the paper.
#
# arXiv compiles with pdfLaTeX (TeX Live 2025), so the bundle's main.tex gets \pdfoutput=1
# (which is omitted from the working copy because it breaks the local XeTeX/tectonic build).
# The bundle contains only the files arXiv needs: main.tex, arxiv.sty, references.bib, the
# precompiled main.bbl (so arXiv need not re-run bibtex), and exactly the included figures.
#
# Usage:  bash writeup/make_arxiv_bundle.sh
set -euo pipefail
cd "$(dirname "$0")"                       # writeup/
TECTONIC="${TECTONIC:-$HOME/lcc/bin/tectonic}"
OUT=arxiv_submission
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/figures"

# 1. main.tex with \pdfoutput=1 forced right after \documentclass.
awk '/^\\documentclass\{article\}/{print; print "\\pdfoutput=1"; next} {print}' main.tex > "$OUT/main.tex"

# 2. preprint style + bibliography database.
cp arxiv.sty references.bib "$OUT/"

# 3. precompiled bibliography, so arXiv does not depend on re-running bibtex.
"$TECTONIC" --keep-intermediates main.tex >/dev/null 2>&1 || true
if [ -f main.bbl ]; then
  cp main.bbl "$OUT/"
else
  echo "WARN: main.bbl not produced; arXiv will run bibtex on references.bib"
fi

# 4. only the figures actually included by main.tex (no extraneous files).
grep -oE 'figures/[A-Za-z0-9_]+\.pdf' main.tex | sort -u | while read -r fig; do
  cp "$fig" "$OUT/$fig"
done

# 5. drop local build leftovers, then tar.
rm -f main.aux main.log main.out main.bbl main.blg
tar czf "$OUT.tar.gz" "$OUT"

echo "=== arXiv bundle: $OUT.tar.gz ==="
find "$OUT" -type f | sort
echo "size: $(du -h "$OUT.tar.gz" | cut -f1)"
