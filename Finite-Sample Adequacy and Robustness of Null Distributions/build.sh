#!/usr/bin/env bash
# build.sh -- render the revised manuscript and the response letter to PDF.
#
# Uses pandoc + xelatex (both present at /opt/anaconda3/bin/pandoc and
# /Library/TeX/texbin/xelatex). Figures referenced from draft.md resolve under
# figures/, so the script must be run from this directory.
#
# Usage:  ./build.sh          (build both)
#         ./build.sh draft    (manuscript only)
#         ./build.sh letter   (response letter only)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TARGET="${1:-all}"

COMMON=(
  --pdf-engine=xelatex
  --include-in-header=latex_preamble.tex
  -V geometry:margin=1in
  -V fontsize=10pt
  --toc
  --toc-depth=2
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
  --standalone
)

build_draft() {
  echo "==> draft.pdf"
  pandoc draft.md "${COMMON[@]}" \
    -V documentclass=article \
    -V linestretch=1.15 \
    -o draft.pdf
  echo "    $(du -h draft.pdf | cut -f1)  draft.pdf"
}

build_letter() {
  echo "==> response_letter.pdf"
  pandoc response_letter.md "${COMMON[@]}" \
    -V documentclass=article \
    -V linestretch=1.15 \
    -o response_letter.pdf
  echo "    $(du -h response_letter.pdf | cut -f1)  response_letter.pdf"
}

# The editor's decision letter requires an editable manuscript source at this stage and
# states that a PDF is not acceptable as the manuscript file.  The DOCX build produces
# exactly the same Markdown through the same pandoc front end, with equations converted
# to native Word OMML so they remain editable in Word rather than being flattened to
# images.  Page layout will differ from the PDF because the reference DOCX styling is not
# the LaTeX stylesheet, so the PDF remains the typeset reference copy.
build_docx() {
  echo "==> draft.docx"
  pandoc draft.md \
    --from markdown+tex_math_dollars \
    --to docx \
    --toc --toc-depth=2 \
    --resource-path=.:figures \
    -o draft.docx
  echo "    $(du -h draft.docx | cut -f1)  draft.docx"
}

build_letter_docx() {
  echo "==> response_letter.docx"
  pandoc response_letter.md \
    --from markdown+tex_math_dollars \
    --to docx \
    --toc --toc-depth=2 \
    -o response_letter.docx
  echo "    $(du -h response_letter.docx | cut -f1)  response_letter.docx"
}

build_supp_docx() {
  echo "==> supplementary_material.docx"
  pandoc supplementary_material.md \
    --from markdown+tex_math_dollars \
    --to docx \
    --toc --toc-depth=2 \
    --resource-path=.:figures \
    -o supplementary_material.docx
  echo "    $(du -h supplementary_material.docx | cut -f1)  supplementary_material.docx"
}

case "$TARGET" in
  docx)   build_draft; build_docx; build_letter_docx; build_supp_docx ;;
  draft)  build_draft ;;
  letter) build_letter ;;
  all)    build_draft; build_letter ;;
  *) echo "usage: $0 [all|draft|letter]" >&2; exit 2 ;;
esac

echo "done."
