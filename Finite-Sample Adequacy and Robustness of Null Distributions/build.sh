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
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
  --standalone
)

# A table of contents is a navigation aid, not part of a submitted manuscript, so it is
# added only to the documents that are read standalone and are not typeset by the journal:
# the reproduction vignette and the supplementary material.  The manuscript and the
# point-by-point response are built without one.
TOC=(--toc --toc-depth=2)

# Figures are built in two steps.  Pandoc alone would leave each numbered caption outside
# its float, so LaTeX could set the caption on one page and the image on the next; Pandoc
# also uses the Markdown image's alternative text as the float caption, which produced a
# second, unnumbered caption.  code/pin_figures.py rewrites the intermediate .tex so the
# real caption sits inside a non-floating figure environment, then xelatex typesets it.
render_pdf() {
  local src="$1" out="$2"; shift 2
  local tex
  mkdir -p .build
  tex=".build/${out%.pdf}.tex"
  pandoc "$src" "${COMMON[@]}" "$@" -t latex -o "$tex"
  python3 code/pin_figures.py "$tex"
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=.build "$tex" >/dev/null 2>&1
  cp ".build/${out%.pdf}.pdf" "$out"
}

build_draft() {
  echo "==> draft.pdf"
  render_pdf draft.md draft.pdf -V documentclass=article -V linestretch=1.15
  echo "    $(du -h draft.pdf | cut -f1)  draft.pdf"
}

build_letter() {
  echo "==> response_letter.pdf"
  render_pdf response_letter.md response_letter.pdf -V documentclass=article -V linestretch=1.15
  echo "    $(du -h response_letter.pdf | cut -f1)  response_letter.pdf"
}

build_supp() {
  echo "==> supplementary_material.pdf"
  render_pdf supplementary_material.md supplementary_material.pdf -V documentclass=article -V linestretch=1.15 "${TOC[@]}"
  echo "    $(du -h supplementary_material.pdf | cut -f1)  supplementary_material.pdf"
}

build_vignette() {
  echo "==> vignette.pdf"
  render_pdf vignette.md vignette.pdf -V documentclass=article -V linestretch=1.15 -V geometry:margin=1.9cm -V mainfont="Helvetica Neue" -V monofont="Menlo" --highlight-style=tango "${TOC[@]}"
  echo "    $(du -h vignette.pdf | cut -f1)  vignette.pdf"
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
    --resource-path=.:figures \
    -o draft.docx
  echo "    $(du -h draft.docx | cut -f1)  draft.docx"
}

build_letter_docx() {
  echo "==> response_letter.docx"
  pandoc response_letter.md \
    --from markdown+tex_math_dollars \
    --to docx \
    -o response_letter.docx
  echo "    $(du -h response_letter.docx | cut -f1)  response_letter.docx"
}

build_supp_docx() {
  echo "==> supplementary_material.docx"
  pandoc supplementary_material.md \
    --from markdown+tex_math_dollars \
    --to docx \
    "${TOC[@]}" \
    --resource-path=.:figures \
    -o supplementary_material.docx
  echo "    $(du -h supplementary_material.docx | cut -f1)  supplementary_material.docx"
}

case "$TARGET" in
  docx)   build_draft; build_docx; build_letter_docx; build_supp_docx ;;
  draft)  build_draft ;;
  letter) build_letter ;;
  supp)   build_supp ;;
  vignette) build_vignette ;;
  all)    build_draft; build_letter; build_supp; build_vignette ;;
  *) echo "usage: $0 [all|draft|letter|supp|vignette|docx]" >&2; exit 2 ;;
esac

echo "done."
