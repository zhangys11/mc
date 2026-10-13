#!/usr/bin/env python3
"""Post-process the pandoc-generated DOCX into a submission-ready Word file.

Two things that pandoc does not do for a journal submission are fixed here:

1. A table-of-contents field is emitted, but Word only populates it when the document
   is opened with ``w:updateFields`` set; otherwise the reader sees an empty TOC.
2. Word's default table style lets wide tables run past the text margins.  Every table
   is given autofit and a preferred width of 100% of the text column, so the tables of
   the manuscript stay inside the page.

Run from the package root after ``./build.sh docx``:

    python3 code/postprocess_docx.py draft.docx
"""
from __future__ import annotations

import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches


def set_update_fields(doc):
    """Ask Word to refresh fields (the TOC) when the document is opened.

    Args:
        doc: the ``python-docx`` document to modify in place.
    """
    settings = doc.settings.element
    for existing in settings.findall(qn("w:updateFields")):
        settings.remove(existing)
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


def fit_tables(doc, width_in=6.5):
    """Give every table autofit layout and full text-column width.

    Args:
        doc: the ``python-docx`` document to modify in place.
        width_in: usable text width of the page in inches.

    Returns:
        int: the number of tables adjusted.
    """
    n = 0
    for table in doc.tables:
        table.autofit = True
        tbl_pr = table._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for existing in tbl_pr.findall(qn(tag)):
                tbl_pr.remove(existing)
        w = OxmlElement("w:tblW")
        w.set(qn("w:type"), "pct")
        w.set(qn("w:w"), "5000")        # 5000 fiftieths of a percent = 100%
        tbl_pr.append(w)
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "autofit")
        tbl_pr.append(layout)
        n += 1
    return n


def main():
    """Apply the post-processing to the file named on the command line."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    doc = Document(path)
    set_update_fields(doc)
    n = fit_tables(doc)
    doc.save(path)
    print(f"post-processed {path}: TOC refresh enabled, {n} tables fitted to the text column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
