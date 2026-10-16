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


def has_fields(doc):
    """Report whether the document contains a field (a TOC, in practice).

    Args:
        doc: the ``python-docx`` document to inspect.

    Returns:
        bool: True when at least one ``w:instrText`` element is present.
    """
    return doc.element.body.findall(".//" + qn("w:instrText")) != []


def set_update_fields(doc):
    """Ask Word to refresh the table of contents when the document is opened.

    The manuscript and the point-by-point response are built without a table of contents,
    so they carry no field to refresh; setting ``w:updateFields`` on them would make Word
    offer to update fields that do not exist.  The flag is therefore written only when the
    document actually contains one.

    Args:
        doc: the ``python-docx`` document to modify in place.

    Returns:
        bool: True if the flag was written, False if there was no field to refresh.
    """
    settings = doc.settings.element
    for existing in settings.findall(qn("w:updateFields")):
        settings.remove(existing)
    if not has_fields(doc):
        return False
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)
    return True


def drop_duplicate_figure_labels(doc):
    """Remove the figure labels pandoc copies from image alternative text.

    Pandoc writes a Markdown image's alternative text into a paragraph styled
    ``ImageCaption`` placed directly above the image. Every figure in this manuscript
    already carries its own numbered caption below it, so the result is a duplicated and
    sometimes inconsistent label ("Study 3: Monte Carlo precision" above "Figure 3:
    ..."). The alternative text is retained in the drawing description for screen
    readers, so nothing is lost by dropping the visible copy.

    The paragraphs are removed at the XML level because their style name is exposed by
    python-docx as ``Image Caption`` while the stored style id is ``ImageCaption``, and
    matching on the id is the reliable test.

    Args:
        doc: the ``python-docx`` document to modify in place.

    Returns:
        int: the number of label paragraphs removed.
    """
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    removed = 0
    for para in list(doc.element.body.iter(w + "p")):
        style = para.find(w + "pPr/" + w + "pStyle")
        if style is None or style.get(w + "val") != "ImageCaption":
            continue
        para.getparent().remove(para)
        removed += 1
    return removed


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
    toc = set_update_fields(doc)
    dup = drop_duplicate_figure_labels(doc)
    n = fit_tables(doc)
    doc.save(path)
    note = "TOC refresh enabled" if toc else "no TOC field (nothing to refresh)"
    print(f"post-processed {path}: {note}, {dup} duplicate figure labels removed, "
          f"{n} tables fitted to the text column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
