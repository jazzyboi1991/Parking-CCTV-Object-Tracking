#!/usr/bin/env python3
"""Apply the original Word template's formatting to the existing report draft."""
from copy import deepcopy
from pathlib import Path
import re

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent / "보고서"
TEMPLATE = ROOT / "충북대학교_Machine_Learning_HW1 최종본.docx"
DRAFT = ROOT / "머신러닝 객체 탐지 과제 보고서.docx"
OUTPUT = ROOT / "머신러닝 객체 탐지 보고서 서식 적용.docx"


def replace_text(paragraph, text):
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def clone_with_text(source, text):
    element = deepcopy(source._p)
    for child in list(element):
        if child.tag != qn("w:pPr"):
            element.remove(child)
    run = deepcopy(source.runs[0]._r) if source.runs else None
    if run is None:
        raise ValueError("Template paragraph lacks a formatted run")
    for child in list(run):
        if child.tag != qn("w:rPr"):
            run.remove(child)
    from docx.oxml import OxmlElement
    node = OxmlElement("w:t")
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    run.append(node)
    element.append(run)
    return element


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists: {OUTPUT}")
    template = Document(TEMPLATE)
    draft = Document(DRAFT)
    paragraphs = template.paragraphs
    def find(text):
        return next(p for p in paragraphs if text in p.text)
    title = find("Cloud Computing Homework 1")
    subtitle = find("문제 및 답안 작성 양식")
    heading = find("교재 연습문제 답안")
    body_example = find("과제 지시사항")
    draft_paras = [p.text.strip() for p in draft.paragraphs if p.text.strip()]
    start = next(i for i, text in enumerate(draft_paras) if text == "1. 과제 목적")
    report = draft_paras[start:]
    if len(report) < 10:
        raise SystemExit("Report draft content is incomplete")
    body = template.element.body
    marker = title._p
    title_xml = clone_with_text(title, "주차장 영상의 객체 탐지 보고서")
    subtitle_xml = clone_with_text(subtitle, "머신러닝 과제 수행 보고서")
    headings = []
    for text in report:
        if re.match(r"^[1-7]\. ", text):
            headings.append(clone_with_text(heading, text))
        else:
            # Copy the regular text run of the template's instruction paragraph.
            normal = deepcopy(body_example._p)
            for child in list(normal):
                if child.tag != qn("w:pPr"):
                    normal.remove(child)
            source_run = body_example.runs[-1]
            run = deepcopy(source_run._r)
            for child in list(run):
                if child.tag != qn("w:rPr"):
                    run.remove(child)
            from docx.oxml import OxmlElement
            t = OxmlElement("w:t")
            t.text = text
            run.append(t)
            normal.append(run)
            headings.append(normal)
    index = body.index(marker)
    for element in list(body)[index:]:
        if element.tag != qn("w:sectPr"):
            body.remove(element)
    sect = body.find(qn("w:sectPr"))
    for element in [title_xml, subtitle_xml, *headings]:
        sect.addprevious(element)
    replace_text(find("Cloud Computing"), "Machine Learning")
    for table in template.tables:
        for row in table.rows:
            if row.cells[0].text.strip() == "수강과목":
                replace_text(row.cells[1].paragraphs[0], "머신러닝")
            elif row.cells[0].text.strip() == "제출일자":
                replace_text(row.cells[1].paragraphs[0], "____________________________")
    for section in template.sections:
        for paragraph in section.footer.paragraphs:
            if "Cloud Computing" in paragraph.text:
                for run in paragraph.runs:
                    run.text = run.text.replace("Cloud Computing", "Machine Learning")
    template.save(OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
