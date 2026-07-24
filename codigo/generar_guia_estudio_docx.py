from __future__ import annotations

import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "documento quarto" / "Guia_estudio_exposicion.qmd"
OUTPUT = ROOT / "documento quarto" / "Guia_estudio_exposicion.docx"


ACRONYM_EXPANSIONS = [
    (r"\bAUC-ROC\b", "área bajo la curva ROC, Receiver Operating Characteristic"),
    (r"(?<!AUC-)\bROC\b", "Receiver Operating Characteristic, curva característica operativa del receptor"),
    (r"\bF1-score\b", "media armónica entre precisión y sensibilidad"),
    (r"\bF1\b", "media armónica entre precisión y sensibilidad"),
    (r"\bMLP\b", "Multilayer Perceptron, perceptrón multicapa"),
    (r"\bSMOTE\b", "Synthetic Minority Over-sampling Technique, sobremuestreo sintético de minorías"),
    (r"\bNDVI\b", "Normalized Difference Vegetation Index, índice de vegetación de diferencia normalizada"),
    (r"\bEVI\b", "Enhanced Vegetation Index, índice de vegetación mejorado"),
    (r"\bNDRE\b", "Normalized Difference Red Edge, índice de diferencia normalizada de borde rojo"),
    (r"\bGLI\b", "Green Leaf Index, índice de hoja verde"),
    (r"\bMSE\b", "Mean Squared Error, error cuadrático medio"),
    (r"\bTN\b", "true negative, verdadero negativo"),
    (r"\bFP\b", "false positive, falso positivo"),
    (r"\bFN\b", "false negative, falso negativo"),
    (r"\bTP\b", "true positive, verdadero positivo"),
    (r"\bAIC\b", "Akaike Information Criterion, criterio de información de Akaike"),
    (r"\bBIC\b", "Bayesian Information Criterion, criterio de información bayesiano"),
    (r"\bLasso\b", "Least Absolute Shrinkage and Selection Operator, selección con penalización L1"),
    (r"\bL1\b", "penalización por valor absoluto"),
    (r"\bk-fold\b", "validación cruzada en k particiones"),
]

CONCEPT_TERMS = [
    "AIC",
    "AUC",
    "AUC-ROC",
    "BIC",
    "Dropout",
    "Early Stopping",
    "EVI",
    "F1",
    "F1-score",
    "FN",
    "FP",
    "GLI",
    "L1",
    "Lasso",
    "MLP",
    "MSE",
    "NDRE",
    "NDVI",
    "ROC",
    "Random Forest",
    "SMOTE",
    "TN",
    "TP",
    "accuracy",
    "altura media",
    "autocorrelación espacial",
    "capa oculta",
    "clasificación binaria",
    "coeficiente",
    "colapsar categorías",
    "colinealidad",
    "conjunto de datos desbalanceado",
    "conjunto de entrenamiento",
    "conjunto de prueba",
    "coordenadas artificiales",
    "correlación de Spearman",
    "curvas de aprendizaje",
    "entropía cruzada categórica",
    "error cuadrático medio",
    "escala ordinal",
    "especificidad",
    "estandarización",
    "exactitud",
    "falso negativo",
    "falso positivo",
    "fuga de información",
    "grilla",
    "índice de Moran",
    "índices espectrales",
    "importancia por permutación",
    "interacciones multiplicativas",
    "intervalo de confianza",
    "matriz de confusión",
    "matriz de vecindad",
    "número de neuronas",
    "odds ratio",
    "precisión",
    "promedio macro",
    "promedio ponderado",
    "regresión logística",
    "sensibilidad",
    "severidad",
    "softmax",
    "sobreajuste",
    "subajuste",
    "tasa de aprendizaje",
    "umbral",
    "validación",
    "validación cruzada",
    "validación cruzada estratificada",
    "valor p",
    "variable ordinal",
    "Youden",
]

ACRONYM_REGEXES = [(re.compile(pattern, flags=re.IGNORECASE), expansion) for pattern, expansion in ACRONYM_EXPANSIONS]
CONCEPT_REGEX = re.compile(
    "|".join(re.escape(term) for term in sorted(CONCEPT_TERMS, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("\\", "")
    return text.strip()


def acronym_expansion_at(text: str, start: int) -> tuple[int, str] | None:
    for regex, expansion in ACRONYM_REGEXES:
        match = regex.match(text, start)
        if match:
            return match.end(), expansion
    return None


def concept_end_at(text: str, start: int) -> int | None:
    match = CONCEPT_REGEX.match(text, start)
    if match:
        return match.end()
    return None


def text_run(
    text: str,
    bold: bool = False,
    italic: bool = False,
    size: int = 24,
    highlight: bool = False,
) -> str:
    props = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
    if highlight:
        props.insert(0, '<w:highlight w:val="yellow"/>')
    if bold:
        props.insert(0, "<w:b/>")
        props.insert(1, "<w:bCs/>")
    if italic:
        props.insert(0, "<w:i/>")
        props.insert(1, "<w:iCs/>")
    safe = escape(text)
    return f"<w:r><w:rPr>{''.join(props)}</w:rPr><w:t xml:space=\"preserve\">{safe}</w:t></w:r>"


def text_runs(text: str, highlight_terms: bool = False, size: int = 24, bold: bool = False) -> str:
    clean_text = clean_inline(text)
    runs: list[str] = []
    index = 0
    while index < len(clean_text):
        acronym = acronym_expansion_at(clean_text, index)
        if acronym:
            end, expansion = acronym
            term = clean_text[index:end]
            runs.append(text_run(term, bold=bold, size=size, highlight=highlight_terms))
            runs.append(text_run(f" ({expansion})", bold=bold, size=size, highlight=False))
            index = end
            continue

        concept_end = concept_end_at(clean_text, index)
        if concept_end:
            runs.append(
                text_run(
                    clean_text[index:concept_end],
                    bold=bold,
                    size=size,
                    highlight=highlight_terms,
                )
            )
            index = concept_end
            continue

        next_index = index + 1
        while next_index < len(clean_text):
            if acronym_expansion_at(clean_text, next_index) or concept_end_at(clean_text, next_index):
                break
            next_index += 1
        runs.append(text_run(clean_text[index:next_index], bold=bold, size=size))
        index = next_index
    return "".join(runs)


def paragraph(
    text: str,
    style: str | None = None,
    bullet: bool = False,
    highlight: bool = False,
) -> str:
    text = clean_inline(text)
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if bullet:
        ppr.append('<w:ind w:left="720" w:hanging="360"/>')
    prefix = "• " if bullet else ""
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{text_runs(prefix + text, highlight_terms=highlight)}</w:p>"


def heading(text: str, level: int) -> str:
    style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
    size = {1: 32, 2: 28, 3: 25}.get(level, 24)
    clean_text = clean_inline(text)
    full_highlight = clean_text.lower() in {
        "cómo decirlo en la exposición",
        "conceptos asociados",
    }
    if full_highlight:
        runs = text_run(clean_text, bold=True, size=size, highlight=True)
    else:
        runs = text_runs(clean_text, bold=True, size=size)
    return (
        f"<w:p><w:pPr><w:pStyle w:val=\"{style}\"/>"
        '<w:spacing w:before="240" w:after="120"/></w:pPr>'
        f"{runs}</w:p>"
    )


def title_page() -> str:
    return "".join(
        [
            '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="160"/></w:pPr>'
            + text_run("Guía de estudio para exposición", bold=True, size=40)
            + "</w:p>",
            '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>'
            + text_run("Taller: Perceptrón Multicapa y detección de Verticillium sp.", italic=True, size=28)
            + "</w:p>",
            '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="500"/></w:pPr>'
            + text_run("Material de apoyo para exposición - 23 de julio de 2026", size=22)
            + "</w:p>",
            '<w:p><w:pPr><w:jc w:val="both"/><w:spacing w:after="300"/></w:pPr>'
            + text_run(
                "Este documento resume qué se hizo, por qué se hizo y cómo explicar cada una de las 19 preguntas del taller. Está escrito como guía de estudio para una exposición en línea.",
                size=24,
            )
            + "</w:p>",
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        ]
    )


def parse_qmd_body(text: str) -> list[str]:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                lines = lines[idx + 1 :]
                break

    blocks: list[str] = [title_page()]
    buffer: list[str] = []
    highlight_context = False

    def flush() -> None:
        nonlocal buffer
        if buffer:
            blocks.append(paragraph(" ".join(buffer), highlight=highlight_context))
            buffer = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            clean_text = clean_inline(text).lower()
            blocks.append(heading(text, min(level, 3)))
            if level <= 3:
                highlight_context = clean_text == "conceptos asociados"
            continue
        if stripped.startswith("- "):
            flush()
            blocks.append(paragraph(stripped[2:], bullet=True, highlight=highlight_context))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush()
            blocks.append(paragraph(stripped, highlight=highlight_context))
            continue
        buffer.append(stripped)
    flush()
    return blocks


def document_xml(blocks: list[str]) -> str:
    sect = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
        + "".join(blocks)
        + sect
        + "</w:body></w:document>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos" w:cs="Aptos"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="360" w:after="160"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="153F66"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="280" w:after="140"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F6F5B"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="220" w:after="120"/><w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="263238"/><w:sz w:val="25"/><w:szCs w:val="25"/></w:rPr>
  </w:style>
</w:styles>"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def core_xml() -> str:
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Guía de estudio para exposición</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OpenXML Generator</Application>
</Properties>"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    blocks = parse_qmd_body(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml())
        docx.writestr("_rels/.rels", rels_xml())
        docx.writestr("word/document.xml", document_xml(blocks))
        docx.writestr("word/_rels/document.xml.rels", document_rels_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/core.xml", core_xml())
        docx.writestr("docProps/app.xml", app_xml())
    print(OUTPUT)


if __name__ == "__main__":
    main()
