"""Deterministic synthetic corpus for the ControlRoom qualification benchmark.

The corpus is generated from fixed templates (no randomness, no clock) so every run
indexes byte-identical inputs; ``examples/controlroom_qualification/MANIFEST.json`` pins
the SHA-256 of every generated file and a test refuses a silent corpus change.

Shape (mirrors the ControlRoom product model, which SlimX-RAG does not know about):

- workspace ``alpha`` holds two *projects*, ``atlas`` and ``borealis``; ControlRoom
  expresses a project as an explicit ``document_ids`` set, so cross-project isolation is
  tested through that set, never through ``workspace_id`` alone;
- workspace ``beta`` holds project ``cascade`` and deliberately repeats identifiers that
  also exist in ``alpha`` (``PX-4471-B``, ``LT-0917``, ``47.5 kN``) as the strongest
  possible cross-workspace lure.

Formats: Markdown (heading sections), plain text (field blocks), Python code, PDF (real
multi-page files with repeated headers/footers, written by a minimal writer below) and
DOCX (real OOXML packages with Title/Heading styles and a table). All content is
synthetic; the company, people, parts and incidents are fictional.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
import uuid
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

DATASET_VERSION = "cq-2026-09-25.1"

WORKSPACES: dict[str, str] = {"alpha": "ws-alpha-7f3a", "beta": "ws-beta-2c9e"}

_NAMESPACE = uuid.UUID("8d3f5f2a-2c4e-4d1a-9b5e-controlroomq".replace("controlroomq", "6a1f2c3d4e5f"))

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def document_id_for(name: str) -> str:
    """Stable UUID-shaped document id (ControlRoom sends UUIDs)."""
    return str(uuid.uuid5(_NAMESPACE, name))


@dataclass(frozen=True, slots=True)
class CorpusDoc:
    name: str
    workspace: str  # key into WORKSPACES
    project: str
    filename: str
    mime_type: str | None
    content: bytes
    ingest: str = "file"  # file -> /api/index/file ; text -> /api/index
    title: str | None = None

    @property
    def workspace_id(self) -> str:
        return WORKSPACES[self.workspace]

    @property
    def document_id(self) -> str:
        return document_id_for(self.name)


@dataclass(frozen=True, slots=True)
class Corpus:
    version: str
    docs: tuple[CorpusDoc, ...]
    updates: dict[str, CorpusDoc] = field(default_factory=dict)  # name -> replacement content
    deleted: tuple[str, ...] = ()  # names deleted in the lifecycle phase

    def by_name(self, name: str) -> CorpusDoc:
        for doc in self.docs:
            if doc.name == name:
                return doc
        raise KeyError(name)

    def project_docs(self, workspace: str, project: str) -> list[CorpusDoc]:
        return [d for d in self.docs if d.workspace == workspace and d.project == project]

    def name_by_document_id(self) -> dict[str, str]:
        return {d.document_id: d.name for d in self.docs}


# --- minimal deterministic PDF writer ------------------------------------------------


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Write a text-only PDF (Helvetica, one content stream per page). ASCII only."""
    objects: list[bytes] = []
    n_pages = len(pages)
    page_numbers = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{n} 0 R" for n in page_numbers)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for lines in pages:
        ops = ["BT", "/F1 10 Tf", "12 TL", "50 800 Td"]
        for line in lines:
            ops.append(f"({_pdf_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_number = len(objects) + 2
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    return bytes(out)


def _wrap(paragraph: str, width: int = 88) -> list[str]:
    return textwrap.wrap(paragraph, width=width) or [""]


def _pdf_pages(header: str, footer: str, pages: list[tuple[str, list[str]]]) -> list[list[str]]:
    """Each page: header line, title line, wrapped paragraphs, footer line."""
    out: list[list[str]] = []
    for title, paragraphs in pages:
        lines = [header, "", title, ""]
        for paragraph in paragraphs:
            lines.extend(_wrap(paragraph))
            lines.append("")
        lines.append(footer)
        out.append(lines)
    return out


# --- minimal deterministic DOCX writer -----------------------------------------------

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

_STYLES_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_W_NS}">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/></w:style>
</w:styles>
"""

_OOXML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
_CONTENT_TYPES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
    '<Default Extension="xml" ContentType="application/xml"/>\n'
    f'<Override PartName="/word/document.xml" ContentType="{_OOXML}.document.main+xml"/>\n'
    f'<Override PartName="/word/styles.xml" ContentType="{_OOXML}.styles+xml"/>\n'
    "</Types>\n"
)

_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_ROOT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    f'<Relationship Id="rId1" Type="{_REL_TYPE}/officeDocument" Target="word/document.xml"/>\n'
    "</Relationships>\n"
)

_DOC_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    f'<Relationship Id="rId1" Type="{_REL_TYPE}/styles" Target="styles.xml"/>\n'
    "</Relationships>\n"
)

_STYLE_IDS = {"title": "Title", "h1": "Heading1", "h2": "Heading2"}


def _docx_paragraph(kind: str, text: str) -> str:
    style = _STYLE_IDS.get(kind)
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{_xml_escape(text)}</w:t></w:r></w:p>'


def _docx_table(rows: list[list[str]]) -> str:
    cells = []
    for row in rows:
        tcs = "".join(
            f'<w:tc><w:p><w:r><w:t xml:space="preserve">{_xml_escape(cell)}</w:t></w:r></w:p></w:tc>' for cell in row
        )
        cells.append(f"<w:tr>{tcs}</w:tr>")
    return "<w:tbl>" + "".join(cells) + "</w:tbl>"


def build_docx(blocks: list[tuple[str, object]]) -> bytes:
    """Write a DOCX from (kind, payload) blocks: kind in title|h1|h2|p|table."""
    body: list[str] = []
    for kind, payload in blocks:
        if kind == "table":
            assert isinstance(payload, list)
            body.append(_docx_table(payload))
        else:
            body.append(_docx_paragraph(kind, str(payload)))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>' + "".join(body) + "<w:sectPr/></w:body></w:document>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, text in (
            ("[Content_Types].xml", _CONTENT_TYPES_XML),
            ("_rels/.rels", _ROOT_RELS_XML),
            ("word/document.xml", document_xml),
            ("word/_rels/document.xml.rels", _DOC_RELS_XML),
            ("word/styles.xml", _STYLES_XML),
        ):
            info = zipfile.ZipInfo(path, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text.encode("utf-8"))
    return buffer.getvalue()


# --- content: workspace alpha / project atlas -----------------------------------------

_ATLAS_PARTS_MD = """# Atlas Gantry Parts Catalog

## Overview
The Atlas gantry robot (model AG-7) is built from the parts listed below. Part numbers follow the
PX (linear motion), GX (gearboxes) and CC (cable management) scheme. Always quote the full part
number including the suffix letter when ordering from Nordlys Components AS.

## PX-4471-B carriage bearing
Part number: PX-4471-B
Description: sealed linear carriage bearing for the 25 mm profile rail
Rated dynamic load: 14.2 kN
Preload class: C1
Replacement interval: 6,000 operating hours
Supplier: Nordlys Components AS

## PX-4471-C carriage bearing (high-temperature variant)
Part number: PX-4471-C
Description: same envelope as PX-4471-B with fluoropolymer seals rated to 140 C
Rated dynamic load: 12.8 kN
Preload class: C1
Replacement interval: 4,500 operating hours
Supplier: Nordlys Components AS

## GX-210 servo gearbox
Part number: GX-210
Ratio: 25:1
Backlash: 3 arcmin
Mounting torque specification: 45 N-m (see the calibration procedure, step 3)
Lubrication: sealed for life

## GX-2100 industrial gearbox (not fitted to Atlas)
Part number: GX-2100
The GX-2100 is used only on the Cascade press line in the Bergen plant. It is listed here for
cross-reference because its number is often confused with the GX-210 servo gearbox.
Ratio: 100:1

## Cable chain CC-88
Part number: CC-88
Description: closed-profile cable chain, 88 mm inner width, 1.5 m travel
Replacement interval: 12,000 operating hours or on visible link wear
"""

_ATLAS_GLOSSARY_MD = """# Atlas Glossary

## OEE
Overall Equipment Effectiveness: availability multiplied by performance multiplied by quality.
The Atlas program target OEE is 85 percent, measured per shift.

## TQM
Total Quality Management: the plant-wide improvement programme that owns the Atlas OEE reviews.

## MTBF
Mean time between failures. The Atlas gantry MTBF target is 1,200 operating hours; the March
and May incidents are both counted against this figure.

## LOTO
Lockout-tagout: the energy-isolation procedure applied before any cell entry. On Atlas the
lockout tag is applied to the main disconnect, as described in the safety manual.
"""

_ATLAS_TIMELINE_MD = """# Atlas Firmware Release Timeline

2026-01-20 firmware 2.2.0 initial production release for the AG-7 controller
2026-02-11 firmware 2.3.1 adds the payload-limit interlock that halts motion above 120 kg
2026-04-08 firmware 2.4.0 improves homing repeatability and adds the laser offset self-check
2026-06-30 firmware 2.5.0 planned: remote diagnostics and encrypted service log export
"""

_ATLAS_FIRMWARE_TXT = """Atlas Firmware 2.3.1 Release Notes

VERSION
2.3.1

RELEASE DATE
2026-02-11

KEY CHANGE
Adds the payload-limit interlock that halts gantry motion when the measured payload exceeds 120 kg.

CHECKSUM
sha256:9f1c47e2b0aa5d3c

UPGRADE NOTE
Controllers running 2.2.0 must be re-homed after the upgrade because the encoder table format changed.
"""

_ATLAS_CONTROLLER_PY = '''"""Atlas gantry controller helpers (AG-7)."""

from __future__ import annotations

MAX_PAYLOAD_KG = 120
LASER_OFFSET_MM = 0.42


class InterlockError(RuntimeError):
    """Raised when the payload-limit interlock halts motion."""


def compute_gantry_offset(laser_mm: float) -> float:
    """Apply the 0.42 mm laser head offset loaded from calib_v3.cfg."""
    return laser_mm - LASER_OFFSET_MM


def check_payload(payload_kg: float) -> None:
    """Halt motion above MAX_PAYLOAD_KG by raising InterlockError."""
    if payload_kg > MAX_PAYLOAD_KG:
        raise InterlockError(f"payload {payload_kg} kg exceeds {MAX_PAYLOAD_KG} kg")
'''

_ATLAS_CONTRACT_MD = """# Supply Agreement HR-2026-0042

## Parties
This supply agreement HR-2026-0042 is between Halvorsen Robotics AS (the buyer) and
Nordlys Components AS (the supplier) for linear motion parts used on the Atlas gantry.

## Term
The agreement starts on 2025-08-01 and expires on 2026-07-31 unless renewed in writing
ninety days before expiry.

## Pricing
The annual contract value is EUR 184,000, invoiced quarterly. Prices for PX-4471-B and
PX-4471-C bearings are fixed for the term.

## Contacts
Marte Solheim is the supplier account manager. Escalations go to the Halvorsen purchasing lead.
"""

_ATLAS_NOTES_W12_MD = """# Atlas weekly notes, week 12

- The gantry keeps stalling whenever the payload goes over the rated limit; the interlock
  trips at 120 kg exactly as designed, but operators are surprised by it.
- Kwame proposed lowering the interlock threshold to 110 kg to leave a safety margin;
  the team agreed to collect data first.
- Priya will schedule the bearing replacement that was overdue since the March incident.
- Reminder: the calibration sign-off must happen within 24 hours of the procedure.
"""

_ATLAS_MAINTENANCE_V1 = """Atlas Maintenance Log

LAST SERVICE
2026-04-01

TECHNICIAN
Jonas Berg

NOTES
Replaced cable chain CC-88 and re-greased the rail.
"""

_ATLAS_MAINTENANCE_V2 = """Atlas Maintenance Log

LAST SERVICE
2026-06-15

TECHNICIAN
Priya Nair

NOTES
Replaced the PX-4471-B carriage bearing and verified the laser offset.
"""

_ATLAS_OBSOLETE_MD = """# ZEPHYR-9 bracket specification (obsolete)

## Purpose
The ZEPHYR-9 bracket was the original camera mount on the Atlas prototype. It was withdrawn
after the AG-7 redesign and must not be fitted to production gantries.

## Dimensions
The ZEPHYR-9 bracket is 140 mm long with four M6 mounting holes.
"""

_ATLAS_FAQ_MD = """# Atlas operator FAQ

## How do I reset the controller after an interlock trip?
Reduce the payload below the limit, press and hold the blue reset button for three seconds,
then re-home the gantry from the pendant. The controller logs the trip in the service log.

## Why does the gantry move slowly after a restart?
After every restart the controller runs a homing pass at reduced speed until the encoder
table is verified. This takes about ninety seconds.

## Who can approve a calibration sign-off?
Only the shift engineer on duty can sign off a calibration, and it must happen within 24 hours.
"""

_BOILERPLATE = """## Confidentiality
This document is the confidential property of Halvorsen Robotics AS. It may not be copied,
distributed or disclosed outside the company without written permission from the document owner.

## Document control
This document is controlled. Printed copies are uncontrolled and may be out of date; the
current revision is held in the document management system.

## Review cycle
This document is reviewed annually by the quality department and whenever the referenced
procedure changes.
"""

_ATLAS_BOILERPLATE_A_MD = (
    "# Atlas cell 4 work instruction cover sheet\n\n"
    + _BOILERPLATE
    + ("\n## Distribution\nDistribution list: cell 4 operators and the cell 4 shift engineer.\n")
)
_ATLAS_BOILERPLATE_B_MD = (
    "# Atlas cell 9 work instruction cover sheet\n\n"
    + _BOILERPLATE
    + ("\n## Distribution\nDistribution list: cell 9 operators and the cell 9 shift engineer.\n")
)

_HANDOVER_SENTENCES = [
    "Handover starts with the handover checklist and the handover log.",
    "The outgoing operator completes the handover form before the handover meeting.",
    "Every handover records open alarms, and the handover log is signed by both operators.",
    "During handover the pendant stays in the handover cradle.",
    "The handover meeting covers the handover notes from the previous shift.",
    "If the handover is late, the handover log must say why.",
]


def _handover_text() -> str:
    paragraphs = []
    for i in range(12):
        paragraphs.append(" ".join(_HANDOVER_SENTENCES[(i + k) % len(_HANDOVER_SENTENCES)] for k in range(4)))
        if i == 7:
            paragraphs.append(
                "Spare fuses for the cell are kept in the spare fuse box, code FB-77, behind the pendant cradle."
            )
    return "Atlas Shift Handover Procedure\n\n" + "\n\n".join(paragraphs) + "\n"


def _atlas_calibration_pdf() -> bytes:
    header = "Halvorsen Robotics - Atlas Calibration Procedure"
    footer = "Rev 3 - Confidential - Halvorsen Robotics AS"
    pages = [
        (
            "1. Scope",
            [
                "This procedure describes the calibration of the Atlas gantry robot (model AG-7) after a "
                "bearing or gearbox replacement and at every 6,000 hour service.",
                "The procedure takes about two hours and requires the cell to be locked out.",
            ],
        ),
        (
            "2. Tools and materials",
            [
                "Torque wrench TW-15 (calibrated within the last twelve months), the laser alignment head, "
                "the service USB stick and a clean lint-free cloth.",
                "The pendant must be on firmware 2.3.1 or later.",
            ],
        ),
        (
            "3. Step 2: Laser offset",
            [
                "Mount the laser alignment head on the carriage. Set the laser head offset to 0.42 mm using "
                "the offset screw and confirm the reading on the pendant.",
                "Record the offset value on the calibration sheet.",
            ],
        ),
        (
            "4. Step 3: Gearbox mounting torque",
            [
                "Tighten the four GX-210 gearbox mounting bolts to 45 N-m using torque wrench TW-15, in a "
                "cross pattern. Do not exceed the specification: over-torque was the root cause of the May incident.",
            ],
        ),
        (
            "5. Step 4: Load the configuration",
            [
                "Insert the service USB stick and load the configuration file calib_v3.cfg from the pendant "
                "service menu. The controller applies the laser offset from this file.",
                "Reject the file if the checksum shown on the pendant does not match the service sheet.",
            ],
        ),
        (
            "6. Sign-off",
            [
                "The calibration must be signed off by the shift engineer within 24 hours. Unsigned "
                "calibrations expire and the gantry returns to reduced-speed mode.",
            ],
        ),
    ]
    return build_pdf(_pdf_pages(header, footer, pages))


_SAFETY_TOPICS: list[tuple[str, str]] = [
    ("Purpose of this manual", "This manual defines the safety rules for the Atlas gantry cell."),
    ("Responsibilities", "The cell owner is responsible for enforcing the rules in this manual."),
    ("Training", "Operators must complete the Atlas safety course before working in the cell."),
    ("Personal protective equipment", "Safety shoes and eye protection are mandatory in the cell."),
    ("Cell layout", "The Atlas cell has five bays numbered 1 to 5 from the loading door."),
    ("Guarding", "Fixed guards must only be removed by maintenance with the cell locked out."),
    ("Interlocked doors", "All access doors are interlocked and stop the gantry when opened."),
    ("Emergency stops", "Emergency stop buttons are located at bays 1, 3 and 5 and on the pendant."),
    ("Pendant use", "The pendant enabling switch must be held in the middle position to jog."),
    ("Speed limits in manual mode", "Manual mode speed is limited to 250 mm per second."),
    ("Payload limits", "Never exceed the rated payload of 120 kg; the interlock enforces it."),
    ("Load handling", "Loads must be secured to the carriage plate with two straps."),
    ("Housekeeping", "Keep the floor of the cell free of cables and packaging."),
    ("Lighting", "Cell lighting must provide at least 500 lux at the work surface."),
    ("Noise", "Noise inside the cell is normally below 80 dB(A) during production."),
    ("Ventilation", "The cell extraction must run whenever the gantry is in production."),
    (
        "Lockout-tagout",
        "Lockout tag LT-0917 must be applied to the main disconnect before entering the cell for "
        "maintenance. Only the person who applied the tag may remove it.",
    ),
    ("Stored energy", "Release pneumatic pressure at the manifold before any maintenance."),
    ("Working at height", "Use the approved platform when working on the gantry beam."),
    ("Electrical safety", "The control cabinet is opened only by authorised electricians."),
    ("Hot surfaces", "The gearbox housing can reach 70 C after continuous operation."),
    ("Laser alignment head", "The laser alignment head is class 2; do not stare into the beam."),
    ("Light curtain", "The light curtain response time is 14 ms and it is tested every shift."),
    ("Safety mats", "Pressure-sensitive mats stop the gantry when stepped on during production."),
    ("Restart after a stop", "After any safety stop the operator must inspect the cell before restart."),
    ("Incident reporting", "Every incident, including near misses, is reported within the shift."),
    ("Hearing protection", "Hearing protection is mandatory above 85 dB(A), for example during air blow-off."),
    ("Chemical handling", "Only the approved rail grease may be used inside the cell."),
    ("Visitors", "Visitors may enter the cell only with an escort and with the gantry stopped."),
    ("Revision history", "Revision 4 added the light curtain test and the hearing protection rule."),
]


def _atlas_safety_pdf() -> bytes:
    header = "Halvorsen Robotics - Atlas Safety Manual"
    footer = "Revision 4 - Controlled document - do not copy"
    pages = []
    for number, (topic, fact) in enumerate(_SAFETY_TOPICS, start=1):
        pages.append(
            (
                f"Section {number}: {topic}",
                [
                    fact,
                    "This section applies to every person entering the Atlas gantry cell, including "
                    "maintenance staff, operators, engineers and escorted visitors.",
                    "Deviations from this section require written approval from the cell owner and "
                    "must be recorded in the safety log before the work starts.",
                ],
            )
        )
    return build_pdf(_pdf_pages(header, footer, pages))


def _atlas_incident_march_docx() -> bytes:
    return build_docx(
        [
            ("title", "Incident Report IR-2026-031"),
            ("h1", "Summary"),
            (
                "p",
                "On 2026-03-14 the Atlas gantry stalled during a 47.5 kN press cycle in cell 4. No one was "
                "injured. The line was stopped for six hours.",
            ),
            ("h1", "Timeline"),
            ("p", "08:12 the gantry stalled mid-stroke and the controller raised a following-error alarm."),
            ("p", "08:30 the shift engineer applied lockout tag LT-0917 and entered the cell."),
            ("p", "14:05 the line restarted after the bearing was replaced and the calibration procedure was run."),
            ("h1", "Root cause"),
            (
                "p",
                "The PX-4471-B carriage bearing had exceeded its 6,000 operating hour replacement interval by "
                "roughly 900 hours and had seized on the rail.",
            ),
            ("h1", "Corrective actions"),
            (
                "p",
                "Dr. Ines Halvorsen approved the corrective actions: replace the bearing, add the replacement "
                "interval to the maintenance planner and re-run the calibration procedure.",
            ),
            ("h1", "Attendees"),
            ("table", [["Name", "Role"], ["Ines Halvorsen", "Engineering director"], ["Jonas Berg", "Technician"]]),
        ]
    )


def _atlas_incident_may_docx() -> bytes:
    return build_docx(
        [
            ("title", "Incident Report IR-2026-052"),
            ("h1", "Summary"),
            (
                "p",
                "On 2026-05-02 the Atlas gantry tripped an overcurrent fault at 38.0 kN during the second "
                "shift. No one was injured. The line was stopped for ninety minutes.",
            ),
            ("h1", "Root cause"),
            (
                "p",
                "The GX-210 gearbox mounting torque was measured at 48 N-m, above the 45 N-m specification, "
                "which distorted the housing and increased the motor current.",
            ),
            ("h1", "Corrective actions"),
            (
                "p",
                "Torque the gearbox bolts to specification with the calibrated wrench TW-15 and add a torque "
                "check to the calibration sign-off sheet. Approved by Kwame Mensah, production manager.",
            ),
        ]
    )


# --- content: workspace alpha / project borealis ------------------------------------

_BOREALIS_PARTS_MD = """# Borealis Conveyor Parts

## BX-3310 drive roller
Part number: BX-3310
Description: crowned drive roller, 120 mm diameter
Rated load: 9.5 kN

## PX-4471-B compatibility note
PX-4471-B bearings from the Atlas program are NOT approved for Borealis carriages because the
Borealis rail is 20 mm. Use BX-4471 instead; the rated dynamic load of BX-4471 is 11.0 kN.

## Emergency stop relay ER-12
Part number: ER-12
The emergency stop relay for the Borealis conveyor is ER-12; the emergency stop buttons are at the
head and tail pulleys.
"""

_BOREALIS_GLOSSARY_MD = """# Borealis Glossary

## OEE
Overall Equipment Effectiveness for the conveyor line. The Borealis target OEE is 78 percent.

## MTBF
Mean time between failures. The Borealis MTBF target is 2,000 operating hours.
"""

_BOREALIS_TIMELINE_MD = """# Borealis Release Timeline

2026-02-02 conveyor control software 1.0 released
2026-04-19 belt tracking incident BR-2026-019
2026-05-20 control software 1.1 adds belt-drift alarm
"""

_BOREALIS_NOTES_TXT = """Borealis line notes

BELT
The belt is a 1,200 mm wide polyurethane belt supplied by Skagen Belting.

MOTOR
The head drive motor is rated 7.5 kW and runs at 1,450 rpm.

CONTACT
Line lead is Ines Halvorsen on Tuesdays and Thursdays.
"""


def _borealis_spec_pdf() -> bytes:
    header = "Halvorsen Robotics - Borealis Conveyor Specification"
    footer = "Rev 1 - Confidential"
    pages = [
        ("1. General", ["The Borealis conveyor moves finished parts from the press line to packing."]),
        ("2. Belt", ["Belt speed is 1.8 m/s at full production and 0.6 m/s in setup mode."]),
        ("3. Drive", ["The head drive motor rating is 7.5 kW with a BX-3310 drive roller."]),
        ("4. Safety", ["Emergency stops are at the head and tail pulleys and act through relay ER-12."]),
    ]
    return build_pdf(_pdf_pages(header, footer, pages))


def _borealis_incident_docx() -> bytes:
    return build_docx(
        [
            ("title", "Incident Report BR-2026-019"),
            ("h1", "Summary"),
            ("p", "On 2026-04-19 the Borealis belt drifted into the side guide and tore over 1.5 m of edge."),
            ("h1", "Root cause"),
            ("p", "The tail pulley was misaligned by 4 mm after a roller change."),
            ("h1", "Corrective actions"),
            ("p", "Ines Halvorsen approved adding a belt-drift alarm to the control software 1.1 release."),
        ]
    )


# --- content: workspace beta / project cascade ---------------------------------------

_CASCADE_PARTS_MD = """# Cascade Press Line Parts

## PX-4471-B (shared spare)
Part number: PX-4471-B
Cascade keeps 12 units of PX-4471-B in the shared spares cage for the Bergen plant. The rated
dynamic load of PX-4471-B is 14.2 kN.

## GX-2100 industrial gearbox
Part number: GX-2100
Ratio: 100:1
The GX-2100 drives the Cascade press feed.
"""

_CASCADE_GLOSSARY_MD = """# Cascade Glossary

## OEE
Overall Equipment Effectiveness. The Cascade press line target OEE is 91 percent.
"""

_CASCADE_NOTES_TXT = """Cascade press line notes

PRESS FORCE
The Cascade press applies 950 kN at bottom dead centre.

LOCKOUT
Lockout tag LT-0917 is also used on the Cascade main disconnect.
"""


def _cascade_spec_pdf() -> bytes:
    header = "Halvorsen Robotics - Cascade Press Specification"
    footer = "Rev 2 - Confidential"
    pages = [
        ("1. General", ["The Cascade press line forms steel brackets for the Bergen plant."]),
        ("2. Press force", ["Press force is 950 kN at bottom dead centre with a 47.5 kN pre-load."]),
        ("3. Lockout", ["Lockout tag LT-0917 must be applied to the Cascade main disconnect before entry."]),
    ]
    return build_pdf(_pdf_pages(header, footer, pages))


def _cascade_incident_docx() -> bytes:
    return build_docx(
        [
            ("title", "Incident Report CP-2026-007"),
            ("h1", "Summary"),
            ("p", "On 2026-03-14 the Cascade press stalled at 47.5 kN pre-load because of a jammed feed."),
            ("h1", "Corrective actions"),
            ("p", "Approved by Ines Halvorsen: clear the feed and add a jam sensor."),
        ]
    )


# --- assembly -------------------------------------------------------------------------


def _text(value: str) -> bytes:
    return value.encode("utf-8")


_MD = "text/markdown"
_TXT = "text/plain"


def _md(name: str, workspace: str, project: str, filename: str, body: str) -> CorpusDoc:
    return CorpusDoc(name, workspace, project, filename, _MD, _text(body))


def _txt(name: str, workspace: str, project: str, filename: str, body: str) -> CorpusDoc:
    return CorpusDoc(name, workspace, project, filename, _TXT, _text(body))


def build_corpus() -> Corpus:
    docs: list[CorpusDoc] = [
        # alpha / atlas
        _md("atlas-parts-catalog", "alpha", "atlas", "atlas-parts-catalog.md", _ATLAS_PARTS_MD),
        CorpusDoc(
            "atlas-calibration-procedure",
            "alpha",
            "atlas",
            "atlas-calibration-procedure.pdf",
            "application/pdf",
            _atlas_calibration_pdf(),
            title="Atlas Calibration Procedure",
        ),
        CorpusDoc(
            "atlas-incident-2026-03-14",
            "alpha",
            "atlas",
            "atlas-incident-2026-03-14.docx",
            _DOCX_MIME,
            _atlas_incident_march_docx(),
        ),
        CorpusDoc(
            "atlas-incident-2026-05-02",
            "alpha",
            "atlas",
            "atlas-incident-2026-05-02.docx",
            _DOCX_MIME,
            _atlas_incident_may_docx(),
        ),
        _md("atlas-glossary", "alpha", "atlas", "atlas-glossary.md", _ATLAS_GLOSSARY_MD),
        _md("atlas-release-timeline", "alpha", "atlas", "atlas-release-timeline.md", _ATLAS_TIMELINE_MD),
        _txt("atlas-firmware-notes", "alpha", "atlas", "atlas-firmware-notes.txt", _ATLAS_FIRMWARE_TXT),
        CorpusDoc(
            "atlas-controller", "alpha", "atlas", "atlas_controller.py", "text/x-python", _text(_ATLAS_CONTROLLER_PY)
        ),
        CorpusDoc(
            "atlas-safety-manual",
            "alpha",
            "atlas",
            "atlas-safety-manual.pdf",
            "application/pdf",
            _atlas_safety_pdf(),
            title="Atlas Safety Manual",
        ),
        _md("atlas-vendor-contract", "alpha", "atlas", "atlas-vendor-contract.md", _ATLAS_CONTRACT_MD),
        _md("atlas-meeting-notes-w12", "alpha", "atlas", "atlas-meeting-notes-w12.md", _ATLAS_NOTES_W12_MD),
        CorpusDoc(
            "atlas-maintenance-log",
            "alpha",
            "atlas",
            "atlas-maintenance-log.txt",
            "text/plain",
            _text(_ATLAS_MAINTENANCE_V1),
            ingest="text",
            title="Atlas Maintenance Log",
        ),
        _md("atlas-obsolete-spec", "alpha", "atlas", "atlas-obsolete-spec.md", _ATLAS_OBSOLETE_MD),
        _md("atlas-faq", "alpha", "atlas", "atlas-faq.md", _ATLAS_FAQ_MD),
        _md("atlas-boilerplate-a", "alpha", "atlas", "atlas-cell4-cover.md", _ATLAS_BOILERPLATE_A_MD),
        _md("atlas-boilerplate-b", "alpha", "atlas", "atlas-cell9-cover.md", _ATLAS_BOILERPLATE_B_MD),
        _txt("atlas-shift-handover", "alpha", "atlas", "atlas-shift-handover.txt", _handover_text()),
        # alpha / borealis
        _md("borealis-parts-catalog", "alpha", "borealis", "borealis-parts.md", _BOREALIS_PARTS_MD),
        CorpusDoc(
            "borealis-conveyor-spec",
            "alpha",
            "borealis",
            "borealis-conveyor-spec.pdf",
            "application/pdf",
            _borealis_spec_pdf(),
            title="Borealis Conveyor Specification",
        ),
        CorpusDoc(
            "borealis-incident-2026-04-19",
            "alpha",
            "borealis",
            "borealis-incident.docx",
            _DOCX_MIME,
            _borealis_incident_docx(),
        ),
        _md("borealis-glossary", "alpha", "borealis", "borealis-glossary.md", _BOREALIS_GLOSSARY_MD),
        _md("borealis-timeline", "alpha", "borealis", "borealis-timeline.md", _BOREALIS_TIMELINE_MD),
        _txt("borealis-notes", "alpha", "borealis", "borealis-notes.txt", _BOREALIS_NOTES_TXT),
        # beta / cascade
        _md("cascade-parts-catalog", "beta", "cascade", "cascade-parts.md", _CASCADE_PARTS_MD),
        CorpusDoc(
            "cascade-press-spec",
            "beta",
            "cascade",
            "cascade-press-spec.pdf",
            "application/pdf",
            _cascade_spec_pdf(),
            title="Cascade Press Specification",
        ),
        CorpusDoc("cascade-incident", "beta", "cascade", "cascade-incident.docx", _DOCX_MIME, _cascade_incident_docx()),
        _md("cascade-glossary", "beta", "cascade", "cascade-glossary.md", _CASCADE_GLOSSARY_MD),
        _txt("cascade-notes", "beta", "cascade", "cascade-notes.txt", _CASCADE_NOTES_TXT),
    ]
    updates = {
        "atlas-maintenance-log": CorpusDoc(
            "atlas-maintenance-log",
            "alpha",
            "atlas",
            "atlas-maintenance-log.txt",
            "text/plain",
            _text(_ATLAS_MAINTENANCE_V2),
            ingest="text",
            title="Atlas Maintenance Log",
        )
    }
    return Corpus(version=DATASET_VERSION, docs=tuple(docs), updates=updates, deleted=("atlas-obsolete-spec",))


def write_corpus(corpus: Corpus, directory: Path) -> dict[str, object]:
    """Write every document (and update) under ``directory``; return the file manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for doc in corpus.docs:
        relpath = f"{doc.workspace}/{doc.project}/{doc.filename}"
        target = directory / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(doc.content)
        files[relpath] = hashlib.sha256(doc.content).hexdigest()
    for name, doc in corpus.updates.items():
        relpath = f"{doc.workspace}/{doc.project}/updates/{doc.filename}"
        target = directory / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(doc.content)
        files[relpath] = hashlib.sha256(doc.content).hexdigest()
        files[f"#update:{name}"] = files[relpath]
    return {
        "dataset_version": corpus.version,
        "document_count": len(corpus.docs),
        "deleted": list(corpus.deleted),
        "files": dict(sorted(files.items())),
    }


def corpus_manifest(corpus: Corpus) -> dict[str, object]:
    """The manifest without writing files (same content as :func:`write_corpus`)."""
    files: dict[str, str] = {}
    for doc in corpus.docs:
        files[f"{doc.workspace}/{doc.project}/{doc.filename}"] = hashlib.sha256(doc.content).hexdigest()
    for name, doc in corpus.updates.items():
        digest = hashlib.sha256(doc.content).hexdigest()
        files[f"{doc.workspace}/{doc.project}/updates/{doc.filename}"] = digest
        files[f"#update:{name}"] = digest
    return {
        "dataset_version": corpus.version,
        "document_count": len(corpus.docs),
        "deleted": list(corpus.deleted),
        "files": dict(sorted(files.items())),
    }


def manifest_json(corpus: Corpus) -> str:
    return json.dumps(corpus_manifest(corpus), indent=2, sort_keys=True) + "\n"
