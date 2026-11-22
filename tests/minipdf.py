"""Generator minimalnego pliku PDF bez dodatkowych zależności."""

_ASCII_ONLY = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789 .,-:;!?/()'"
)


def build_pdf(lines, title="Dokument testowy"):
    """
    Zbudować zawartość poprawnego pliku PDF z jednym wierszem tekstu na stronę.
    :param lines: Lista wierszy tekstu, po jednym na stronę
    :param title: Tytuł dokumentu zapisany w metadanych
    :return: Bajty pliku PDF
    """
    pages_count = len(lines)
    first_page_obj = 3
    font_obj = first_page_obj + 2 * pages_count
    info_obj = font_obj + 1

    def page_obj_number(index):
        return first_page_obj + 2 * index

    def content_obj_number(index):
        return first_page_obj + 2 * index + 1

    kids = " ".join(f"{page_obj_number(index)} 0 R" for index in range(pages_count))
    bodies = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {pages_count} >>".encode(),
        font_obj: (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
            b" /Encoding /WinAnsiEncoding >>"
        ),
    }

    bodies[info_obj] = (
        f"<< /Title ({_escape(_ascii_only(title))}) "
        f"/Producer (radlab-data tests) >>"
    ).encode("latin-1")

    for index, line in enumerate(lines):
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({_escape(_ascii_only(line))}) Tj ET"
        ).encode("latin-1")
        bodies[content_obj_number(index)] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
        bodies[page_obj_number(index)] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_obj_number(index)} 0 R >>"
        ).encode()

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in range(1, info_obj + 1):
        offsets[number] = len(pdf)
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(bodies[number])
        pdf.extend(b"\nendobj\n")

    start_xref = len(pdf)
    size = info_obj + 1
    pdf.extend(f"xref\n0 {size}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for number in range(1, size):
        pdf.extend(f"{offsets[number]:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {size} /Root 1 0 R /Info {info_obj} 0 R >>\n"
        f"startxref\n{start_xref}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def _ascii_only(text: str) -> str:
    return "".join(char for char in text if char in _ASCII_ONLY)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
