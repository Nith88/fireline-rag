"""Build a minimal text PDF in memory, so tests need no PDF-writing dependency."""


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """One page per string. Each string is wrapped into ~70-character lines."""
    objs: list[bytes] = []

    def add(body: str) -> int:
        objs.append(body.encode("latin-1"))
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    kids_index = add("")  # object 2: Pages, filled in below
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for text in pages:
        lines, line = [], ""
        for w in text.split():
            if len(line) + len(w) > 70:
                lines.append(line)
                line = ""
            line = f"{line} {w}".strip()
        lines.append(line)
        stream = "BT /F1 10 Tf 12 TL 40 800 Td " + " T* ".join(f"({_esc(ln)}) Tj" for ln in lines) + " ET"
        content = add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        kids.append(add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {content} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        ))
    objs[kids_index - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n".encode() + b"%%EOF\n"
    return out
