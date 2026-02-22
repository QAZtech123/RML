from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
SRC_MD = ROOT / "rml_recursive_meta_learning_short_paper.md"
OUT_HTML = ROOT / "rml_recursive_meta_learning_short_paper.html"
OUT_PDF = ROOT / "rml_recursive_meta_learning_short_paper.pdf"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RML Short Paper</title>
  <style>
    body {{
      font-family: "Times New Roman", Georgia, serif;
      max-width: 820px;
      margin: 2.5rem auto;
      line-height: 1.5;
      font-size: 12pt;
      color: #111;
      padding: 0 1rem;
      background: #fff;
    }}
    h1, h2, h3 {{
      line-height: 1.25;
      margin-top: 1.4em;
      margin-bottom: 0.5em;
    }}
    h1 {{
      font-size: 22pt;
      margin-top: 0.2em;
    }}
    h2 {{ font-size: 16pt; }}
    h3 {{ font-size: 13pt; }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 10.5pt;
      background: #f5f5f5;
      padding: 0.1rem 0.2rem;
      border-radius: 3px;
    }}
    pre {{
      background: #f7f7f7;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 0.7rem;
      overflow-x: auto;
    }}
    p {{
      margin: 0.5em 0;
      text-align: justify;
    }}
    ul, ol {{
      margin-top: 0.3em;
      margin-bottom: 0.8em;
    }}
    hr {{
      border: 0;
      border-top: 1px solid #ccc;
      margin: 1.2rem 0;
    }}
    @page {{
      size: A4;
      margin: 20mm;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _find_edge() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    in_path = shutil.which("msedge")
    if in_path:
        return Path(in_path)
    return None


def build_html() -> None:
    if not SRC_MD.exists():
        raise FileNotFoundError(f"Missing markdown source: {SRC_MD}")
    md_text = SRC_MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        output_format="html5",
    )
    html = HTML_TEMPLATE.format(body=body)
    OUT_HTML.write_text(html, encoding="utf-8")


def build_pdf() -> bool:
    edge = _find_edge()
    if edge is None:
        print("PDF step skipped: Microsoft Edge not found.")
        return False

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    file_url = OUT_HTML.resolve().as_uri()
    cmd = [
        str(edge),
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={OUT_PDF.resolve()}",
        file_url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print("PDF step failed.")
        if exc.stderr:
            print(exc.stderr.strip())
        return False

    if OUT_PDF.exists():
        print(f"PDF written: {OUT_PDF}")
        return True
    print("PDF step reported success but output file not found.")
    return False


def main() -> None:
    build_html()
    print(f"HTML written: {OUT_HTML}")
    build_pdf()


if __name__ == "__main__":
    main()

