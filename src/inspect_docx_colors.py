from pathlib import Path
import re
import zipfile

path = Path(__file__).resolve().parents[1] / "paper_rewriting_output" / "final_paper" / "paper.docx"
with zipfile.ZipFile(path) as z:
    for name in ("word/styles.xml", "word/document.xml", "word/theme/theme1.xml"):
        text = z.read(name).decode("utf-8", errors="ignore")
        print(name, sorted(set(re.findall(r"(?:val|last)=\"([^\"]+)\"", text))))
        print("known-blue", [(x, text.lower().count(x.lower())) for x in ("0f4761", "0f243e", "4f81bd")])
        if name == "word/styles.xml":
            for style_id in ("Heading1", "Heading2", "Title"):
                marker = f'w:styleId="{style_id}"'
                pos = text.find(marker)
                print(style_id, text[pos:pos+1200] if pos >= 0 else "missing")
