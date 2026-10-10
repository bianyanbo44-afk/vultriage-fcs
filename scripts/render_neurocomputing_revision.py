"""Render the manuscript, supplement, and cover letter for local visual QA."""
from pathlib import Path
import argparse
import re
import pymupdf
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[1]
paper=root/'paper/neurocomputing_revision'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--documents', nargs='+', choices=['main','supplementary','cover_letter'], default=['main','supplementary','cover_letter'])
parser.add_argument('--output', type=Path, default=root/'tmp/neurocomputing-qa')
args = parser.parse_args()
out=args.output
out.mkdir(parents=True,exist_ok=True)
for name in args.documents:
    doc=pymupdf.open(paper/f'{name}.pdf')
    target=out/name
    target.mkdir(exist_ok=True)
    paths=[]
    for i,page in enumerate(doc):
        path=target/f'page-{i+1:02d}.png'
        page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4),alpha=False).save(path)
        paths.append(path)
    for start in range(0,len(paths),6):
        canvas=Image.new('RGB',(1680,1620),'#E9EDF0')
        draw=ImageDraw.Draw(canvas)
        for j,path in enumerate(paths[start:start+6]):
            pic=Image.open(path).convert('RGB');pic.thumbnail((535,750))
            x=(j%3)*560+12;y=(j//3)*810+32
            canvas.paste(pic,(x,y));draw.text((x,y-22),path.stem,fill='black')
        canvas.save(target/f'contact-{start//6+1}.jpg', quality=88, optimize=True)
    print(name,'pages',len(doc))
    if name == 'main':
        tex = (paper/'main.tex').read_text(encoding='utf-8')
        body = tex.split('\\end{frontmatter}', 1)[1].split('\\bibliographystyle', 1)[0]
        body = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', '', body, flags=re.S)
        body = re.sub(r'\\begin\{(equation|align)\}.*?\\end\{\1\}', '$ [display equation] $', body, flags=re.S)
        body = re.sub(r'\\section\*?\{([^}]+)\}', r'# \1', body)
        body = re.sub(r'\\subsection\*?\{([^}]+)\}', r'## \1', body)
        body = re.sub(r'\\paragraph\{([^}]+)\}', r'### \1\n', body)
        body = re.sub(r'\\(?:cite\w*|label|ref|eqref)\{[^}]*\}', '', body)
        body = re.sub(r'\\(?:url|texttt)\{([^}]+)\}', r'`\1`', body)
        body = re.sub(r'\\(?:begin|end)\{enumerate\}', '', body)
        body = body.replace('\\method', 'VulTriage').replace('\\item', '- ').replace('\\%', '%')
        body = body.replace('--', '-').replace('~', ' ')
        (out/'draft.md').write_text(body, encoding='utf-8')
        abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)
        abstract = abstract.replace('\\%', '%').replace('$', '')
        (out/'abstract.md').write_text('# English Abstract\n\n'+abstract.strip()+'\n', encoding='utf-8')
