#!/usr/bin/env python3
"""Verify publication bytes and render all pages for separate visual inspection."""
import hashlib,json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(sys.argv[1]).resolve()
m=json.loads((root/'CORE_IP_EVIDENCE_MANIFEST.json').read_text())
assert 50<=m['monograph_pages']<=80, 'monograph page budget'
for entry in m['outputs']:
 p=Path(entry['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],p
pdf=root/'CORE_IP_TECHNICAL_REFERENCE.pdf';out=root/'core-ip-verification';out.mkdir(exist_ok=True)
info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);(out/'pdfinfo.txt').write_text(info)
outline=subprocess.check_output(['mutool','show',str(pdf),'outline'],text=True);(out/'outline.txt').write_text(outline)
assert 'Appendix A' in outline and 'GPU pipeline measurements' in outline, 'bookmarks absent'
text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);(out/'pages.txt').write_text(text)
pages=text.split('\f');pages=pages[:-1] if not pages[-1].strip() else pages
assert len(pages)==m['pages'];assert all(p.strip() for p in pages),'blank page'
subprocess.run(['pdftoppm','-scale-to','700','-png','-r','65',str(pdf),str(out/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
images=sorted(out.glob('page-*.png'))
assert len(images)==m['pages']
for start in range(0,len(images),9):
 selected=images[start:start+9];sheet=Image.new('RGB',(1530,2220),'#aabbc0');draw=ImageDraw.Draw(sheet)
 for j,path in enumerate(selected):
  im=Image.open(path);im.thumbnail((490,690));x=(j%3)*510+10;y=(j//3)*740+30;sheet.paste(im,(x,y));draw.text((x,y-20),f'Page {start+j+1}',fill='black')
 sheet.save(out/f'contact-{start+1:03d}-{start+len(selected):03d}.jpg',quality=90)
receipt={'schema':'jope.core-ip-render-verification/v1','pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(pages),'monograph_pages':m['monograph_pages'],'bookmarks_verified':True,'nonempty_pages':len(pages),'output_hashes_verified':True,'visual_inspection':'pending; inspect every contact sheet and retain separate receipt'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
