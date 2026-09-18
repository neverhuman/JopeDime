#!/usr/bin/env python3
"""Render actual Ratatui TestBackend cells to portable SVG and PNG artifacts."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
bold_path=font_path.with_name('DejaVuSansMono-Bold.ttf')
font=ImageFont.truetype(str(font_path),15);bold=ImageFont.truetype(str(bold_path),15)
for source in a.directory.glob('view-*.cells.json'):
    data=json.loads(source.read_text());w,h=data['columns']*10,data['rows']*20
    image=Image.new('RGB',(w,h),'#09111c');draw=ImageDraw.Draw(image)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#09111c"/>']
    for c in data['cells']:
        x,y=c['x']*10,c['y']*20
        draw.rectangle((x,y,x+10,y+20),fill=c['bg'])
        svg.append(f'<rect x="{x}" y="{y}" width="10" height="20" fill="{c["bg"]}"/>')
        if len(c['text'])==1 and 0x2800<=ord(c['text'])<=0x28ff:
            bits=ord(c['text'])-0x2800
            for bit,(dx,dy) in enumerate([(2,2),(2,7),(2,12),(7,2),(7,7),(7,12),(2,17),(7,17)]):
                if bits&(1<<bit):
                    draw.ellipse((x+dx-1,y+dy-1,x+dx+1,y+dy+1),fill=c['fg'])
                    svg.append(f'<circle cx="{x+dx}" cy="{y+dy}" r="1.1" fill="{c["fg"]}"/>')
        else:
            draw.text((x,y),c['text'],font=bold if c['bold'] else font,fill=c['fg'])
            if c['text'].strip():svg.append(f'<text x="{x}" y="{y+15}" fill="{c["fg"]}" font-family="DejaVu Sans Mono,monospace" font-size="15" font-weight="{700 if c["bold"] else 400}">{html.escape(c["text"])}</text>')
    svg.append('</svg>');base=source.name.replace('.cells.json','')
    (a.directory/(base+'.svg')).write_text(''.join(svg));image.save(a.directory/(base+'.png'))
print(json.dumps(dict(rendered_from='actual Ratatui cells',font_sha256=hashlib.sha256(font_path.read_bytes()).hexdigest(),views=len(list(a.directory.glob('view-*.cells.json'))))))
