#!/usr/bin/env python3
"""Render recorded terminal frames, preserving their live timestamps and measured states.
One-time dependency: pyte==0.8.2 (no effect on the Rust runtime).
"""
import argparse,hashlib,json,re
from pathlib import Path
import pyte
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
manifest=json.loads((a.capture/'frames.json').read_text());raw=(a.capture/manifest['ansi']).read_bytes()
assert hashlib.sha256(raw).hexdigest()==manifest['ansi_sha256']
a.output.mkdir(parents=True,exist_ok=True)
screen=pyte.Screen(160,55);stream=pyte.Stream(screen);previous=0
palette={'black':'000000','red':'ff7487','green':'63e0ae','brown':'fac565','blue':'5ccdff','magenta':'b59fff','cyan':'5ccdff','white':'dae5f2','brightblack':'7e95ae','brightwhite':'ffffff'}
def color(value,background):
    if value=='default':return '#09111c' if background else '#dae5f2'
    return '#'+palette.get(value,value if re.fullmatch('[0-9a-fA-F]{6}',value) else ('09111c' if background else 'dae5f2'))
receipts=[]
for frame in manifest['frames']:
    stream.feed(raw[previous:frame['offset']].decode('utf-8',errors='replace'));previous=frame['offset']
    cells=[]
    for y in range(frame['rows']):
        for x in range(frame['columns']):
            c=screen.buffer[y][x];fg=color(c.fg,False);bg=color(c.bg,True)
            if c.reverse:fg,bg=bg,fg
            cells.append(dict(x=x,y=y,text=c.data,fg=fg,bg=bg,bold=c.bold))
    path=a.output/f"view-{frame['view']}.cells.json"
    payload=dict(schema='parallel.ratatui_snapshot.v1',columns=160,rows=55,view=frame['view'],cells=cells,source='actual live terminal byte stream',timestamp=frame['timestamp'],ansi_sha256=manifest['ansi_sha256'],ansi_offset=frame['offset'],decoder='pyte==0.8.2')
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    (a.output/f"view-{frame['view']}.txt").write_text('\n'.join(screen.display)+'\n')
    receipts.append(dict(view=frame['view'],cells_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),timestamp=frame['timestamp']))
(a.output/'terminal-render.json').write_text(json.dumps(dict(schema='parallel.tui_terminal_render.v1',passed=True,source_ansi_sha256=manifest['ansi_sha256'],frames=receipts),indent=2)+'\n')
print(json.dumps(dict(frames=len(receipts),output=str(a.output),source='recorded live terminal')))
