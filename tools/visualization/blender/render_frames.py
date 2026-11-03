"""Render selected frames from an already saved .blend; no scene rebuild."""
import argparse
from pathlib import Path
import sys
import bpy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--frame",type=int,action="append",required=True)
args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
args.output.mkdir(parents=True,exist_ok=True)
for frame in args.frame:
    bpy.context.scene.frame_set(frame)
    bpy.context.scene.render.filepath=str(args.output/f"review-{frame:06d}.png")
    bpy.ops.render.render(write_still=True)
