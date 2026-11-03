"""Build review stills from native fused inference. Run with Blender --python.

No model is reimplemented here. Node values and x*w pulses come from validated
native affine taps. Travel time and spatial layout are editorial, not latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_trace import affine_arrays, validate

CYAN = (0.025, 0.72, 1.0, 1)
VIOLET = (0.58, 0.18, 1.0, 1)
WHITE = (0.72, 0.85, 1.0, 1)
MUTED = (0.20, 0.33, 0.47, 1)
AMBER = (1.0, 0.57, 0.16, 1)


def material(name, color, strength=1.0, surface=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    if surface:
        shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Metallic"].default_value = 0.65
        shader.inputs["Roughness"].default_value = 0.28
    else:
        shader = tree.nodes.new("ShaderNodeEmission")
        shader.inputs["Color"].default_value = color
        shader.inputs["Strength"].default_value = strength
    tree.links.new(shader.outputs[0], out.inputs["Surface"])
    return mat


def cube(name, location, scale, mat, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Soft machined edge", "BEVEL")
        mod.width, mod.segments = bevel, 2
    return obj


def text(body, location, size, mat, name=None):
    curve = bpy.data.curves.new(name or body[:48], "FONT")
    curve.body, curve.size = body, size
    curve.extrude = 0
    curve.space_character = 1.1
    obj = bpy.data.objects.new(name or body[:48], curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(mat)
    return obj


def line(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius
    curve.bevel_resolution = 1
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, xyz in zip(spline.points, points):
        p.co = (*xyz, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def sphere(name, position, radius, mat):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def curves(obj):
    animation = obj.animation_data
    if animation is None or animation.action is None:
        return []
    action = animation.action
    result = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(animation.action_slot)
            if bag:
                result.extend(bag.fcurves)
    return result


def interpolation(obj, mode):
    for curve in curves(obj):
        for point in curve.keyframe_points:
            point.interpolation = mode


def choose_layers(case):
    layers = case["trace"]["affine"]
    selected = []
    for origin in ("dope", "jope"):
        candidates = [v for v in layers if origin in v["network"].lower()]
        # Prefer the contiguous backbone and ranker's observed dense layers.
        if origin == "dope":
            priority = [v for v in candidates if ".backbone." in v["name"]]
            candidates = priority + [v for v in candidates if v not in priority]
        else:
            priority = [v for wanted in ("ctx_bilinear_u", "mm_prior_out.0") for v in candidates if v["name"] == wanted]
            candidates = priority + [v for v in candidates if v not in priority]
        seen = set()
        for layer in candidates:
            if layer["name"] in seen:
                continue
            seen.add(layer["name"])
            selected.append(layer)
            if len(seen) == 2:
                break
    if len(selected) != 4:
        raise ValueError("Need two distinct native affine operations per fused constituent")
    return selected


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--edges", type=int, default=80)
    parser.add_argument("--layout", choices=("board", "cinematic", "technical"), default="board")
    parser.add_argument("--engine-proof", type=Path)
    parser.add_argument("--emit-metrics", type=Path)
    parser.add_argument("--show-gp-tree", action="store_true", help="Temporary private debug preview of the real winning GP syntax")
    parser.add_argument("--code-pages", type=Path, help="Complete source page root, containing case-*/manifest.json")
    parser.add_argument("--code-hold-frames", type=int, default=6)
    parser.add_argument("--selection", type=Path, help="Measured embedding-diversity selection receipt")
    parser.add_argument("--render-frame", type=int, action="append", default=[])
    parser.add_argument("--render-animation", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 32 or args.edges < 1 or args.edges > 256:
        parser.error("frames >=32 and edges in 1..256 required")
    data, validation = validate(args.trace)
    if args.frames < 32 * len(data["cases"]):
        parser.error("each captured case needs at least 32 replay frames")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "board":
        from board import build
        build(data, args, validation)
        return
    if args.layout == "cinematic":
        from cinematic import build
        build(data, args, validation)
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, args.frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.world.color = (0.006, 0.009, 0.02)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.render.film_transparent = False

    ink = material("Type / ice", WHITE, 1)
    dim = material("Type / secondary", MUTED, 1)
    teal = material("DOPE / origin", CYAN, 1.2)
    purple = material("JOPE / origin", VIOLET, 1.2)
    gold = material("Proof / amber", AMBER, 1)
    panel = material("Black ceramic", (0.012, 0.023, 0.043, 1), surface=True)
    structure = material("Unlit weights / structure", (0.026, 0.052, 0.080, 1), .7)
    floor = material("Stage / navy", (0.005, 0.011, 0.025, 1), surface=True)
    cube("Stage", (0, 0, -0.30), (35, 22, .2), floor)

    # Orthographic overhead composition, with actual 3D depth on nodes and arcs.
    bpy.ops.object.camera_add(location=(0, 0, 32))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 29.0
    scene.camera.rotation_euler = (0, 0, 0)
    bpy.ops.object.light_add(type="AREA", location=(0, 3, 13))
    bpy.context.object.data.energy, bpy.context.object.data.shape, bpy.context.object.data.size = 1800, "DISK", 18

    text("JAIN", (-13.1, 6.8, .2), .72, ink)
    text("F U S E D   I N F E R E N C E", (-7.6, 7.0, .2), .22, dim)
    text("FROM TABLE TO NEURAL GUIDANCE", (-13.1, 6.14, .2), .24, teal)
    text("NATIVE CPU CAPTURE  /  DESIGN PREVIEW", (5.7, 7.0, .2), .20, gold)
    line("Header rule", [(-13.1,5.72,.15),(13.0,5.72,.15)], .012, structure)
    text("01  DATA", (-13.1, 4.98, .2), .25, ink)
    text("02  DOPE ENCODER", (-7.9, 4.98, .2), .25, teal)
    text("03  JOPE RANKER", (2.7, 4.98, .2), .25, purple)
    text("Frozen checkpoint · changing activations", (-7.9, 4.48, .2), .21, dim)
    text("Color = weight origin", (8.3, 4.49, .2), .20, dim)

    # Four small neural microscopes display selected connections; no implied
    # cross-layer edge or residual is invented between independent affine taps.
    cases = data["cases"]
    first = choose_layers(cases[0])
    slots = [(-5.7,2.15),(-5.7,-1.1),(4.9,2.15),(4.9,-1.1)]
    all_cases = [choose_layers(case) for case in cases]
    edge_receipts = []
    case_frames = []
    for case_index, case in enumerate(cases):
        start = 1 + (case_index * args.frames) // len(cases)
        end = ((case_index + 1) * args.frames) // len(cases)
        case_frames.append((start, end))
    for li, (layer, (cx, cy)) in enumerate(zip(first, slots)):
        origin = "dope" if li < 2 else "jope"
        color, labelmat = (CYAN, teal) if li < 2 else (VIOLET, purple)
        x, w, b, y = affine_arrays(layer)
        # Select nodes from strongest weights once; use same identities for all
        # captured tables. This is a declared subset, never all-neuron coverage.
        candidates = np.argsort(-np.abs(w).reshape(-1), kind="stable")
        picked, ins, outs = [], set(), set()
        for flat in candidates:
            dst, src = np.unravel_index(flat, w.shape)
            if len(ins | {int(src)}) > 14 or len(outs | {int(dst)}) > 14:
                continue
            ins.add(int(src)); outs.add(int(dst)); picked.append((int(dst), int(src)))
            if len(picked) == args.edges:
                break
        ins, outs = sorted(ins), sorted(outs)
        positions_in = {n:(cx-2.6,cy+(i-(len(ins)-1)/2)*.13,.4) for i,n in enumerate(ins)}
        positions_out = {n:(cx+2.6,cy+(i-(len(outs)-1)/2)*.13,.4) for i,n in enumerate(outs)}
        cube(f"{origin} affine bay {li}",(cx,cy,.0),(7.7,2.82,.10),panel,.10)
        text(f"{origin.upper()}  /  AFFINE {li%2+1:02}",(cx-3.55,cy+1.06,.2),.19,labelmat)
        short = layer["name"].replace("dope_model.router.", "").replace("adapter.", "")
        text(short[-52:],(cx-3.55,cy-1.13,.2),.15,dim)
        text(f"{w.shape[1]} → {w.shape[0]}   ·   {len(picked)} selected edges",(cx+.13,cy+1.06,.2),.16,dim)
        normalization = max(abs(float(affine_arrays(layers[li])[0][0,src]*w[dst,src]))
                            for layers in all_cases for dst,src in picked) or 1.0
        activation_scale = max(float(np.max(np.abs(affine_arrays(layers[li])[3][0]))) for layers in all_cases) or 1.
        input_scale = max(float(np.max(np.abs(affine_arrays(layers[li])[0][0]))) for layers in all_cases) or 1.
        for side, indices, positions, scale_value in (("in",ins,positions_in,input_scale),("out",outs,positions_out,activation_scale)):
            for n in indices:
                mat = material(f"{li}/{side}/{n}/activation",color,0)
                obj = sphere(f"{origin}/{layer['name']}/{side}[{n}]",positions[n],.072,mat)
                emission = mat.node_tree.nodes.get("Emission").inputs["Strength"]
                for ci, layers in enumerate(all_cases):
                    xx,ww,bb,yy=affine_arrays(layers[li])
                    value=float(xx[0,n] if side=="in" else yy[0,n])
                    emission.default_value=2.8*abs(value)/scale_value if value else 0
                    emission.keyframe_insert("default_value",frame=case_frames[ci][0])
                interpolation(mat.node_tree,"CONSTANT")
                obj["tensor_index"],obj["side"]=n,side
        for ei,(dst,src) in enumerate(picked):
            begin,end=Vector(positions_in[src]),Vector(positions_out[dst])
            offset=.12+.46*((ei%9)/8)
            points=[begin.lerp(end,t/12)+Vector((0,math.sin(math.pi*t/12)*offset,.12*math.sin(math.pi*t/12))) for t in range(13)]
            width=.004+.011*abs(float(w[dst,src]))/(float(np.max(np.abs(w))) or 1)
            obj=line(f"{origin}/{layer['name']}/w[{dst},{src}]",points,width,structure)
            obj["weight"],obj["weight_origin"]=float(w[dst,src]),origin
            pulse_mat=material(f"pulse/{li}/{ei}",color,0)
            pulse=sphere(f"term/{li}/w[{dst},{src}]*x[{src}]",begin,.028,pulse_mat)
            emission=pulse_mat.node_tree.nodes.get("Emission").inputs["Strength"]
            terms=[]
            for ci,layers in enumerate(all_cases):
                xx,ww,bb,yy=affine_arrays(layers[li])
                term=float(xx[0,src]*ww[dst,src]); terms.append(term)
                start,finish=case_frames[ci]
                # Editorial propagation phase follows operation order; magnitude
                # stays bound to one captured case, with no activation tweening.
                run_length=finish-start+1
                phase_start=start+int((li%2)*run_length*.22)+(ei%7)
                duration=max(8,int(run_length*.50))
                emission.default_value=0
                emission.keyframe_insert("default_value",frame=start)
                if phase_start>start:
                    emission.keyframe_insert("default_value",frame=phase_start-1)
                emission.default_value=3.8*(abs(term)/normalization)**.6 if term else 0
                emission.keyframe_insert("default_value",frame=phase_start)
                emission.keyframe_insert("default_value",frame=min(phase_start+duration,finish))
                if phase_start+duration<finish:
                    emission.default_value=0
                    emission.keyframe_insert("default_value",frame=phase_start+duration+1)
                for t,position in enumerate(points):
                    pulse.location=position
                    pulse.keyframe_insert("location",frame=phase_start+duration*t/12)
            interpolation(pulse,"LINEAR")
            interpolation(pulse_mat.node_tree,"CONSTANT")
            edge_receipts.append({"layer":layer["name"],"origin":origin,"out":dst,"in":src,"weight":float(w[dst,src]),"terms":terms})

    # Table text is switched at exact case boundaries, never interpolated.
    for ci,case in enumerate(cases):
        objects=[]
        objects.append(text(case["id"],(-13.05,4.46,.2),.25,ink))
        objects.append(text(f"{case['rows']} rows × {case['features']} inputs",(-13.05,3.96,.2),.20,dim))
        preview=case.get("table_preview",[])[:7]
        header=case.get("input_header",[])[:3]
        objects.append(text("  ".join(str(h)[:7] for h in header),(-13.05,3.35,.2),.20,teal))
        for ri,row in enumerate(preview):
            objects.append(text("  ".join(f"{float(v): 5.2f}" for v in row[:3]),(-13.05,2.91-ri*.36,.2),.20,ink))
        objects.append(text(f"DATA  {case['dataset_sha256'][:16]}",(-13.05,-.15,.2),.15,dim))
        objects.append(text("Numeric regression\nSynthetic diagnostic table\nSame frozen fused checkpoint",(-13.05,-1.15,.2),.20,dim))
        start,end=case_frames[ci]
        for obj in objects:
            for frame,visible in ((1,False),(start,True),(end+1,False)):
                obj.hide_render=not visible
                obj.keyframe_insert("hide_render",frame=frame)
            interpolation(obj,"CONSTANT")

    text("FROZEN WEIGHTS",(-13.05,-3.35,.2),.18,dim)
    text("CYAN  DOPE",(-13.05,-3.83,.2),.20,teal)
    text("VIOLET  JOPE",(-13.05,-4.23,.2),.20,purple)
    line("Proof rule",[(-13.1,-4.79,.15),(13,-4.79,.15)],.012,structure)
    text("RECORDED ACTIVATIONS · TIME EXPANDED",(-13.05,-5.35,.2),.24,gold)
    text("Brightness = |input × weight|   /   pulse motion = editorial replay",(-13.05,-5.88,.2),.20,dim)
    text("FUSED DIAGNOSTIC · SELECTED AFFINE CONNECTIONS",(-13.05,-6.52,.2),.18,ink)
    text("Native trace parity passed  ·  Source / tokens per second: pending",(-1.8,-6.52,.2),.18,dim)
    text("SOURCE CAPTURE",(9.1,2.95,.2),.20,ink)
    text("Next proof gate",(9.1,2.4,.2),.20,gold)
    text("Table → fresh fit\n→ exact source diff\n→ execute exported code",(9.1,1.5,.2),.21,dim)
    text("No token-rate\nmeasurement yet",(9.1,-.5,.2),.20,dim)
    text("Weights fixed.\nActivations change.",(9.1,-2.1,.2),.20,ink)
    text(f"BUNDLE  {data['bundle_digest'][-16:]}",(5.0,-5.34,.2),.17,dim)
    text(f"TRACE   {validation['trace_sha256'][:16]}",(5.0,-5.83,.2),.17,dim)

    # Numeric binding is a direct scalar socket; no RGB-to-float conversion.
    receipt={"schema":"jope.blender-review/v1","blender_version":bpy.app.version_string,
             "blender_build_hash":bpy.app.build_hash.decode(),"trace_sha256":validation["trace_sha256"],
             "bundle_digest":data["bundle_digest"],"case_frames":case_frames,"frames":args.frames,
             "fps":args.fps,"selection":"strongest absolute weight, capped at 14 input/output indices per affine",
             "edge_count":len(edge_receipts),"selected_edges":edge_receipts,
             "scope":"native fused diagnostic; editorial travel time; no whole-pipeline/source/rate claim"}
    scene["trace_sha256"]=validation["trace_sha256"]
    scene["evidence_scope"]=receipt["scope"]
    # Pack a small binding receipt into the blend; raw private arrays stay external.
    embedded=bpy.data.texts.new("TRACE_BINDING_RECEIPT.json")
    embedded.write(json.dumps(receipt,indent=2))
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    scene.frame_set(args.render_frame[0] if args.render_frame else 16)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    for frame in args.render_frame:
        scene.frame_set(frame)
        scene.render.filepath=str(args.output.parent/f"review-{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    if args.render_animation:
        scene.render.filepath=str(args.output.parent/"frames"/"frame-")
        bpy.ops.render.render(animation=True)
    print(json.dumps({"blend":str(args.output),"edges":len(edge_receipts),"stills":args.render_frame}))


if __name__=="__main__":
    main()
