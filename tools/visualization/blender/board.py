"""Static JAIN review board with an opaque, complete-source inspection pane.

Layout is editorial. Every model-colored item comes from native observations.
Code pages preserve the complete emitted file, including fit and prediction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

import bpy
from mathutils import Vector
import numpy as np

from build_scene import cube, text, line, material, interpolation
from validate_trace import engine_evidence
from code_pages import reconstruct_source


def actual_round_states(data, spans, root):
    """Replay every recorded ranker call before displaying that run's source."""
    if data["schema"] != "jain.visual-run-trace/v1":
        return data, spans, []
    states, intervals, mapping = [], [], []
    for case, (start, end) in zip(data["cases"],spans):
        journal_path=root/case["native_journal_path"]
        if sha(journal_path) != case["native_journal_sha256"]:
            raise ValueError("Actual round journal changed")
        journal=json.loads(journal_path.read_text())
        conditioning=next(c for c in journal["calls"] if c["identity"]["phase"]=="fit_conditioning")
        admissions=[c for c in journal["calls"] if c["identity"]["phase"]=="ranker_admission"]
        for i,call in enumerate(admissions):
            first=start+(36*i)//len(admissions)
            last=start+(36*(i+1))//len(admissions)-1 if i+1<len(admissions) else end
            trace={family:[{**record,"call_id":c["call_id"]} for c in (conditioning,call) for record in c["trace"][family]]
                   for family in ("affine","tensors")}
            states.append({**case,"trace":trace})
            intervals.append((first,last))
            mapping.append({"case_id":case["id"],"run_id":case["run_id"],"call_id":call["call_id"],
                            "round":call["identity"]["round"],"frames":[first,last],
                            "selected_programs":call["outputs"]["selected_programs"]})
    return {**data,"cases":states}, intervals, mapping


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pages(root, cid, proof):
    path = root / cid / "manifest.json"
    data = json.loads(path.read_text())
    if data["schema"] != "jain.code-pages/v1" or data["source_sha256"] != proof["source_sha256"]:
        raise ValueError("Complete source pages do not match executed source")
    raw = reconstruct_source(data)
    for page in data["pages"]:
        filename = page["file"]
        if Path(filename).name != filename or not filename.endswith(".png"):
            raise ValueError("Invalid source-page path")
        if sha(path.parent / filename) != page["sha256"]:
            raise ValueError("Source-page image changed")
    if hashlib.sha256(raw).hexdigest() != proof["source_sha256"] or len(raw) != data["source_bytes"]:
        raise ValueError("Complete source coverage failed")
    return data, path


def build(data, args, validation):
    if args.code_pages is None or args.engine_proof is None:
        raise ValueError("Board requires --code-pages and --engine-proof for complete verified source")
    if args.code_hold_frames < 1:
        raise ValueError("Each complete source page needs at least one frame")
    evidence = engine_evidence(args.engine_proof, data, args.emit_metrics)
    cases = data["cases"]
    pages = [load_pages(args.code_pages, c["id"], p) for c, p in zip(cases, evidence)]
    spans, cursor, cues = [], 1, {}
    for case, (manifest, _), proof in zip(cases, pages, evidence):
        start = cursor
        source_start = start + 36
        cursor = source_start + manifest["page_count"] * args.code_hold_frames + 6
        spans.append((start, cursor - 1))
        source = Path(proof["source_path"]).read_text()
        cue = {"encoding": start + 12, "source_first": source_start,
               "source_last": source_start + (manifest["page_count"] - 1) * args.code_hold_frames}
        for name, pattern in (("training", r"(?m)^    fn fit\("), ("prediction", r"(?m)^    fn predict\("),
                              ("fitted_state", r"(?m)^fn fitted_model\(")):
            match = re.search(pattern, source)
            if match:
                page_index = next(i for i, p in enumerate(manifest["pages"])
                                  if p["source_char_start"] <= match.start() < p["source_char_end"])
                rows_before = sum(r["source_char_end"] <= match.start() for r in manifest["pages"][page_index]["rows"])
                if rows_before >= 12 and page_index+1 < manifest["page_count"]:
                    page_index += 1
                cue[name] = source_start + page_index * args.code_hold_frames
        cues[case["id"]] = cue

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples, scene.cycles.max_bounces = 24, 3
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, cursor - 1
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    scene.view_settings.exposure = 0
    scene.world = bpy.data.worlds.new("Neutral night")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.002, .004, .009, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .2
    camera = bpy.data.objects.new("Static board camera", bpy.data.cameras.new("Static board camera"))
    scene.collection.objects.link(camera)
    camera.location = (0, -30, 16)
    camera.rotation_euler = (Vector((0, 0, .5)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 31.5
    scene.camera = camera
    ink = material("UI text", (.78, .85, .93, 1), 1)
    muted = material("UI secondary", (.36, .45, .56, 1), 1)
    paper = material("Opaque UI", (.006, .011, .019, 1), 1)
    border = material("UI rules", (.035, .060, .085, 1), 1)
    positive = material("Positive data label", (.02, .64, .92, 1), 1)
    negative = material("Negative data label", (.9, .48, .10, 1), 1)
    verified = material("Verified evidence", (.15, .65, .40, 1), 1)
    mono = bpy.data.fonts.load("/System/Library/Fonts/Menlo.ttc") if Path("/System/Library/Fonts/Menlo.ttc").exists() else None

    def hud(body, x, y, size=.28, mat=ink, mono_font=False):
        obj = text(body, (0, 0, 0), size, mat)
        obj.parent, obj.location, obj.rotation_euler = camera, (x, y, -6), (0, 0, 0)
        if mono_font and mono:
            obj.data.font = mono
        return obj

    def tile(name, x, y, width, height, mat, depth=-6.05):
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata([(-width/2,-height/2,0),(width/2,-height/2,0),(width/2,height/2,0),(-width/2,height/2,0)],[],[(0,1,2,3)])
        obj=bpy.data.objects.new(name,mesh);scene.collection.objects.link(obj)
        mesh.materials.append(mat)
        obj.parent, obj.location, obj.rotation_euler = camera, (x, y, depth), (0, 0, 0)
        return obj

    def show(obj, start, end):
        for frame, hidden in sorted({1: True, start: False, end + 1: True}.items()):
            obj.hide_render = hidden
            obj.keyframe_insert("hide_render", frame=frame)
        interpolation(obj, "CONSTANT")

    hud("JAIN", -14.57, 7.45, .72)
    hud("DATA  →  PROBLEM ENCODING  →  VERIFIED RUST", -10.9, 7.65, .27)
    hud("RECORDED INFERENCE · TIME EXPANDED", 3.34, 7.62, .25, muted)
    hud("TABULAR DATA", -14.57, 6.85, .33)
    hud("PROBLEM ENCODING", -7.81, 6.85, .33)
    hud("RUST · COMPLETE EMITTED FILE", 3.34, 6.85, .33)
    hud("+ positive", -7.8, 6.26, .23, positive)
    hud("− negative", -5.85, 6.26, .23, negative)
    hud("Frozen weights · changing activity", -3.4, 6.26, .23, muted)

    selection = json.loads(args.selection.read_text()) if args.selection else None
    labels = {s["engine_case_id"]: s for s in selection["selected"]} if selection else {}
    for ci, (case, proof, span) in enumerate(zip(cases, evidence, spans)):
        start, end = span
        objects = []
        label = labels.get(case["id"], {})
        if label and label["dataset_sha256"] != case["dataset_sha256"]:
            raise ValueError("Selected dataset identity differs from actual inference")
        objects.append(hud(f"{ci+1}/{len(cases)}  {case['id'].upper()}", -14.57, 6.18, .29))
        description = label.get("description", "Captured numeric regression table")
        for li, chunk in enumerate(re.findall(r".{1,34}(?:\s+|$)", description)[:2]):
            objects.append(hud(chunk.strip(), -14.57, 5.68-li*.35, .23, muted))
        objects.append(hud(f"Synthetic · 5 of {case['rows']} FIT rows", -14.57, 4.76, .23, muted))
        csv_path = args.engine_proof / f"{case['id']}.csv"
        if sha(csv_path) != case["dataset_sha256"]:
            raise ValueError("Table preview bytes differ from inference")
        with csv_path.open() as stream:
            rows = list(csv.reader(stream))
        rng = random.Random(int(case["dataset_sha256"][:16], 16))
        row_ids = sorted(rng.sample(case["fit_row_ids"], min(5, len(case["fit_row_ids"]))))
        headers = ["row", *rows[0]]
        widths = [.57, 1.05, 1.05, 1.05, 1.05, 1.05]
        positions = np.cumsum([0, *widths[:-1]]) - 14.57
        for r in range(6):
            vals = headers if r == 0 else [str(row_ids[r-1]), *[f"{float(v):.2f}".replace("-", "−") for v in rows[row_ids[r-1]+1]]]
            for col, value in enumerate(vals):
                x, top = float(positions[col]), 4.15-r*.48
                objects.append(tile(f"Actual table {ci}/{r}/{col}", x+widths[col]/2, top, widths[col]-.015, .46, border if r == 0 else paper))
                obj = hud(value, x+widths[col]-.09, top-.10, .255, ink, True)
                obj.data.align_x = "RIGHT"
                objects.append(obj)
        objects.append(hud("MEASURED DESCRIPTORS", -14.57, .68, .25))
        objects.append(hud("v256 + 64×17 column statistics", -14.57, -1.58, .22, muted))
        objects.append(hud("FIT only · complete finite values", -14.57, -1.98, .22, muted))
        objects.append(hud(f"RUN {case.get('run_id',proof['engine_run_id'])[-16:]}", -14.57, -2.6, .20, muted, True))
        for obj in objects:
            show(obj, start, end)

    from network_graph import build_network, _calibrate, _atlas, _material, _mesh
    # One exact per-case atlas keeps the descriptor grid fast and deterministic.
    descriptor_values=np.asarray([c["outputs"]["dope629"][:256] for c in cases])
    descriptor_normalized,descriptor_scale=_calibrate(descriptor_values)
    descriptor_image,descriptor_sha=_atlas("Actual FIT v256",descriptor_normalized)
    descriptor_material=_material("Actual FIT descriptors",descriptor_image,spans)
    vertices,faces,uv=[],[],[]
    for j in range(256):
        x=-14.45+(j%32)*.17;y=.18-(j//32)*.17;offset=len(vertices)
        vertices.extend([(x,y,-6.01),(x+.14,y,-6.01),(x+.14,y+.14,-6.01),(x,y+.14,-6.01)])
        faces.append(tuple(range(offset,offset+4)));uv.append([((j+.5)/256,.5)]*4)
    descriptor=_mesh(scene.collection,"Measured v256 grid",vertices,faces,uv,descriptor_material)
    descriptor.parent=camera
    network_data,network_spans,round_mapping=actual_round_states(data,spans,args.engine_proof)
    network = build_network(network_data, network_spans, camera,bounds=(-8.,3.,-4.,7.))
    # One conceptual boundary follows the recorded neural computation. It is
    # not a learned edge or invented neural source decoder.
    seal = tile("Synthesis boundary", 2.58, 1.3, .52, 8.8, paper)
    hud("SYNTHESIS", 2.32, -3.58, .18, muted)

    page_receipts = []
    for ci, (case, proof, (manifest, manifest_path), (start, end)) in enumerate(zip(cases,evidence,pages,spans)):
        source_start = start+36
        for pi, page in enumerate(manifest["pages"]):
            image = bpy.data.images.load(str(manifest_path.parent/page["file"]), check_existing=True)
            if tuple(image.size) != (manifest["width"], manifest["height"]):
                raise ValueError("Source-page image dimensions differ from the verified manifest")
            image.pack()
            mat = bpy.data.materials.new(f"Complete source {ci}/{pi}")
            mat.use_nodes = True
            tree=mat.node_tree
            tree.nodes.clear()
            tex=tree.nodes.new("ShaderNodeTexImage");tex.image=image;tex.interpolation="Closest"
            emit=tree.nodes.new("ShaderNodeEmission");emit.inputs["Strength"].default_value=1
            out=tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(tex.outputs["Color"],emit.inputs["Color"]);tree.links.new(emit.outputs[0],out.inputs["Surface"])
            # Exact 684×660 page at 1920×1080. Opaque, nearer than all 3D geometry.
            width=manifest["width"]/(1920/31.5);height=manifest["height"]/(1920/31.5)
            mesh=bpy.data.meshes.new(f"Source plane {ci}/{pi}")
            mesh.from_pydata([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,.5,0)],[],[(0,1,2,3)])
            coords=mesh.uv_layers.new(name="UVMap")
            for loop,coordinate in zip(coords.data,[(0,0),(1,0),(1,1),(0,1)]):loop.uv=coordinate
            obj=bpy.data.objects.new(f"FULL SOURCE {ci}/{pi+1}",mesh);scene.collection.objects.link(obj)
            obj.data.materials.append(mat);obj.parent=camera
            obj.location=(3.34+width/2,6.56-height/2,-5.9);obj.rotation_euler=(0,0,0);obj.scale=(width,height,1)
            first=source_start+pi*args.code_hold_frames
            last=first+args.code_hold_frames-1 if pi+1<len(manifest["pages"]) else end
            show(obj,first,last)
        verify=proof["verification"]
        for obj in [hud(f"✓ 64 self-checks + {verify['engine_rows']} engine rows",3.34,-4.88,.25,verified),
                    hud(f"max |Δ| {verify['max_abs_error']:.2e}",3.34,-5.32,.24,verified),
                    hud(f"ALL {manifest['source_lines']} lines · {manifest['source_bytes']:,} bytes",3.34,-5.78,.24,muted)]:
            show(obj,source_start,end)
        metric=proof.get("emission_metric")
        if metric and metric.get("debug_assertions") is False:
            rate = f"{metric['value']/1e6:.2f} M" if metric['value'] >= 1e6 else f"{metric['value']/1000:,.1f} k"
            display=[hud(f"{rate} Rust lexical tokens/s",3.34,-6.32,.28),
                     hud("Release · warm repeated file · loop overhead included",3.34,-6.78,.21,muted)]
        else:
            display=[hud("Release emission measurement pending",3.34,-6.32,.24,muted)]
        for obj in display:show(obj,source_start,end)
        page_receipts.append({"case_id":case["id"],"manifest_sha256":sha(manifest_path),"source_sha256":proof["source_sha256"],
                              "pages":manifest["page_count"],"all_source_bytes_reconstructed":True,"source_begin":source_start})

    if args.show_gp_tree:
        for case,proof,(start,end) in zip(cases,evidence,spans):
            tree=proof.get("gp_tree")
            if not tree:raise ValueError("No verified winning GP tree")
            objs=[hud("GP TREE · OPERATOR PREVIEW",-7.8,-4.0,.26),
                  hud("Temporary debug · parameter counts; full native tree retained",-7.8,-4.37,.18,muted)]
            leaves=[]
            def weight(node):
                return max(1,len(node[0])*.35) if isinstance(node,list) else 1
            def draw(node,left,right,depth,parent=None):
                x=(left+right)/2;y=-4.92-depth*.82
                name=node[0] if isinstance(node,list) else str(node)
                ob=hud(name,x,y,.22,ink,True);ob.data.align_x="CENTER";objs.append(ob)
                if parent:
                    a=(parent[0],parent[1],-6.02);b=(x,y+.25,-6.02)
                    edge=line("Literal GP AST edge",[a,b],.012,muted);edge.parent=camera;objs.append(edge)
                if isinstance(node,list):
                    children=[c for c in node[1:] if isinstance(c,list) or c=="X"]
                    parameters=[c for c in node[1:] if isinstance(c,str) and c!="X"]
                    if parameters:
                        note=hud(f"{len(parameters)} params",x,y-.36,.18,muted,True);note.data.align_x="CENTER";objs.append(note)
                    total=sum(weight(c) for c in children);pos=left
                    for child in children:
                        width=(right-left)*weight(child)/max(1,total)
                        draw(child,pos,pos+width,depth+1,(x,y-.10));pos+=width
            draw(tree,-7.8,2.0,0)
            for obj in objs:show(obj,start+36,end)

    hud("Static layout · measured activations · source pages preserve every emitted byte",-14.57,-7.65,.23,muted)
    compositor=bpy.data.node_groups.new("Measured activity glow","CompositorNodeTree")
    compositor.interface.new_socket("Image",in_out="OUTPUT",socket_type="NodeSocketColor")
    layer=compositor.nodes.new("CompositorNodeRLayers")
    glow=compositor.nodes.new("CompositorNodeGlare")
    glow.inputs["Type"].default_value="Bloom"
    glow.inputs["Threshold"].default_value=2.0
    glow.inputs["Strength"].default_value=.14
    output_node=compositor.nodes.new("NodeGroupOutput")
    compositor.links.new(layer.outputs["Image"],glow.inputs["Image"])
    compositor.links.new(glow.outputs["Image"],output_node.inputs[0])
    scene.compositing_node_group=compositor
    receipt={"schema":"jain.full-code-board/v1","trace_sha256":validation["trace_sha256"],
             "native_source_commit":data.get("source_commit"),"bundle_digest":data["bundle_digest"],
             "blender_version":bpy.app.version_string,"blender_build_hash":bpy.app.build_hash.decode(),
             "case_frames":spans,"cue_frames":cues,"actual_frames":cursor-1,"fps":args.fps,
             "code_pages":page_receipts,"network":network["receipt"],"gp_tree_debug":bool(args.show_gp_tree),
             "native_round_frames":round_mapping,"descriptor_atlas_f32_sha256":descriptor_sha,
             "descriptor_calibration":descriptor_scale,
             "gp_tree_projection":"actual operator parent-child tree; parameter atom counts displayed; full literal tree retained in source evidence",
             "scope":"private native visualization; exact full-source inspection pages; editorial replay time"}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    bpy.data.texts.new("PRIVATE_BOARD_RECEIPT.json").write(json.dumps(receipt,indent=2))
    scene.frame_set(args.render_frame[0] if args.render_frame else next(iter(cues.values()))["source_first"])
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    for frame in args.render_frame:
        scene.frame_set(frame);scene.render.filepath=str(output.parent/f"review-{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
    if args.render_animation:
        scene.render.filepath=str(output.parent/"frames/frame-");bpy.ops.render.render(animation=True)
    print(json.dumps({"blend":str(output),"cue_frames":cues,"actual_frames":cursor-1}))
