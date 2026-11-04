"""Reference composition: real sampled NN layers, measured cells, full Rust.

Ingress/egress ribbons and the reflective stage are presentation geometry.
Internal neural strands retain actual captured parameter/channel identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import bpy
import numpy as np
from mathutils import Vector

from build_scene import interpolation, material, text
from h256_network import _curves
from h256_scene import full_source_pages
from input_influence import load_influence

from jain_display_network import build_display_network


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(data, validation, args):
    pages = full_source_pages(data, args, validation)
    if pages is None:
        raise ValueError("Cinematic composition requires complete verified Rust")
    influence = load_influence(args.influence, args.trace, data)
    from round_trip_metrics import load_metrics
    metrics = load_metrics(args.metrics, args.trace, data)
    cases = data["cases"]
    catalog = json.loads(args.dataset_catalog.read_text()) if args.dataset_catalog else None
    names = {item["case_id"]: item for item in catalog["cases"]} if catalog else {}
    if catalog:
        if len(catalog["cases"]) != len(cases) or len(names) != len(cases) or set(names) != {case["id"] for case in cases}:
            raise ValueError("Dataset name catalog differs from captured cases")
        for case in cases:
            item = names[case["id"]]
            if item["dataset_sha256"] != case["dataset_sha256"] or item["rows"] != case["rows"]:
                raise ValueError("Dataset display metadata differs from actual captured table")
            if not isinstance(item["display_name"], str) or not 1 <= len(item["display_name"]) <= 80:
                raise ValueError("Dataset display name must be bounded text")
    spans, cues, cursor = [], {}, 1
    for case, page in zip(cases, pages):
        first = cursor
        start = first + 36
        last = start + page["manifest"]["page_count"] * args.code_hold_frames + 5
        if args.animation_loop:
            start, last = first, first + args.loop_frames - 1
        spans.append((first, last)); cursor = last + 1
        cue = {"source_first": start, "source_last": last}
        for name, pattern in (() if args.animation_loop else (("training", r"(?m)^    fn fit\("), ("prediction", r"(?m)^    fn predict\("))):
            match = re.search(pattern, page["source"])
            if match:
                index = next(i for i, p in enumerate(page["manifest"]["pages"])
                             if p["source_char_start"] <= match.start() < p["source_char_end"])
                cue[name] = start + index * args.code_hold_frames
        cues[case["id"]] = cue

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.max_bounces = 4
    scene.cycles.glossy_bounces = 3
    scene.cycles.diffuse_bounces = 1
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, spans[-1][1]
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    scene.world = bpy.data.worlds.new("Midnight stage")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.001, .002, .006, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .04
    camera = bpy.data.objects.new("JAIN composition camera", bpy.data.cameras.new("JAIN composition camera"))
    scene.collection.objects.link(camera)
    camera.location = (3.2, -35, 14)
    camera.rotation_euler = (Vector((0, 0, .3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 38.
    scene.camera = camera
    ink = material("Window ink", (.78, .85, .94, 1), 1)
    muted = material("Quiet secondary", (.22, .31, .43, 1), 1)
    cyan = material("Cyan glow", (.015, .66, .92, 1), 2.)
    violet = material("Violet glow", (.44, .12, .74, 1), 1.8)
    panel = material("Window glass", (.003, .006, .013, 1), 1)
    header_mat = material("Window title bar", (.012, .019, .034, 1), 1)
    font = bpy.data.fonts.load("/System/Library/Fonts/Menlo.ttc")
    font.pack()

    def hud(body, x, y, size=.25, mat=ink, mono=False, centered=False):
        obj = text(body, (0, 0, 0), size, mat)
        obj.parent, obj.location, obj.rotation_euler = camera, (x, y, -5.8), (0, 0, 0)
        if mono: obj.data.font = font
        if centered: obj.data.align_x = "CENTER"
        return obj

    def rounded(name, x, y, w, h, radius, mat, z=-6.0, outline=False):
        points = []
        for cx, cy, start in ((x+w-radius, y+h-radius, 0), (x+radius, y+h-radius, 90),
                              (x+radius, y+radius, 180), (x+w-radius, y+radius, 270)):
            for i in range(9):
                angle = math.radians(start + i*90/8)
                points.append((cx + radius*math.cos(angle), cy + radius*math.sin(angle), z))
        if outline:
            curve = bpy.data.curves.new(name, "CURVE"); curve.dimensions = "3D"
            curve.bevel_depth, curve.bevel_resolution = .010, 2
            spline = curve.splines.new("POLY"); spline.points.add(len(points)-1)
            for point, co in zip(spline.points, points): point.co = (*co, 1)
            spline.use_cyclic_u = True
            curve.materials.append(mat)
            obj = bpy.data.objects.new(name, curve)
        else:
            mesh = bpy.data.meshes.new(name); mesh.from_pydata(points, [], [tuple(range(len(points)))])
            mesh.materials.append(mat); obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj); obj.parent = camera
        return obj

    def show(obj, first, last):
        for frame, hidden in sorted({1: True, first: False, last+1: True}.items()):
            obj.hide_render = hidden; obj.keyframe_insert("hide_render", frame=frame)
        interpolation(obj, "CONSTANT")

    # Match the reference's deliberately sparse composition.
    hud("JAIN", -17.85, 9.15, 1.08)
    hud("Just An Inference Network", -17.76, 8.45, .38, ink)
    rounded("Brand light", -18.45, 9.02, .045, 1.02, .02, cyan, -5.7)
    label = hud("TABULAR INPUT", -18.32, 3.45, .34)
    label.data.space_character = 1.25
    label = hud("FULL SOURCE CODE", 12.65, 9.40, .31, centered=True)
    label.data.space_character = 1.3
    rounded("Input card", -18.34, -9.87, 11.40, 12.73, .24, panel, -6.0)
    rounded("Input card rim", -18.34, -9.87, 11.40, 12.73, .24, material("Input rim", (.05,.24,.39,1), 1.5), -5.96, True)
    rounded("Source card", 6.95, -9.87, 11.40, 18.57, .26, panel, -5.99)
    rounded("Source card rim", 6.95, -9.87, 11.40, 18.57, .26, violet, -5.94, True)
    rounded("Source title bar", 6.98, 8.10, 11.34, .56, .21, header_mat, -5.92)
    for i, rgb in enumerate(((1., .10, .17, 1), (1., .69, .10, 1), (.18, .85, .30, 1))):
        rounded("Editor control", 7.22+i*.23, 8.29, .13, .13, .065, material("Editor dot", rgb, 1), -5.75)
    hud("RUST", 8.06, 8.28, .18, muted, True)

    table_rows = []
    for case, span in zip(cases, spans):
        values = influence["cases"][case["id"]]
        scores = np.asarray(values["scores"])
        csv_path = (args.trace.parent / values["bindings"]["csv_path"]).resolve()
        if not csv_path.is_relative_to(args.trace.parent.resolve()) or digest(csv_path) != case["dataset_sha256"]:
            raise ValueError("Expanded table CSV is not the actual captured dataset")
        with csv_path.open(newline="") as stream:
            reader = csv.reader(stream)
            csv_header, raw_rows = next(reader), list(reader)
        visible_ids = case["fit_row_ids"][:13]
        raw_table = [[float(cell) for cell in raw_rows[index]] for index in visible_ids]
        header = ["row", *case["input_header"]]
        if len(header) != 6 or len(values["raw_table"]) != 5 or csv_header != case["input_header"] or len(raw_rows) != case["rows"]:
            raise ValueError("Reference table requires row ID, four inputs and last-column target")
        widths = [1., 2., 2., 2., 2., 2.]
        display_name = names.get(case["id"], {}).get("display_name", case["id"].upper())
        objects = [hud(display_name, -18.25, 4.72, .34),
                   hud(f"{case['rows']:,} rows", -18.25, 4.08, .30, ink)]
        sensitivity_by_row = {rid: score for rid, score in zip(values["row_ids"],scores)}
        for row in range(len(raw_table)+1):
            row_values = header if row == 0 else [str(visible_ids[row-1]),
                *[f"{float(v):.2f}".replace("-", "−") for v in raw_table[row-1]]]
            y = 2.09 - row*.88
            for col, (width, value) in enumerate(zip(widths, row_values)):
                x = -18.12 + sum(widths[:col])
                target = col == len(widths)-1
                sensitivity = float(sensitivity_by_row[visible_ids[row-1]][col-1]) if row and 0 < col < 5 and visible_ids[row-1] in sensitivity_by_row else 0.
                if row == 0:
                    rgb = (.34,.015,.48,1) if target else (.022,.045,.065,1)
                elif target:
                    rgb = (.030,.009,.055,1)
                else:
                    rgb = (.006, .012+.020*sensitivity, .021+.028*sensitivity, 1)
                cell = rounded("Table cell", x, y-.32, width-.015, .86, .008,
                               material("Measured cell tint", rgb, 1), -5.88)
                objects.append(cell)
                obj = hud(str(value), x+width/2, y-.025, .34, ink, True, True)
                objects.append(obj)
                if row and 0 < col < 5 and sensitivity > 0:
                    objects.append(rounded("Relative cell influence", x+.10, y-.30,
                        max(.001,(width-.20)*sensitivity), .012, .005, cyan, -5.77))
        for obj in objects: show(obj, *span)
        table_rows.append({"case_id":case["id"], "target_column":case["input_header"][-1],
                           "target_display_index":5,"feature_score_shape":[5,4],
                           "displayed_row_ids":visible_ids,"displayed_rows":len(raw_table),
                           "sensitivity_row_ids":values["row_ids"],
                           "dataset_name":display_name,"total_rows":case["rows"]})

    # Real reflective floor; the emissive neurons supply the reflected colors.
    bpy.ops.mesh.primitive_plane_add(size=160, location=(0,0,-4.2))
    floor = bpy.context.object; floor.name = "Reflective midnight floor"
    floor_mat = bpy.data.materials.new("Black polished stage"); floor_mat.use_nodes = True
    principled = floor_mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (.006, .009, .018, 1)
    principled.inputs["Metallic"].default_value = .92
    principled.inputs["Roughness"].default_value = .27
    floor.data.materials.append(floor_mat)
    period = args.loop_frames if args.animation_loop else 144
    pose_step = 5 if args.animation_loop else 9
    network = build_display_network(data, spans, camera, width=11.,
                                    pulse_frames=period if args.animation_loop else 48)
    rotor = bpy.data.objects[network["root_object_name"]]
    rotor.location = (0., 0, .4)
    rotor.scale = (1., 1, .85)
    for frame in range(1, period+2, pose_step):
        phase = 2*math.pi*(frame-1)/period
        rotor.rotation_euler = (math.radians(.7)*math.sin(phase), math.radians(1.)*math.cos(phase),
                                 math.radians(-24+3*math.sin(phase)))
        rotor.keyframe_insert("rotation_euler", frame=frame)
    interpolation(rotor, "LINEAR")
    for curve in _curves(rotor): curve.modifiers.new("CYCLES")
    poses = {}
    for frame in range(1, period+2, pose_step):
        scene.frame_set(frame)
        poses[frame] = (camera.matrix_world.inverted() @ rotor.matrix_world).copy()

    def connection(name, values, rgb, fixed, port, incoming):
        mat = bpy.data.materials.new(name); mat.use_nodes = True
        tree = mat.node_tree; tree.nodes.clear()
        emit = tree.nodes.new("ShaderNodeEmission"); emit.inputs["Color"].default_value = rgb
        for (first, last), value in zip(spans, values):
            for frame in range(first, last+1, 5 if args.animation_loop else max(1,last-first)):
                phase = ((frame-first)/period + (0 if incoming else .65)) % 1
                envelope = .25 + .75 * ((1+math.sin(2*math.pi*phase))/2)**3
                emit.inputs["Strength"].default_value = 0. if value == 0 else (.035 + 1.65*math.sqrt(float(value)))*envelope
                emit.inputs["Strength"].keyframe_insert("default_value", frame=frame)
            emit.inputs["Strength"].default_value = 0. if value == 0 else (.035 + 1.65*math.sqrt(float(value)))*(.25+.75*((1+math.sin(2*math.pi*((last-first)/period+(0 if incoming else .65))))/2)**3)
            emit.inputs["Strength"].keyframe_insert("default_value", frame=last)
        for curve in _curves(tree):
            for point in curve.keyframe_points: point.interpolation = "LINEAR" if args.animation_loop else "CONSTANT"
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(emit.outputs[0], output.inputs["Surface"])
        curve = bpy.data.curves.new(name, "CURVE"); curve.dimensions = "3D"
        curve.bevel_depth, curve.bevel_resolution, curve.resolution_u = .0028, 1, 12
        spline = curve.splines.new("BEZIER"); spline.bezier_points.add(2)
        for point in spline.bezier_points: point.handle_left_type = point.handle_right_type = "AUTO"
        curve.materials.append(mat); obj = bpy.data.objects.new(name, curve)
        scene.collection.objects.link(obj); obj.parent = camera
        for frame in range(1, period+2, pose_step):
            p = poses[frame] @ Vector(port)
            dynamic = Vector((p.x,p.y,-6.12)); locked = Vector(fixed)
            a,b = (locked,dynamic) if incoming else (dynamic,locked)
            middle = a.lerp(b,.58); middle.x = a.x*.38+b.x*.62
            for point, position in zip(spline.bezier_points,(a,middle,b)):
                point.co = position; point.keyframe_insert("co", frame=frame)
        for fcurve in _curves(curve):
            for point in fcurve.keyframe_points: point.interpolation = "LINEAR"
            fcurve.modifiers.new("CYCLES")

    input_ports = network["input_node_ports"]
    for row in range(5):
        for col in range(4):
            scores = [influence["cases"][case["id"]]["scores"][row][col] for case in cases]
            for branch in range(4):
                index = (row*4+col)*4+branch
                port = input_ports[round(index*(len(input_ports)-1)/79)]
                connection("Conceptual cell ingress", scores, (.018,.62,.92,1),
                           (-6.92, .15+(index/79-.5)*1.15, -6.12), port, True)
    output_ports = network["output_node_ports"]
    residuals = [np.abs(np.asarray(case["outputs"]["ensemble_delta_prior"],dtype=float).reshape(-1)) for case in cases]
    bound = max(float(np.max(v)) for v in residuals)
    for index, port in enumerate(output_ports):
        values = [float(v[index%len(v)]/bound) if bound else 0. for v in residuals]
        connection("Conceptual source egress", values, (.48,.08,.85,1),
                   (6.92,.15+(index/max(1,len(output_ports)-1)-.5)*1.15,-6.12), port, False)

    # Same exact Rust lexer count and paired round-trip sample used by native
    # telemetry; the display never substitutes a warm repeated-emission rate.
    hud("TOTAL INFERENCE", 0., 9.45, .29, centered=True)
    for case, span in zip(cases, spans):
        metric = metrics["cases"][case["id"]]
        rate = metric["equivalent_rust_tokens_per_s"]
        labels = [hud(f"{metric['total_ms']:,.1f} ms",0.,8.48,.72,centered=True),
                  hud(f"{rate:,.0f} Rust tokens/sec",0.,7.80,.37,centered=True),
                  hud("Equivalent · data → complete Rust",0.,7.31,.20,muted,centered=True)]
        for label in labels:show(label,*span)

    page_receipts = []
    for case, info, span in zip(cases,pages,spans):
        manifest = info["manifest"]; first = cues[case["id"]]["source_first"]
        width = 11.15; height = width*manifest["height"]/manifest["width"]
        for index, page in enumerate(manifest["pages"] if not args.animation_loop else []):
            image = bpy.data.images.load(str(info["manifest_path"].parent/page["file"]), check_existing=True)
            image.pack(); mat = bpy.data.materials.new("Complete cinematic Rust page"); mat.use_nodes = True
            tree = mat.node_tree; tree.nodes.clear()
            texture = tree.nodes.new("ShaderNodeTexImage"); texture.image = image; texture.interpolation = "Closest"
            emit = tree.nodes.new("ShaderNodeEmission"); emit.inputs["Strength"].default_value = 1.
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(texture.outputs["Color"],emit.inputs["Color"]); tree.links.new(emit.outputs[0],output.inputs["Surface"])
            mesh = bpy.data.meshes.new("Whole source raster")
            mesh.from_pydata([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,.5,0)],[],[(0,1,2,3)])
            uv = mesh.uv_layers.new(name="UVMap")
            for loop, co in zip(uv.data,((0,0),(1,0),(1,1),(0,1))): loop.uv = co
            mesh.materials.append(mat); obj = bpy.data.objects.new("Complete Rust page",mesh)
            scene.collection.objects.link(obj); obj.parent = camera
            obj.location, obj.scale = (7.10+width/2,8.05-height/2,-5.86),(width,height,1)
            show(obj,first+index*args.code_hold_frames,
                 first+(index+1)*args.code_hold_frames-1 if index+1<len(manifest["pages"]) else span[1])
        page_receipts.append({"case_id":case["id"],"manifest_sha256":digest(info["manifest_path"]),
                              "manifest_path":str(info["manifest_path"]),
                              "rust_sha256":info["binding"]["rust_sha256"],"page_count":manifest["page_count"],
                              "all_original_bytes_reconstructed":True})

    # HUD glass/text and conceptual connectors do not appear in floor rays.
    for obj in scene.objects:
        if obj.parent == camera:
            obj.visible_glossy = False
            obj.visible_diffuse = False

    compositor = bpy.data.node_groups.new("Cinematic glow", "CompositorNodeTree")
    compositor.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    source = compositor.nodes.new("CompositorNodeRLayers")
    glow = compositor.nodes.new("CompositorNodeGlare"); glow.inputs["Type"].default_value = "Bloom"
    glow.inputs["Threshold"].default_value, glow.inputs["Strength"].default_value = 1.6,.15
    output = compositor.nodes.new("NodeGroupOutput")
    compositor.links.new(source.outputs["Image"],glow.inputs["Image"])
    compositor.links.new(glow.outputs["Image"],output.inputs[0]); scene.compositing_node_group = compositor
    result = {"schema":"jain.reference-cinematic-scene/v1","brand":"JAIN — Just An Inference Network",
        "trace_sha256":digest(args.trace),"influence_sha256":digest(args.influence),
        "round_trip_metrics_sha256":digest(args.metrics),"round_trip_metrics":metrics["cases"],"metric_device":"cpu",
        "dataset_catalog_sha256":digest(args.dataset_catalog) if args.dataset_catalog else None,
        "layout":"equal thirds: table / actual sampled neural layers / complete Rust",
        "source_hashes":{name:digest(HERE/name) for name in ("jain_cinematic.py","jain_display_network.py")},
        "network":network,"tables":table_rows,"case_frames":spans,"source_cues":cues,
        "whole_source_pages":page_receipts,"floor":{"z":-4.2,"metallic":.92,"roughness":.27},
        "connectors":{"input_strands":80,"output_strands":len(output_ports),
            "scope":"conceptual ingress/egress; input brightness uses fitted-GP cell sensitivity; output brightness uses measured ensemble residual magnitude"},
        "input_sensitivity_method":influence.get("method"),"rotation":{"center_degrees":-24,"amplitude_degrees":3,"period_frames":period},
        "animation_loop":args.animation_loop,"fps":args.fps,"loop_frames":period,
        "code_pages_root":str(args.code_pages.resolve()),
        "source_viewport":{"x":round((7.10/38+.5)*args.width),
                           "y":round((.5-8.05/(38*args.height/args.width))*args.height),
                           "width":round(11.15/38*args.width),
                           "height":round(11.15/38*args.width*pages[0]["manifest"]["height"]/pages[0]["manifest"]["width"])},
        "timing":"recorded inference; editorial rotation, pulse travel and source-page dwell","video_rendered":False,
        "blender_version":bpy.app.version_string}
    destination = args.output; destination.parent.mkdir(parents=True,exist_ok=True)
    destination.with_suffix(".receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    bpy.data.texts.new("JAIN_CINEMATIC_RECEIPT.json").write(json.dumps(result,indent=2))
    frames = [] if args.save_only else args.render_frame or [cues[cases[0]["id"]]["source_first"],
                                  cues[cases[0]["id"]].get("training",cues[cases[0]["id"]]["source_first"]),
                                  cues[cases[-1]["id"]].get("prediction",cues[cases[-1]["id"]]["source_first"])]
    scene.frame_set(frames[0] if frames else 1); bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    for frame in frames:
        if not scene.frame_start <= frame <= scene.frame_end: raise ValueError("Still frame outside scene")
        scene.frame_set(frame); scene.render.filepath = str(destination.parent/f"jain-{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
    print(json.dumps({"scene":str(destination),"frames":frames,"pages":sum(v["page_count"] for v in page_receipts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace","saved-validation","code-pages","gp-proof","influence","metrics","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--width",type=int,default=3840)
    parser.add_argument("--height",type=int,default=2160)
    parser.add_argument("--samples",type=int,default=24)
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--animation-loop",action="store_true")
    parser.add_argument("--loop-frames",type=int,default=120)
    parser.add_argument("--dataset-catalog",type=Path)
    parser.add_argument("--save-only",action="store_true")
    parser.add_argument("--code-hold-frames",type=int,default=6)
    parser.add_argument("--render-frame",type=int,action="append",default=[])
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if min(args.width,args.height,args.samples,args.code_hold_frames)<1: parser.error("Positive dimensions/samples/hold required")
    if not 1 <= args.fps <= 120 or args.loop_frames < 2 or args.animation_loop and args.loop_frames % 5:
        parser.error("FPS1..120 and loop period >=2 (multiple of5 for animation) required")
    if args.output.parent.exists() and any(args.output.parent.iterdir()):
        raise FileExistsError("Use a fresh output directory; preserve previous previews")
    from h256_stage2_validate import validate
    data,validation = validate(args.trace,saved_receipt=args.saved_validation,gp_root=args.gp_proof)
    build(data,validation,args)


if __name__ == "__main__":
    main()
