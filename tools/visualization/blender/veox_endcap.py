#!/usr/bin/env python3
"""Build and render the six-second VEOX closing shot from the supplied vector mark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

WIDTH, HEIGHT, FPS, FRAMES = 1920, 1080, 30, 180
COPY = ("Turn what if into what’s next.",)
LETTER_PULSES = {
    "V": {"color": (.01, .72, 1.), "peak": 24, "end": 43, "strength": 4.0},
    "E": {"color": (.035, .34, 1.), "peak": 42, "end": 62, "strength": 4.5},
    "O": {"color": (.28, .09, 1.), "peak": 60, "end": 82, "strength": 5.0},
    "X": {"color": (.56, .075, 1.), "peak": 82, "end": 179, "strength": 9.0},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def contours(data):
    """Sample only the absolute SVG path commands used by this supplied mark.

    The O's opposite-winding inner contour is retained, as are both separate V
    components. Unsupported commands fail rather than silently changing a logo.
    """
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", data)
    cursor, start, points, result, index = (0., 0.), None, [], [], 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command not in {"M", "L", "C", "H", "V", "Z"}:
            raise ValueError(f"Unsupported logo SVG command {command}")
        if command == "Z":
            if not points or start is None:
                raise ValueError("Invalid closed SVG contour")
            result.append(points)
            cursor, points, start = start, [], None
            continue
        count = {"M": 2, "L": 2, "C": 6, "H": 1, "V": 1}[command]
        values = list(map(float, tokens[index:index + count]))
        if len(values) != count:
            raise ValueError("Truncated SVG path")
        index += count
        if command == "C":
            p0, p1, p2, p3 = cursor, values[:2], values[2:4], values[4:]
            for step in range(1, 33):
                t = step / 32
                points.append(tuple((1-t)**3*p0[k] + 3*(1-t)**2*t*p1[k]
                                    + 3*(1-t)*t*t*p2[k] + t**3*p3[k] for k in (0, 1)))
            cursor = tuple(p3)
        else:
            cursor = (values[0], cursor[1]) if command == "H" else (
                (cursor[0], values[0]) if command == "V" else tuple(values))
            if command == "M":
                if points:
                    raise ValueError("Unclosed SVG contour")
                start = cursor
            points.append(cursor)
    if points or not result:
        raise ValueError("Logo contains an unclosed or empty path")
    return result


def smooth(value):
    value = min(1., max(0., value))
    return value * value * (3. - 2. * value)


def material(bpy, name, rgb, *, emission=0., roughness=.25, metallic=.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    surface = nodes.new("ShaderNodeBsdfPrincipled")
    surface.inputs["Base Color"].default_value = (*rgb, 1)
    surface.inputs["Metallic"].default_value = metallic
    surface.inputs["Roughness"].default_value = roughness
    surface.inputs["Emission Color"].default_value = (*rgb, 1)
    surface.inputs["Emission Strength"].default_value = emission
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(surface.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])
    mix.inputs[0].default_value = 1
    return mat, mix.inputs[0], surface


def animate(socket, values):
    for frame, value in values:
        socket.default_value = value
        socket.keyframe_insert("default_value", frame=frame)


def make_curve(bpy, name, coordinates, mat, radius):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 20
    curve.bevel_depth, curve.bevel_resolution = radius, 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(coordinates) - 1)
    for point, coordinate in zip(spline.bezier_points, coordinates):
        point.co = coordinate
        point.handle_left_type = point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    return obj


def aim(obj, target):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def build(args):
    import bpy
    args.output = args.output.resolve()
    if args.output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Generated scene must live outside the checkout")
    if args.output.exists() or args.output.with_suffix(".receipt.json").exists():
        raise ValueError("Use a fresh output scene")
    logo_bytes = args.logo.read_bytes()
    logo_sha, font_sha = hashlib.sha256(logo_bytes).hexdigest(), digest(args.font)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage, scene.render.fps = 100, FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.world = bpy.data.worlds.new("Near-black studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.001, .002, .006, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .12
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_denoising = True

    floor, _, _ = material(bpy, "Obsidian mirror", (.022, .028, .045), roughness=.17, metallic=.85)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.name = "Reflective floor"
    bpy.context.object.location.z = 1.6
    bpy.context.object.data.materials.append(floor)

    marks = {}
    ivory = (.84, .87, .92, 1.)
    for letter, pulse in LETTER_PULSES.items():
        mark, reveal, surface = material(bpy, f"VEOX {letter} pulse", ivory[:3], emission=.38, roughness=.18, metallic=.26)
        animate(reveal, [(1, 0), (7, 0), (19, 1), (180, 1)])
        peak, end, strength = pulse["peak"], pulse["end"], pulse["strength"]
        color = (*pulse["color"], 1.)
        if letter == "X":
            # The arriving energy collects in X: one broad crest, a quiet
            # after-pulse, and a long violet hold through the final fade.
            power = [(1,.38),(64,.38),(82,strength),(98,6.2),(116,7.0),(135,5.3),(155,3.8),(180,2.8)]
            colors = [(1,ivory),(64,ivory),(78,color),(180,color)]
        else:
            power = [(1,.38),(peak-13,.38),(peak,strength),(peak+5,strength*.76),(end,.38),(180,.38)]
            colors = [(1,ivory),(peak-13,ivory),(peak-4,color),(peak+6,color),(end,ivory),(180,ivory)]
        animate(surface.inputs["Emission Strength"], power)
        animate(surface.inputs["Emission Color"], colors)
        marks[letter] = mark

    tree = ET.fromstring(logo_bytes)
    paths = [node.attrib["d"] for node in tree.iter() if node.tag.endswith("}path")]
    if len(paths) != 5 or tree.attrib.get("viewBox") != "0 0 79 20":
        raise ValueError("Expected the supplied five-path VEOX vector logo")
    scale = 8.8 / 79.
    for index, data in enumerate(paths):
        curve = bpy.data.curves.new(f"VEOX supplied path {index}", "CURVE")
        curve.dimensions, curve.fill_mode = "2D", "BOTH"
        curve.resolution_u, curve.extrude = 32, .025
        curve.bevel_depth, curve.bevel_resolution = .007, 3
        for contour in contours(data):
            spline = curve.splines.new("POLY")
            spline.points.add(len(contour)-1)
            for point, (x, y) in zip(spline.points, contour):
                point.co = ((x-39.5)*scale, (10-y)*scale, 0, 1)
            spline.use_cyclic_u = True
        obj = bpy.data.objects.new(f"VEOX vector {index}", curve)
        bpy.context.collection.objects.link(obj)
        obj.rotation_euler[0], obj.location.z = math.pi / 2, 3.92
        obj.data.materials.append(marks[("O", "X", "E", "V", "V")[index]])

    font = bpy.data.fonts.load(str(args.font))
    for index, line in enumerate(COPY):
        ink, opacity, _ = material(bpy, f"Closing copy {index}", (.71, .80, .87) if index == 0 else (.88, .93, .97), emission=.7)
        animate(opacity, [(1, 0), (43 + index*4, 0), (65 + index*4, 1), (180, 1)])
        text = bpy.data.curves.new(f"CTA {index}", "FONT")
        text.body, text.font, text.align_x = line, font, "CENTER"
        text.size, text.extrude, text.bevel_depth = .53, .001, .0006
        obj = bpy.data.objects.new(f"CTA {line}", text)
        bpy.context.collection.objects.link(obj)
        obj.rotation_euler[0], obj.location = math.pi / 2, (0, -.04, 2.2)
        obj.data.materials.append(ink)

    # Decorative closing filaments inherit the film's palette; they assert no model topology.
    for side in (-1, 1):
        for lane in range(13):
            tint = (.01, .56, .82) if side < 0 else (.43, .08, .78)
            thread, alpha, shader = material(bpy, f"Closing filament {side} {lane}", tint, emission=2.4)
            animate(alpha, [(1, 0), (8+lane, .8), (29+lane, .44), (54+lane, 0), (180, 0)])
            arc = make_curve(bpy, f"Light resolves {side} {lane}", [
                (side*10.5, 1.2+lane*.06, 2.5+lane*.12),
                (side*6.8, .7+lane*.06, 2.2+lane*.09),
                (side*3.7, .3+lane*.03, 2.6+lane*.035),
                (side*.12, .02, 3.0)], thread, .004 + (lane%3)*.001)
            arc.data.bevel_factor_end = 0
            arc.data.keyframe_insert("bevel_factor_end", frame=1+lane)
            arc.data.bevel_factor_end = 1
            arc.data.keyframe_insert("bevel_factor_end", frame=32+lane)

    for name, location, color, power, size in (
        ("Ivory softbox", (0,-4,9), (1,.93,.84), 1250, 8),
    ):
        lamp = bpy.data.lights.new(name, "AREA")
        lamp.energy, lamp.color, lamp.shape, lamp.size = power, color, "DISK", size
        lamp.specular_factor = 0
        obj = bpy.data.objects.new(name, lamp)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        aim(obj, (0,0,2))

    camera_data = bpy.data.cameras.new("VEOX locked camera")
    camera = bpy.data.objects.new("VEOX locked camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0,-20,7.1)
    aim(camera, (0,0,2.9))
    camera_data.type, camera_data.ortho_scale = "ORTHO", 16.8
    scene.camera = camera
    # Fade the entire photograph, including the physical reflection, to exact black.
    compositor = bpy.data.node_groups.new("VEOX finishing", "CompositorNodeTree")
    compositor.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    nodes, links = compositor.nodes, compositor.links
    image = nodes.new("CompositorNodeRLayers")
    glow = nodes.new("CompositorNodeGlare")
    glow.inputs["Type"].default_value = "Fog Glow"
    glow.inputs["Quality"].default_value = "High"
    glow.inputs["Threshold"].default_value = 1.8
    glow.inputs["Strength"].default_value = .32
    links.new(image.outputs["Image"], glow.inputs["Image"])
    multiply = nodes.new("ShaderNodeMix")
    multiply.data_type = "RGBA"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1
    links.new(glow.outputs["Image"], multiply.inputs[6])
    for frame in range(1, FRAMES+1):
        value = smooth((frame-1)/10) * (1-smooth((frame-158)/21))
        multiply.inputs[7].default_value = (value, value, value, 1)
        multiply.inputs[7].keyframe_insert("default_value", frame=frame)
    output = nodes.new("NodeGroupOutput")
    links.new(multiply.outputs[2], output.inputs[0])
    scene.compositing_node_group = compositor
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.dither_intensity = 0
    receipt = {"schema":"jain.veox-endcap-scene/v1", "status":"PASS", "width":WIDTH, "height":HEIGHT,
               "fps":FPS, "frame_start":1, "frame_end":FRAMES, "logo_sha256":logo_sha,
               "font_sha256":font_sha, "worker_sha256":digest(__file__), "copy":list(COPY),
               "logo_paths":len(paths), "vector_cubic_subdivisions":32,
               "letter_pulses":LETTER_PULSES, "strongest_longest_letter":"X",
               "note":"Illustrative brand closing shot; no inferred neural topology or benchmark claims."}
    embedded = bpy.data.texts.new("VEOX_ENDCAP_RECEIPT.json")
    embedded.write(json.dumps(receipt, indent=2))
    scene.frame_set(95)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    if digest(args.logo) != logo_sha or digest(args.font) != font_sha:
        raise ValueError("Logo or font changed during scene construction")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    save_json(args.output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt), flush=True)


def render(args):
    import bpy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_animation import select_gpu, verify_png
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Generated plates must live outside the checkout")
    receipt = json.loads(args.receipt.read_bytes())
    if (receipt.get("schema"), receipt.get("status"), receipt.get("frame_end")) != ("jain.veox-endcap-scene/v1", "PASS", FRAMES):
        raise ValueError("Expected a passing 180-frame VEOX scene receipt")
    if (receipt.get("width"), receipt.get("height"), receipt.get("fps"), receipt.get("frame_start")) != (WIDTH,HEIGHT,FPS,1):
        raise ValueError("Scene receipt has invalid picture/timing")
    if args.output_dir.exists() or not args.frame or len(set(args.frame)) != len(args.frame):
        raise ValueError("Fresh output and unique frame selection required")
    if any(not 1 <= frame <= FRAMES for frame in args.frame):
        raise ValueError("Requested frame escapes the scene")
    before = {"scene_sha256":digest(args.scene), "scene_receipt_sha256":digest(args.receipt),
              "worker_sha256":digest(__file__), "shared_worker_sha256":digest(Path(__file__).with_name("render_animation.py"))}
    bpy.ops.wm.open_mainfile(filepath=str(args.scene), load_ui=False, use_scripts=False)
    embedded = bpy.data.texts.get("VEOX_ENDCAP_RECEIPT.json")
    if embedded is None or json.loads(embedded.as_string()) != receipt:
        raise ValueError("Embedded VEOX receipt differs")
    scene = bpy.context.scene
    if (scene.render.resolution_x,scene.render.resolution_y,scene.render.fps/scene.render.fps_base,scene.frame_start,scene.frame_end) != (WIDTH,HEIGHT,FPS,1,FRAMES):
        raise ValueError("Saved scene picture/timeline differs")
    if (scene.render.image_settings.file_format,scene.render.image_settings.color_mode,scene.render.image_settings.color_depth) != ("PNG","RGB","8"):
        raise ValueError("Saved scene must produce native RGB8 PNG")
    devices = select_gpu(bpy,args.device,None)
    scene.render.engine, scene.cycles.device = "CYCLES", "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling, scene.cycles.adaptive_threshold = True, .005
    scene.cycles.adaptive_min_samples = 16
    scene.cycles.seed, scene.cycles.use_animated_seed = 20261104, False
    scene.cycles.use_denoising, scene.cycles.denoising_use_gpu = True, True
    scene.cycles.denoiser = "OPTIX" if args.device == "OPTIX" else "OPENIMAGEDENOISE"
    scene.render.use_persistent_data = True
    scene.render.resolution_percentage = 100
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema":"jain.veox-endcap-plates/v1", "status":"RUNNING", **before,
              "width":WIDTH,"height":HEIGHT,"fps":FPS,"samples":args.samples,
              "backend":args.device,"devices":devices,"gpu_only":True,"cpu_fallback":False,"frames":[]}
    try:
        with (args.output_dir/"frames.jsonl").open("x") as journal:
            for frame in sorted(args.frame):
                scene.frame_set(frame)
                path = args.output_dir/f"frame-{frame:06d}.png"
                scene.render.filepath = str(path)
                started = time.monotonic_ns()
                if bpy.ops.render.render(write_still=True) != {"FINISHED"}:
                    raise RuntimeError("Blender render did not finish")
                verify_png(path,WIDTH,HEIGHT,8)
                row = {"frame":frame,"file":path.name,"sha256":digest(path),"elapsed_ns":time.monotonic_ns()-started}
                result["frames"].append(row)
                journal.write(json.dumps(row)+"\n"); journal.flush()
                print(json.dumps(row), flush=True)
        if digest(args.scene) != before["scene_sha256"] or digest(args.receipt) != before["scene_receipt_sha256"] or digest(__file__) != before["worker_sha256"] or digest(Path(__file__).with_name("render_animation.py")) != before["shared_worker_sha256"]:
            raise ValueError("Render source changed during execution")
        result["status"] = "PASS"
    except BaseException as error:
        result["status"],result["error"] = "FAIL", f"{type(error).__name__}: {error}"
        raise
    finally:
        save_json(args.output_dir/"render.receipt.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--logo",type=Path,required=True)
    create.add_argument("--font",type=Path,required=True)
    create.add_argument("--output",type=Path,required=True)
    run = sub.add_parser("render")
    run.add_argument("--scene",type=Path,required=True)
    run.add_argument("--receipt",type=Path,required=True)
    run.add_argument("--output-dir",type=Path,required=True)
    run.add_argument("--frame",type=int,action="append",required=True)
    run.add_argument("--samples",type=int,default=96)
    run.add_argument("--device",choices=("METAL","OPTIX"),required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.command == "render" and not 16 <= args.samples <= 4096:
        parser.error("samples must lie in [16,4096]")
    (build if args.command == "build" else render)(args)


if __name__ == "__main__":
    main()
