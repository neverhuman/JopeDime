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
SPATIAL_PULSE = {
    "coordinate": "world X", "start_frame": 19, "start_x": -4.5, "end_x": 4.5,
    "reference_travel_frames": 58, "speed_multiplier": 3,
    "travel_frames": 58 / 3, "front_width": .45, "tail_width": .75,
    "x_capture_start_frame": 37, "x_capture_full_frame": 42,
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


def make_wave_group(bpy):
    """One world-space head and wake cross every glyph at the same velocity."""
    tree = bpy.data.node_groups.new("VEOX shared traveling pulse", "ShaderNodeTree")
    for name, kind in (("Wave", "NodeSocketFloat"), ("Captured", "NodeSocketFloat"),
                       ("Color", "NodeSocketColor"), ("Strength", "NodeSocketFloat")):
        tree.interface.new_socket(name, in_out="OUTPUT", socket_type=kind)
    nodes, links = tree.nodes, tree.links
    output = nodes.new("NodeGroupOutput")

    def math_node(operation, a, b=None):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        for index, value in enumerate((a, b)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                node.inputs[index].default_value = value
            else:
                links.new(value, node.inputs[index])
        return node.outputs[0]

    position = nodes.new("ShaderNodeNewGeometry")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position.outputs["Position"], xyz.inputs[0])
    x = xyz.outputs["X"]
    head = nodes.new("ShaderNodeValue")
    head.name = "Continuous pulse head X"
    spec = SPATIAL_PULSE
    start, travel = spec["start_frame"], spec["travel_frames"]
    velocity = (spec["end_x"] - spec["start_x"]) / travel
    animate(head.outputs[0], [(1, spec["start_x"] + (1-start)*velocity),
                             (start, spec["start_x"]), (start+travel, spec["end_x"]),
                             (44, spec["start_x"] + (44-start)*velocity),
                             (FRAMES, spec["start_x"] + (44-start)*velocity)])
    # A single linear phase crosses material and letter boundaries. Fractional
    # end frame preserves the requested exact 3x time compression at 30fps.
    for layer in tree.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"

    distance = math_node("SUBTRACT", x, head.outputs[0])
    behind = math_node("LESS_THAN", distance, 0.)
    width = math_node("ADD", spec["front_width"],
                      math_node("MULTIPLY", behind, spec["tail_width"]-spec["front_width"]))
    normalized = math_node("DIVIDE", distance, width)
    squared = math_node("MULTIPLY", normalized, normalized)
    wave = math_node("EXPONENT", math_node("MULTIPLY", squared, -1.))
    links.new(wave, output.inputs["Wave"])

    passed = nodes.new("ShaderNodeMapRange")
    passed.interpolation_type = "SMOOTHSTEP"
    passed.clamp = True
    links.new(distance, passed.inputs["Value"])
    for name, value in (("From Min", -.20), ("From Max", .20), ("To Min", 1.), ("To Max", 0.)):
        passed.inputs[name].default_value = value
    capture = nodes.new("ShaderNodeValue")
    capture.name = "X stores the arriving pulse"
    animate(capture.outputs[0], [(1,0.), (spec["x_capture_start_frame"],0.),
                                 (spec["x_capture_full_frame"],1.), (FRAMES,1.)])
    links.new(math_node("MULTIPLY", passed.outputs[0], capture.outputs[0]), output.inputs["Captured"])

    coordinate = nodes.new("ShaderNodeMapRange")
    coordinate.clamp = True
    links.new(x, coordinate.inputs["Value"])
    coordinate.inputs["From Min"].default_value = spec["start_x"]
    coordinate.inputs["From Max"].default_value = spec["end_x"]
    palette = nodes.new("ShaderNodeValToRGB")
    palette.color_ramp.interpolation = "LINEAR"
    stops = ((0., (.005,.36,1.,1.)), (.32,(.02,.18,1.,1.)),
             (.65,(.28,.04,1.,1.)), (1.,(.62,.18,1.,1.)))
    for index, (at, color) in enumerate(stops):
        entry = palette.color_ramp.elements[index] if index < 2 else palette.color_ramp.elements.new(at)
        entry.position, entry.color = at, color
    links.new(coordinate.outputs[0], palette.inputs[0])
    links.new(palette.outputs["Color"], output.inputs["Color"])
    breath = nodes.new("ShaderNodeValue")
    breath.name = "Settled X glow"
    animate(breath.outputs[0], [(1,1.), (44,1.), (90,.86), (116,.93), (150,.72), (180,.61)])
    strength = math_node("MULTIPLY", math_node("ADD", 1.2, coordinate.outputs[0]), breath.outputs[0])
    links.new(strength, output.inputs["Strength"])
    return tree


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
    # This isolated brand shot uses display-referred neon colors. Standard
    # avoids AgX's bright-highlight desaturation, which made pulses look white.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
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
    traveling_wave = make_wave_group(bpy)
    for letter in "VEOX":
        mark, reveal, surface = material(bpy, f"VEOX {letter} pulse", (.25, .27, .30), emission=.08, roughness=.18, metallic=.26)
        animate(reveal, [(1, 0), (7, 0), (19, 1), (180, 1)])
        nodes, links = mark.node_tree.nodes, mark.node_tree.links
        field = nodes.new("ShaderNodeGroup")
        field.node_tree = traveling_wave
        emission = nodes.new("ShaderNodeEmission")
        links.new(field.outputs["Color"], emission.inputs["Color"])
        links.new(field.outputs["Strength"], emission.inputs["Strength"])
        pulse_mix = nodes.new("ShaderNodeMixShader")
        links.new(surface.outputs[0], pulse_mix.inputs[1])
        links.new(emission.outputs[0], pulse_mix.inputs[2])
        # Replace only the opaque surface of the existing reveal mix. A pure
        # colored active face keeps the ivory lighting from washing out neon.
        links.new(pulse_mix.outputs[0], reveal.node.inputs[2])
        if letter == "X":
            stored = nodes.new("ShaderNodeMath")
            stored.operation = "MAXIMUM"
            links.new(field.outputs["Wave"], stored.inputs[0])
            links.new(field.outputs["Captured"], stored.inputs[1])
            links.new(stored.outputs[0], pulse_mix.inputs[0])
        else:
            links.new(field.outputs["Wave"], pulse_mix.inputs[0])
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
        ink, opacity, _ = material(bpy, f"Closing copy {index}", (1., 1., 1.))
        bright_ink = ink.node_tree.nodes.new("ShaderNodeEmission")
        bright_ink.inputs["Color"].default_value = (1., 1., 1., 1.)
        bright_ink.inputs["Strength"].default_value = 1.8
        ink.node_tree.links.new(bright_ink.outputs[0], opacity.node.inputs[2])
        animate(opacity, [(1, 0), (43 + index*4, 0), (65 + index*4, 1), (180, 1)])
        text = bpy.data.curves.new(f"CTA {index}", "FONT")
        text.body, text.font, text.align_x = line, font, "CENTER"
        text.size, text.extrude, text.bevel_depth = .64, .001, .0006
        obj = bpy.data.objects.new(f"CTA {line}", text)
        bpy.context.collection.objects.link(obj)
        obj.rotation_euler[0], obj.location = math.pi / 2, (0, -.04, 2.2)
        obj.data.materials.append(ink)

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
    glow.inputs["Threshold"].default_value = .6
    glow.inputs["Strength"].default_value = .38
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
               "spatial_pulse":SPATIAL_PULSE, "strongest_longest_letter":"X",
               "decorative_ribbons":0, "pulse_surface":"pure colored emission", "view_transform":"Standard",
               "caption":{"size":.64,"emission_strength":1.8,"color":[1.,1.,1.],"steady":True},
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
