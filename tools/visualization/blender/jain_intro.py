"""Build the editable 11-second 3D portion of the JAIN intro in Blender."""
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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jain_display_network import SPHERE


WIDTH, HEIGHT, FPS, LAST = 1920, 1080, 30, 330
BOARD_TOP, PITCH = -3.91, .61
CYAN, VIOLET = (.033, .66, 1., 1.), (.43, .13, .89, 1.)
IVORY = (.79, .80, .74, 1.)
MARKERS = {1: "Go", 28: "Hero stone contact", 75: "Answering move", 97: "Stone to algorithm",
           160: "Proposal and feedback A", 195: "Proposal and feedback B", 232: "Dendrite morph",
           316: "Exact demo camera", LAST: "Intro picture end"}


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def key(owner, field, frames):
    for frame, value in frames:
        setattr(owner, field, value)
        owner.keyframe_insert(field, frame=frame)


def visibility(obj, first, last=LAST):
    if first > 1:
        key(obj, "hide_render", [(1, True), (first, False)])
    if last < LAST:
        key(obj, "hide_render", [(last, False), (last + 1, True)])


def alpha_socket(mat):
    tree = mat.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    original = output.inputs["Surface"].links[0].from_socket
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.label = "Intro visibility"
    tree.links.new(transparent.outputs[0], mix.inputs[1])
    tree.links.new(original, mix.inputs[2])
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    return mix.inputs[0]


def principled(name, color, roughness, metallic=0.):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat, shader


def emitter(name, color, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = color
    emit.inputs["Strength"].default_value = strength
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emit.outputs[0], output.inputs["Surface"])
    return mat, emit.inputs["Strength"]


def curve(name, points, material, width, *, closed=False, parent=None):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.bevel_depth, data.bevel_resolution = width, 2
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, position in zip(spline.points, points, strict=True):
        point.co = (*position, 1.)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    data.materials.append(material)
    return obj


def round_box(name, center, dimensions, material, bevel, *, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.scale = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent, obj.location = parent, center
    obj.data.materials.append(material)
    modifier = obj.modifiers.new("Machined edge radius", "BEVEL")
    modifier.width, modifier.segments = bevel, 5
    obj.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
    return obj


def circle(radius, *, vertical=False, count=72):
    for index in range(count):
        angle = math.tau * index / count
        x, side = radius * math.cos(angle), radius * math.sin(angle)
        yield (x, 0., side) if vertical else (x, side, 0.)


def rounded_outline(width, height, radius, z):
    for x, y, initial in ((width/2-radius, height/2-radius, 0),
                          (-width/2+radius, height/2-radius, 90),
                          (-width/2+radius, -height/2+radius, 180),
                          (width/2-radius, -height/2+radius, 270)):
        for step in range(9):
            angle = math.radians(initial + step * 90 / 8)
            yield x + radius * math.cos(angle), y + radius * math.sin(angle), z


def make_stone(name, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    obj = bpy.context.object
    obj.name = name
    unit = [vertex.co.copy() for vertex in obj.data.vertices]
    for vertex in obj.data.vertices:
        vertex.co.x *= .285
        vertex.co.y *= .285
        vertex.co.z *= .102
    for face in obj.data.polygons:
        face.use_smooth = True
    obj.data.materials.append(material)
    return obj, unit


def grid_point(x, y):
    return ((x - 9) * PITCH, (y - 9) * PITCH, BOARD_TOP)


def land(stone, center, frame):
    position = Vector(center) + Vector((0., 0., .102))
    key(stone, "location", [(frame - 11, position + Vector((0., 0., 2.))),
                            (frame - 3, position + Vector((0., 0., .57))), (frame, position)])
    visibility(stone, frame - 11)


def build_board():
    slab, shader = principled("Black mineral board", (.023, .031, .043, 1.), .27, .35)
    nodes, links = slab.node_tree.nodes, slab.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    stretch = nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    stretch.inputs[1].default_value = (.8, 19., 5.)
    links.new(coords.outputs["Generated"], stretch.inputs[0])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.
    links.new(stretch.outputs[0], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (.010, .015, .024, 1.)
    ramp.color_ramp.elements[1].color = (.039, .048, .056, 1.)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    key(alpha_socket(slab), "default_value", [(1, 1.), (231, 1.), (282, 0.)])
    board = round_box("Beveled Go board", (0., 0., BOARD_TOP-.136), (11.86, 11.86, .272), slab, .10)
    visibility(board, 1, 282)

    line_mat, _ = emitter("Quiet silver Go grid", (.23, .27, .30, 1.), .22)
    key(alpha_socket(line_mat), "default_value", [(1, 1.), (234, 1.), (268, 0.)])
    for index in range(19):
        a = (index - 9) * PITCH
        bound = 9 * PITCH
        for points in (((a, -bound, BOARD_TOP+.009), (a, bound, BOARD_TOP+.009)),
                       ((-bound, a, BOARD_TOP+.009), (bound, a, BOARD_TOP+.009))):
            visibility(curve("Go board intersection lines", points, line_mat, .008), 1, 268)
    for x in (3, 9, 15):
        for y in (3, 9, 15):
            mark = curve("Go star point", list(circle(.041, count=20)), line_mat, .020, closed=True)
            mark.location = Vector(grid_point(x, y)) + Vector((0, 0, .010))
            visibility(mark, 1, 268)

    slate, _ = principled("Slate stones", (.014, .019, .028, 1.), .22)
    shell, _ = principled("Ivory shell stones", IVORY, .26)
    for index, (x, y) in enumerate(((3, 3), (15, 15), (3, 15), (15, 3), (9, 3), (9, 15), (3, 9))):
        stone, _ = make_stone("Opening Go stone", shell if index % 2 else slate)
        stone.location = Vector(grid_point(x, y)) + Vector((0., 0., .102))
        visibility(stone, 1, 258 + index * 2)
    reply, _ = make_stone("Answering slate move", slate)
    land(reply, grid_point(9, 9), 75)
    visibility(reply, 64, 268)

    hero_mat, hero_shader = principled("Hero shell becomes an algorithm", IVORY, .22, .12)
    hero, unit = make_stone("Hero Go stone and algorithm tile", hero_mat)
    hero.shape_key_add(name="Go stone")
    tile = hero.shape_key_add(name="Algorithm tile")
    for target, source in zip(tile.data, unit, strict=True):
        radial = math.hypot(source.x, source.y)
        if radial > 1e-5:
            x = math.copysign(abs(source.x / radial)**.43, source.x)
            y = math.copysign(abs(source.y / radial)**.43, source.y)
            target.co = (.70*x*radial**.32, .47*y*radial**.32, .105*source.z)
    key(tile, "value", [(1, 0.), (97, 0.), (155, 1.)])
    home = Vector(grid_point(15, 9))
    land(hero, home, 28)
    key(hero, "location", [(97, home + Vector((0., 0., .102))),
                           (126, Vector((2.3, -.12, -2.44))), (155, Vector((.7, .06, -1.52))),
                           (232, Vector((.7, .06, -1.52))), (271, Vector((.0, .08, -.32)))])
    key(hero, "rotation_euler", [(1, (0., 0., 0.)), (97, (0., 0., 0.)),
                                 (155, (math.radians(18), 0., math.radians(-8)))])
    key(hero_shader.inputs["Base Color"], "default_value",
        [(1, IVORY), (113, IVORY), (152, (.035, .072, .112, 1.))])
    key(alpha_socket(hero_mat), "default_value", [(1, 1.), (235, 1.), (270, 0.)])
    visibility(hero, 17, 270)

    for index in range(3):
        pulse_mat, strength = emitter(f"Cyan impact ripple {index}", CYAN, 0.)
        pulse = curve("Cyan response on the Go grid", list(circle(1.)), pulse_mat, .018, closed=True)
        pulse.location = home + Vector((0., 0., .025 + index*.004))
        start = 31 + 8*index
        key(pulse, "scale", [(start, (.31, .31, 1.)), (start+31, (3.9, 3.9, 1.))])
        key(strength, "default_value", [(start, .0), (start+4, .52 - .10*index), (start+31, 0.)])
        visibility(pulse, start, start+31)
    seed_mat, seed_strength = emitter("Cyan seed below the stone", CYAN, 0.)
    seed = curve("Cyan intersection seed", list(circle(.31)), seed_mat, .018, closed=True)
    seed.location = home + Vector((0., 0., .022))
    key(seed_strength, "default_value", [(28, 0.), (38, 1.15), (92, .61), (120, 0.)])
    visibility(seed, 28, 120)
    return hero


def glyph(parent, variant, first, last, *, fade=False):
    frame_mat, frame_strength = emitter(f"Algorithm silhouette {variant}", CYAN if variant == "hero" else VIOLET, 1.)
    nodes_mat, nodes_strength = emitter(f"Algorithm operation nodes {variant}", IVORY, 1.)
    tile_z = .105 if variant == "hero" else .04
    frame = curve("Illuminated algorithm tile outline", list(rounded_outline(1.34, .84, .14, tile_z)),
                  frame_mat, .011, closed=True, parent=parent)
    children = [frame]
    if variant == "branch":
        points = ((-.43, 0.), (-.05, .17), (-.05, -.17), (.43, 0.))
        edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    else:
        points = ((-.43, 0.), (0., 0.), (.43, 0.))
        edges = ((0, 1), (1, 2))
    for a, b in edges:
        children.append(curve("Algorithm schematic connector", [(*points[a], tile_z), (*points[b], tile_z)],
                              frame_mat, .012, parent=parent))
    for point in points:
        children.append(round_box("Algorithm schematic operation", (*point, tile_z+.018), (.145, .126, .032),
                                  nodes_mat, .018, parent=parent))
    for child in children:
        visibility(child, first, last)
    if fade:
        for strength in (frame_strength, nodes_strength):
            key(strength, "default_value", [(first, .0), (first+18, .72), (235, .72), (last, .0)])
    return nodes_strength


def bezier(points, count=42):
    p0, p1, p2, p3 = map(Vector, points)
    return [(1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3
            for t in np.linspace(0., 1., count)]


def training(hero):
    home = (.7, .06, -1.50)
    evaluation = (4.5, .32, -1.35)
    hero_strength = glyph(hero, "hero", 132, 270, fade=True)
    aperture_mat, aperture_strength = emitter("Neutral evaluation aperture", (.50, .70, .84, 1.), .0)
    for radius in (.60, .68):
        aperture = curve("Evaluation aperture", list(circle(radius, vertical=True)), aperture_mat, .012, closed=True)
        aperture.location = evaluation
        visibility(aperture, 158, 230)
    key(aperture_strength, "default_value", [(158, 0.), (164, .27), (175, 1.1), (184, .30),
                                             (197, .27), (210, 1.1), (219, .30), (230, 0.)])
    for start, variant, offset in ((160, "linear", .04), (195, "branch", .32)):
        candidate = bpy.data.objects.new(f"Illustrative candidate {variant}", None)
        bpy.context.scene.collection.objects.link(candidate)
        candidate.scale = (.71, .71, .71)
        candidate.rotation_euler = (math.radians(25), 0., math.radians(-8))
        glyph(candidate, variant, start, start+21)
        key(candidate, "location", [(start, home), (start+7, (2.9, .26, -.98)),
                                    (start+13, evaluation), (start+21, evaluation)])
        key(candidate, "scale", [(start, (.18,)*3), (start+8, (.71,)*3), (start+17, (.71,)*3),
                                 (start+21, (.14,)*3)])
        path = bezier((evaluation, (3.4, -.9, -2.86-offset), (1.65, -.94, -2.73-offset), home))
        return_mat, return_strength = emitter(f"Illustrative feedback {variant}", VIOLET, .0)
        track = curve("Temporary evaluation return arc", path, return_mat, .016)
        visibility(track, start+18, start+34)
        key(track.data, "bevel_factor_end", [(start+18, .0), (start+31, 1.)])
        key(track.data, "bevel_factor_start", [(start+18, .0), (start+31, .0), (start+34, 1.)])
        key(return_strength, "default_value", [(start+18, .20), (start+22, 1.23),
                                               (start+31, .98), (start+34, .0)])
        key(hero_strength, "default_value", [(start+28, .72), (start+32, 1.65), (start+36, .72)])


def studio_light(name, pos, aim, color, power, size):
    data = bpy.data.lights.new(name, "AREA")
    data.color, data.energy, data.shape, data.size = color, power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = pos
    obj.rotation_euler = (Vector(aim) - obj.location).to_track_quat("-Z", "Y").to_euler()
    key(data, "energy", [(1, power), (231, power), (300, 0.)])
    visibility(obj, 1, 301)


def frozen_socket_values(tree):
    values = []
    for node in tree.nodes:
        for socket in (*node.inputs, *node.outputs):
            if not hasattr(socket, "default_value"):
                continue
            value = socket.default_value
            if isinstance(value, (int, float, bool)):
                values.append((socket, value))
            elif hasattr(value, "__len__") and not isinstance(value, str):
                values.append((socket, tuple(value)))
    tree.animation_data_clear()
    for socket, value in values:
        socket.default_value = value


def coordinates(mesh):
    value = np.empty(len(mesh.vertices)*3, dtype="<f4")
    mesh.vertices.foreach_get("co", value)
    return value.reshape(-1, 3)


def mesh_digest(objects):
    data = hashlib.sha256()
    for obj in objects:
        mesh = obj.data
        data.update(obj.name.encode())
        data.update(coordinates(mesh).tobytes())
        loops = np.empty(len(mesh.loops), dtype="<i4")
        mesh.loops.foreach_get("vertex_index", loops)
        data.update(loops.tobytes())
        for layer in mesh.uv_layers:
            uv = np.empty(len(mesh.loops)*2, dtype="<f4")
            layer.data.foreach_get("uv", uv)
            data.update(layer.name.encode())
            data.update(uv.tobytes())
    return data.hexdigest()


def network_intro(root, nodes, edges, stages, expected_nodes):
    orb_size = len(SPHERE)
    geometry = {obj.name: coordinates(obj.data).reshape(-1, orb_size, 3) for obj in nodes}
    centers = {name: points.mean(axis=1) for name, points in geometry.items()}
    identity = sorted((float(center[0]), float(center[2]), float(center[1]), name, index)
                      for name, points in centers.items() for index, center in enumerate(points))
    if len(identity) != expected_nodes:
        raise ValueError("Saved network node geometry disagrees with the Dendrite receipt")
    inverse = root.matrix_world.inverted()
    anchors = {name: np.empty((len(points), 3), dtype=np.float32) for name, points in centers.items()}
    for rank, (_, _, _, name, index) in enumerate(identity):
        grid = rank * 361 // len(identity)
        x, y, z = grid_point(grid // 19, grid % 19)
        angle = rank * 2.399963229728653
        anchors[name][index] = inverse @ Vector((x+.048*math.cos(angle), y+.048*math.sin(angle), z+.045))
    for obj in nodes:
        original = geometry[obj.name]
        center = centers[obj.name]
        start = (original - center[:, None, :])*.34 + anchors[obj.name][:, None, :]
        bands = np.argmin(np.abs(center[:, :1] - stages), axis=1)
        obj.shape_key_add(name="Saved Dendrite endpoint")
        for band in sorted(set(bands.tolist())):
            affected = bands == band
            origin = obj.shape_key_add(name=f"Board anchors {band:02d}")
            lift = obj.shape_key_add(name=f"Lift arc {band:02d}")
            source = original.copy()
            source[affected] = start[affected]
            origin.data.foreach_set("co", source.ravel())
            raised = original.copy()
            raised[affected, :, 2] += .96
            lift.data.foreach_set("co", raised.ravel())
            first = 228+band
            key(origin, "value", [(1, 1.), (first, 1.), (294+band, 0.)])
            key(lift, "value", [(1, 0.), (first, 0.), (first+27, 1.), (294+band, 0.)])
        mat = obj.data.materials[0]
        frozen_socket_values(mat.node_tree)
        key(alpha_socket(mat), "default_value", [(1, 0.), (226, 0.), (254, 1.)])
        visibility(obj, 225)
    for obj in edges:
        mat = obj.data.materials[0]
        frozen_socket_values(mat.node_tree)
        band = int(np.argmin(np.abs(stages - coordinates(obj.data)[:, 0].mean())))
        first = 301 + band // 2
        key(alpha_socket(mat), "default_value", [(1, 0.), (first, 0.), (316, 1.)])
        visibility(obj, first)


def camera_intro(scene, final_camera):
    end = final_camera.matrix_world.copy()
    focal = Vector((0., 0., .3))
    end_rotation = end.to_quaternion()
    backward = end_rotation @ Vector((0., 0., 1.))
    data = bpy.data.cameras.new("Go to network perspective")
    data.sensor_width, data.clip_end = 36., 1100.
    camera = bpy.data.objects.new("Intro camera", data)
    scene.collection.objects.link(camera)
    camera.rotation_mode = "QUATERNION"
    poses = [(1, (-6.1, -13.1, 7.9), (-2.25, .1, -2.38), 42.),
             (28, (-5.78, -12.85, 7.36), (-2.03, .13, -2.25), 43.),
             (96, (-4.3, -14.1, 7.0), (-1.22, .17, -2.12), 43.),
             (155, (-3.4, -16.0, 6.0), (-1.0, .15, -1.05), 43.),
             (232, (-2.9, -18.0, 6.4), (-.65, .1, -.75), 44.)]
    for frame, location, aim, lens in poses:
        camera.location = location
        camera.rotation_quaternion = (Vector(aim) - camera.location).to_track_quat("-Z", "Y")
        data.lens = lens
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        data.keyframe_insert("lens", frame=frame)
    for frame, lens in ((272, 65.), (299, 130.), (315, 430.)):
        camera.location = focal + backward * (final_camera.data.ortho_scale*lens/data.sensor_width)
        camera.rotation_quaternion = end_rotation
        data.lens = lens
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        data.keyframe_insert("lens", frame=frame)
    scene.camera = camera
    scene.timeline_markers.get(MARKERS[1]).camera = camera
    scene.timeline_markers.get(MARKERS[316]).camera = final_camera
    return end


def save(args):
    source, receipt_path, output = (path.resolve() for path in (args.source_scene, args.source_receipt, args.output))
    if source == output or output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("Use a fresh intro output path")
    source_receipt = json.loads(receipt_path.read_bytes())
    source_hash = digest(source)
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    embedded = bpy.data.texts.get("JAIN_CINEMATIC_RECEIPT.json")
    if embedded is None or json.loads(embedded.as_string()) != source_receipt:
        raise ValueError("Source Blender scene and receipt disagree")
    network = source_receipt["network"]
    if network["spatial_layout"] != "dendrite":
        raise ValueError("Use the capture-bound Dendrite scene")
    scene.frame_set(1)
    final_camera = scene.camera
    root = bpy.data.objects[network["root_object_name"]]
    root_pose = root.matrix_world.copy()
    root.animation_data_clear()
    root.matrix_world = root_pose
    floor = bpy.data.objects["Reflective midnight floor"]
    nodes = sorted((obj for obj in root.children if obj.name.startswith("JAIN measured neuron samples")),
                   key=lambda obj: obj.name)
    edges = sorted((obj for obj in root.children if obj.name.startswith("JAIN actual sampled weight strands")),
                   key=lambda obj: obj.name)
    if sum(len(obj.data.polygons) for obj in edges) != network["displayed_strands"]:
        raise ValueError("Saved network edge mesh disagrees with its captured inventory")
    geometry_before = mesh_digest(nodes + edges)
    retained = {root, floor, final_camera, *nodes, *edges}
    for obj in list(scene.objects):
        if obj not in retained:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.timeline_markers.clear()
    for frame, name in MARKERS.items():
        scene.timeline_markers.new(name, frame=frame)
    scene.frame_start, scene.frame_end = 1, LAST
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.fps, scene.render.fps_base = FPS, 1.
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.cycles.samples = 24
    scene.render.film_transparent = False
    camera_pose = camera_intro(scene, final_camera)
    hero = build_board()
    training(hero)
    studio_light("Shell and board softbox", (-3., -4., 8.), (0., 0., BOARD_TOP), (1., .91, .82), 1180., 7.)
    studio_light("Cool slate grazing reflection", (4., 4., 4.), (0., 0., BOARD_TOP), (.58, .76, 1.), 850., 6.)
    studio_light("Ivory edge strip", (-8., 2., -.1), (0., 0., BOARD_TOP), (.76, .87, 1.), 280., 5.)
    scene.frame_set(1)
    network_intro(root, nodes, edges, np.array([stage["x"] for stage in network["stage_planes"]]),
                  network["displayed_nodes"])
    scene.frame_set(LAST)
    camera_error = max(abs(camera_pose[r][c] - scene.camera.matrix_world[r][c]) for r in range(4) for c in range(4))
    rotor_error = max(abs(root_pose[r][c] - root.matrix_world[r][c]) for r in range(4) for c in range(4))
    if scene.camera != final_camera or camera_error > 1e-5 or rotor_error > 1e-5:
        raise ValueError("Intro does not land on the exact Dendrite demo pose")
    if any(block.value != 0. for obj in nodes for block in list(obj.data.shape_keys.key_blocks)[1:]):
        raise ValueError("A morph displacement remains at the handoff")
    geometry_after = mesh_digest(nodes + edges)
    if geometry_before != geometry_after or digest(source) != source_hash:
        raise ValueError("Actual Dendrite basis geometry or original demo changed")
    receipt = {
        "schema": "jain.intro-scene/v1", "status": "PASS", "frame_start": 1, "frame_end": LAST,
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "renderer": "CYCLES",
        "blender_version": bpy.app.version_string, "generator_sha256": digest(__file__),
        "source_scene": str(source), "source_scene_sha256": source_hash,
        "source_receipt_sha256": digest(receipt_path), "source_trace_sha256": source_receipt["trace_sha256"],
        "source_operations_sha256": hashlib.sha256(json.dumps(network["operations"], sort_keys=True).encode()).hexdigest(),
        "source_mesh_and_uv_sha256": geometry_before, "final_mesh_and_uv_sha256": geometry_after,
        "displayed_nodes": network["displayed_nodes"], "displayed_strands": network["displayed_strands"],
        "source_endpoint_positions_sha256": network["ordered_endpoint_positions_f64_sha256"],
        "board_anchors": 361, "training_cycles": 2, "training_illustrative": True,
        "final_camera_error": camera_error, "final_rotor_error": rotor_error,
        "dendrite_exact_pose_start": 316, "markers": {str(frame): name for frame, name in MARKERS.items()},
        "handoff": "Frames 331–360 are composed from the existing demonstration.",
    }
    encoded = json.dumps(receipt, indent=2) + "\n"
    bpy.data.texts.new("JAIN_INTRO_RECEIPT.json").write(encoded)
    bpy.ops.file.pack_all()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    output.with_suffix(".receipt.json").write_text(encoded)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source-scene", "source-receipt", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    save(parser.parse_args(sys.argv[sys.argv.index("--")+1:]))


if __name__ == "__main__":
    main()
