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
LOCK_FRAME = 248
TRAIN_FIRST, TRAIN_LAST = 162, 233
TILE_SIZE = (1.42, .79875)
PROBLEM_UV, CANDIDATE_UV, TEST_UV = (.24, .63), (.54, .60), (.80, .63)
MARKERS = {1: "Go", 28: "Hero stone contact", 75: "Answering move", 99: "Stone to algorithm",
           TRAIN_FIRST: "Training circuit", 181: "Evaluation", 186: "Feedback returns",
           208: "Candidate replacement", 235: "Dendrite morph", LOCK_FRAME: "Exact demo camera",
           270: "Forward emphasis", LAST: "Intro picture end"}


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


def editorial_boost(mat, peak):
    tree = mat.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    original = output.inputs["Surface"].links[0].from_socket
    added = tree.nodes.new("ShaderNodeAddShader")
    tree.links.new(original, added.inputs[0])
    tree.links.new(original, added.inputs[1])
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.label = "Editorial emphasis over captured shader"
    tree.links.new(original, mix.inputs[1])
    tree.links.new(added.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    key(mix.inputs[0], "default_value", [(1, 0.), (peak-5, 0.), (peak, .68), (peak+6, 0.)])


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


class ScreenPlane:
    """Use the fixed training camera to author physical, face-on 3D props."""

    def __init__(self, camera, depth):
        self.matrix = camera.matrix_world.copy()
        self.rotation = self.matrix.to_quaternion()
        self.depth = depth
        self.width = depth * camera.data.sensor_width / camera.data.lens
        self.height = self.width * HEIGHT / WIDTH

    def point(self, u, v, front=0.):
        return self.matrix @ Vector(((u-.5)*self.width, (.5-v)*self.height, -self.depth+front))

    def px(self, value):
        return self.width * value / WIDTH

    def group(self, name, uv, front=0.):
        obj = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = self.point(*uv, front)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = self.rotation
        return obj


def lettering(name, text, size, parent, position, mat):
    data = bpy.data.curves.new(name, "FONT")
    data.body, data.size = text, size
    data.align_x = "CENTER"
    data.align_y = "CENTER"
    data.materials.append(mat)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent, obj.location = parent, position
    return obj


def land(stone, center, frame):
    position = Vector(center) + Vector((0., 0., .102))
    key(stone, "location", [(frame - 11, position + Vector((0., 0., 2.))),
                            (frame - 3, position + Vector((0., 0., .57))), (frame, position)])
    visibility(stone, frame - 11)


def build_board(view):
    slab, shader = principled("Warm satin Go board", (.15, .097, .055, 1.), .66)
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
    ramp.color_ramp.elements[0].color = (.113, .069, .037, 1.)
    ramp.color_ramp.elements[1].color = (.174, .113, .065, 1.)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    key(alpha_socket(slab), "default_value", [(1, 1.), (129, 1.), (161, 0.)])
    board = round_box("Beveled Go board", (0., 0., BOARD_TOP-.136), (11.86, 11.86, .272), slab, .10)
    visibility(board, 1, 161)

    line_mat, _ = principled("Dark 19-line Go grid", (.013, .009, .007, 1.), .93)
    key(alpha_socket(line_mat), "default_value", [(1, 1.), (129, 1.), (158, 0.)])
    for index in range(19):
        a = (index - 9) * PITCH
        bound = 9 * PITCH
        for points in (((a, -bound, BOARD_TOP+.009), (a, bound, BOARD_TOP+.009)),
                       ((-bound, a, BOARD_TOP+.009), (bound, a, BOARD_TOP+.009))):
            visibility(curve("Go board intersection lines", points, line_mat, .008), 1, 158)
    for x in (3, 9, 15):
        for y in (3, 9, 15):
            mark = curve("Go star point", list(circle(.041, count=20)), line_mat, .020, closed=True)
            mark.location = Vector(grid_point(x, y)) + Vector((0, 0, .010))
            visibility(mark, 1, 158)

    slate, _ = principled("Slate stones", (.022, .025, .032, 1.), .34)
    shell, _ = principled("Ivory shell stones", (.66, .65, .60, 1.), .30)
    for mat in (slate, shell):
        key(alpha_socket(mat), "default_value", [(1, 1.), (129, 1.), (159, 0.)])
    for index, (x, y) in enumerate(((3, 3), (15, 15), (4, 14), (15, 4), (10, 3), (13, 15), (3, 8))):
        stone, _ = make_stone("Opening Go stone", shell if index % 2 else slate)
        stone.location = Vector(grid_point(x, y)) + Vector((0., 0., .102))
        visibility(stone, 1, 159)
    reply, _ = make_stone("Answering slate move", slate)
    land(reply, grid_point(11, 10), 75)
    visibility(reply, 64, 159)

    hero_mat, hero_shader = principled("Hero shell becomes an algorithm", IVORY, .28)
    hero, unit = make_stone("Hero Go stone and algorithm tile", hero_mat)
    hero.shape_key_add(name="Go stone")
    tile = hero.shape_key_add(name="Algorithm tile")
    for target, source in zip(tile.data, unit, strict=True):
        radial = math.hypot(source.x, source.y)
        if radial > 1e-5:
            x = math.copysign(abs(source.x / radial)**.43, source.x)
            y = math.copysign(abs(source.y / radial)**.43, source.y)
            target.co = (TILE_SIZE[0]/2*x*radial**.28, TILE_SIZE[1]/2*y*radial**.28, .105*source.z)
    key(tile, "value", [(1, 0.), (99, 0.), (140, 1.)])
    home = Vector(grid_point(9, 9))
    land(hero, home, 28)
    key(hero, "location", [(99, home + Vector((0., 0., .102))), (117, view.point(.52, .615, -1.3)),
                           (140, view.point(.56, .60)), (151, view.point(.56, .60)),
                           (TRAIN_FIRST, view.point(*CANDIDATE_UV))])
    hero.rotation_mode = "QUATERNION"
    key(hero, "rotation_quaternion", [(1, (1., 0., 0., 0.)), (99, (1., 0., 0., 0.)),
                                      (140, view.rotation)])
    large = view.width * .25 / TILE_SIZE[0]
    training_size = view.width * .23 / TILE_SIZE[0]
    key(hero, "scale", [(1, (1., 1., 1.)), (99, (1., 1., 1.)), (140, (large, large, .66)),
                        (151, (large, large, .66)), (TRAIN_FIRST, (training_size, training_size, .66)),
                        (204, (training_size, training_size, .66)),
                        (208, (training_size*.95, training_size*.95, .66)),
                        (214, (training_size, training_size, .66))])
    key(hero_shader.inputs["Base Color"], "default_value",
        [(1, IVORY), (106, IVORY), (136, (.009, .015, .028, 1.))])
    hero_shader.inputs["Emission Color"].default_value = (.028, .043, .066, 1.)
    key(hero_shader.inputs["Emission Strength"], "default_value", [(1, 0.), (117, 0.), (140, .78), (233, .78)])
    key(alpha_socket(hero_mat), "default_value", [(1, 1.), (225, 1.), (234, 0.)])
    visibility(hero, 17, TRAIN_LAST)

    contour_mat, contour_strength = emitter("Cyan contour becomes algorithm border", CYAN, 0.)
    contour = curve("Continuous selected stone to algorithm contour", list(circle(.295, count=72)),
                    contour_mat, .010, closed=True, parent=hero)
    for index, point in enumerate(contour.data.splines[0].points):
        angle = math.tau * index / 72
        cos, sin = math.cos(angle), math.sin(angle)
        round_point = (.297*cos, .297*sin, -.035, 1.)
        square_point = (.715*math.copysign(abs(cos)**.43, cos), .404*math.copysign(abs(sin)**.43, sin), .087, 1.)
        key(point, "co", [(28, round_point), (99, round_point), (140, square_point)])
    key(contour_strength, "default_value", [(27, 0.), (28, .65), (34, 1.7), (97, .87),
                                              (140, 1.20), (204, 1.20), (208, .36),
                                              (215, 1.30), (225, 1.30), (234, 0.)])
    emission_color = next(node for node in contour_mat.node_tree.nodes if node.type == "EMISSION").inputs["Color"]
    key(emission_color, "default_value", [(1, CYAN), (207, CYAN), (214, VIOLET)])
    visibility(contour, 28, TRAIN_LAST)

    brace_mat, brace_strength = emitter("White code braces", (.90, .96, 1., 1.), .88)
    braces = lettering("Hero algorithm braces", "{  }", .63, hero, (0., .006, .113), brace_mat)
    key(brace_strength, "default_value", [(124, 0.), (140, .95), (204, .95), (208, .28),
                                            (214, .95), (225, .95), (234, 0.)])
    visibility(braces, 124, TRAIN_LAST)
    revision_mat, revision_strength = emitter("Subsequent candidate violet detail", VIOLET, .95)
    revision = curve("Training candidate revision glyph", [(0., .074, .118), (.075, 0., .118),
                      (0., -.074, .118), (-.075, 0., .118)], revision_mat, .009, closed=True, parent=hero)
    key(revision_strength, "default_value", [(207, 0.), (216, .97), (225, .97), (234, 0.)])
    visibility(revision, 207, TRAIN_LAST)

    for index in range(2):
        pulse_mat, strength = emitter(f"Cyan impact ripple {index}", CYAN, 0.)
        pulse = curve("Cyan response on the Go grid", list(circle(1.)), pulse_mat, .008, closed=True)
        pulse.location = home + Vector((0., 0., .025 + index*.004))
        start = 31 + 8*index
        key(pulse, "scale", [(start, (.31, .31, 1.)), (start+31, (3.9, 3.9, 1.))])
        key(strength, "default_value", [(start, .0), (start+4, .43 - .10*index), (start+31, 0.)])
        visibility(pulse, start, start+31)
    return hero


def bezier(points, count=42):
    p0, p1, p2, p3 = map(Vector, points)
    return [(1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3
            for t in np.linspace(0., 1., count)]


def training(view):
    problem = view.group("Training problem plane", PROBLEM_UV)
    problem_width, problem_height = .20*view.width, .25*view.height
    surface, _ = emitter("Problem slate", (.013, .027, .044, 1.), 1.)
    rail, rail_strength = emitter("Problem grid and result acknowledgement", CYAN, .42)
    quiet, quiet_strength = emitter("Persistent training circuit", (.15, .34, .47, 1.), .36)
    label_mat, label_strength = emitter("Training stage labels", (.70, .84, .94, 1.), .87)
    test_mat, test_strength = emitter("Neutral test marker", (.57, .78, .94, 1.), .57)
    key(alpha_socket(surface), "default_value", [(TRAIN_FIRST, 0.), (166, 1.), (226, 1.), (234, 0.)])
    props = [round_box("Training problem panel", (0., 0., 0.), (problem_width, problem_height, .062),
                       surface, view.px(10), parent=problem)]
    props.append(curve("Training problem border", list(rounded_outline(problem_width, problem_height, view.px(10), .041)),
                       rail, view.px(1.25), closed=True, parent=problem))
    inner_x, inner_y = problem_width*.425, problem_height*.365
    for col in (-.50, 0., .50):
        x = col*inner_x*1.32
        props.append(curve("Illustrative problem grid division", [(x, -inner_y, .042), (x, inner_y, .042)],
                           quiet, view.px(.85), parent=problem))
    for row in (-.60, -.20, .20, .60):
        y = row*inner_y
        props.append(curve("Illustrative problem grid division", [(-inner_x, y, .042), (inner_x, y, .042)],
                           quiet, view.px(.85), parent=problem))
    for row, y in enumerate((-.70, -.40, 0., .40, .70)):
        for col, x in enumerate((-.76, -.32, .16, .57)):
            width = view.px(17 + (row*11 + col*7) % 18)
            pos = (x*inner_x, y*inner_y, .05)
            props.append(curve("Abstract table cell", [pos, (pos[0]+width, pos[1], pos[2])],
                               rail, view.px(1.3), parent=problem))

    test = view.group("Training test plane", TEST_UV, .04)
    test_marker = curve("Training test marker", list(circle(view.px(72))), test_mat, view.px(1.7),
                        closed=True, parent=test)
    props.append(test_marker)
    for y, radius in ((-.18, .24), (0., .41), (.18, .30)):
        points = [(view.px(-72*radius), view.px(72*y), .02), (view.px(72*radius), view.px(72*y), .02)]
        props.append(curve("Neutral test aperture stroke", points, test_mat, view.px(1.6), parent=test))
    for label, uv in (("PROBLEM", (.24, .478)), ("CANDIDATE", (.54, .456)), ("TEST", (.80, .534))):
        parent = view.group(f"{label.title()} training label anchor", uv, .045)
        props.append(lettering(f"Training {label.lower()} label", label, view.px(27), parent, (0., 0., 0.), label_mat))
    for prop in props:
        visibility(prop, TRAIN_FIRST, TRAIN_LAST)

    routes = (
        ("Training problem to algorithm", ((.34, .63), (.363, .63), (.393, .60), (.425, .60)), CYAN, 166, 175),
        ("Training algorithm to test", ((.655, .60), (.686, .60), (.724, .63), (.7625, .63)), CYAN, 174, 182),
        ("Training test to problem", ((.80, .695), (.79, .890), (.26, .916), (.24, .754)), VIOLET, 186, 204),
    )
    for name, anchors, color, first, last in routes:
        path = bezier([view.point(*uv, .052) for uv in anchors])
        line = curve(name, path, quiet, view.px(1.23))
        visibility(line, TRAIN_FIRST, TRAIN_LAST)
        pulse_mat, pulse_strength = emitter(f"{name} pulse shader", color, 0.)
        pulse = curve(f"{name} active pulse", path, pulse_mat, view.px(2.35))
        visibility(pulse, first, last)
        key(pulse.data, "bevel_factor_end", [(first, 0.), (last-3, 1.)])
        key(pulse.data, "bevel_factor_start", [(first+3, 0.), (last, 1.)])
        key(pulse_strength, "default_value", [(first, 0.), (first+2, 1.6), (last-3, 1.6), (last, 0.)])

    key(quiet_strength, "default_value", [(TRAIN_FIRST, .05), (166, .49), (226, .49), (234, 0.)])
    key(label_strength, "default_value", [(TRAIN_FIRST, 0.), (166, .87), (226, .87), (234, 0.)])
    key(test_strength, "default_value", [(TRAIN_FIRST, .05), (167, .59), (177, .59), (182, 1.58),
                                          (188, .67), (226, .59), (234, 0.)])
    key(rail_strength, "default_value", [(TRAIN_FIRST, .05), (166, .44), (199, .44), (204, 1.16),
                                          (211, .51), (226, .51), (234, 0.)])


def studio_light(name, pos, aim, color, power, size):
    data = bpy.data.lights.new(name, "AREA")
    data.color, data.energy, data.shape, data.size = color, power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = pos
    obj.rotation_euler = (Vector(aim) - obj.location).to_track_quat("-Z", "Y").to_euler()
    key(data, "energy", [(1, power), (129, power), (161, 0.)])
    visibility(obj, 1, 161)


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


def network_intro(root, nodes, edges, stages, expected_nodes, view):
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
        angle = rank * 2.399963229728653
        u = CANDIDATE_UV[0] + .210*(grid//19/18.-.5) + .0017*math.cos(angle)
        v = CANDIDATE_UV[1] + .205*(grid%19/18.-.5) + .0027*math.sin(angle)
        anchors[name][index] = inverse @ view.point(u, v, .11)
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
            first, last = 235+band//2, 257+band//2
            key(origin, "value", [(1, 1.), (first, 1.), (last, 0.)])
            key(lift, "value", [(1, 0.), (first, 0.), (first+11, .78), (last, 0.)])
        mat = obj.data.materials[0]
        frozen_socket_values(mat.node_tree)
        band = int(np.median(bands))
        editorial_boost(mat, 275 + round(band * 23 / max(1, len(stages) - 1)))
        key(alpha_socket(mat), "default_value", [(1, 0.), (235, 0.), (251, 1.)])
        visibility(obj, 235)
    for obj in edges:
        mat = obj.data.materials[0]
        frozen_socket_values(mat.node_tree)
        band = int(np.argmin(np.abs(stages - coordinates(obj.data)[:, 0].mean())))
        editorial_boost(mat, 275 + round(band * 23 / max(1, len(stages) - 1)))
        first = 258 + band // 3
        key(alpha_socket(mat), "default_value", [(1, 0.), (first, 0.), (268, 1.)])
        visibility(obj, first)


def connector_digest(connectors):
    data = hashlib.sha256()
    for obj in connectors:
        data.update(obj.name.encode())
        for spline in obj.data.splines:
            data.update(np.asarray([tuple(point.co) for point in spline.bezier_points], dtype="<f4").tobytes())
    return data.hexdigest()


def reveal_connectors(connectors):
    for obj in connectors:
        controls = [(point, point.co.copy()) for spline in obj.data.splines for point in spline.bezier_points]
        obj.data.animation_data_clear()
        for point, coordinate in controls:
            point.co = coordinate
        mat = obj.data.materials[0]
        frozen_socket_values(mat.node_tree)
        incoming = obj.name.startswith(("Conceptual cell ingress", "Table border terminal"))
        editorial_boost(mat, 273 if incoming else 303)
        key(alpha_socket(mat), "default_value", [(1, 0.), (248, 0.), (266, 1.)])
        visibility(obj, 248)


def camera_intro(scene, final_camera):
    end = final_camera.matrix_world.copy()
    focal = Vector((0., 0., .3))
    end_rotation = end.to_quaternion()
    backward = end_rotation @ Vector((0., 0., 1.))
    data = bpy.data.cameras.new("Go to network perspective")
    data.sensor_width, data.clip_end, data.sensor_fit = 36., 1100., "HORIZONTAL"
    camera = bpy.data.objects.new("Intro camera", data)
    scene.collection.objects.link(camera)
    camera.rotation_mode = "QUATERNION"
    training_pose = ((-1.10, -26., 13.1), (0., 0., -1.66), 39.5)
    poses = [(1, (.54, -27., 12.), (0., 0., -1.13), 39.5),
             (28, (.31, -26.65, 11.8), (0., 0., -1.24), 39.5),
             (83, *training_pose), (234, *training_pose)]
    for frame, location, aim, lens in poses:
        camera.location = location
        camera.rotation_quaternion = (Vector(aim) - camera.location).to_track_quat("-Z", "Y")
        data.lens = lens
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        data.keyframe_insert("lens", frame=frame)
    for frame, lens in ((243, 100.), (247, 430.)):
        camera.location = focal + backward * (final_camera.data.ortho_scale*lens/data.sensor_width)
        camera.rotation_quaternion = end_rotation
        data.lens = lens
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        data.keyframe_insert("lens", frame=frame)
    scene.camera = camera
    scene.timeline_markers.get(MARKERS[1]).camera = camera
    scene.timeline_markers.get(MARKERS[LOCK_FRAME]).camera = final_camera
    return end, camera


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
    connector_names = ("Conceptual cell ingress", "Conceptual source egress", "Table border terminal", "Code border terminal")
    connectors = sorted((obj for obj in scene.objects if obj.type == "CURVE" and obj.name.startswith(connector_names)),
                        key=lambda obj: obj.name)
    if sum(len(obj.data.polygons) for obj in edges) != network["displayed_strands"]:
        raise ValueError("Saved network edge mesh disagrees with its captured inventory")
    expected_connectors = source_receipt["connectors"]
    if len(connectors) != expected_connectors["input_strands"] + expected_connectors["output_strands"] + 2:
        raise ValueError("Saved Dendrite panel connector geometry is incomplete")
    geometry_before = mesh_digest(nodes + edges)
    connectors_before = connector_digest(connectors)
    retained = {root, floor, final_camera, *nodes, *edges, *connectors}
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
    camera_pose, opening_camera = camera_intro(scene, final_camera)
    scene.frame_set(120)
    view = ScreenPlane(opening_camera, 21.)
    build_board(view)
    training(view)
    studio_light("Shell and board softbox", (-3., -4., 8.), (0., 0., BOARD_TOP), (1., .92, .83), 870., 8.)
    studio_light("Quiet slate grazing light", (4., 4., 4.), (0., 0., BOARD_TOP), (.69, .82, 1.), 350., 7.)
    studio_light("Ivory edge strip", (-8., 2., -.1), (0., 0., BOARD_TOP), (.76, .87, 1.), 160., 5.)
    visibility(floor, LOCK_FRAME)
    scene.frame_set(1)
    network_intro(root, nodes, edges, np.array([stage["x"] for stage in network["stage_planes"]]),
                  network["displayed_nodes"], view)
    reveal_connectors(connectors)
    scene.frame_set(LAST)
    camera_error = max(abs(camera_pose[r][c] - scene.camera.matrix_world[r][c]) for r in range(4) for c in range(4))
    rotor_error = max(abs(root_pose[r][c] - root.matrix_world[r][c]) for r in range(4) for c in range(4))
    if scene.camera != final_camera or camera_error > 1e-5 or rotor_error > 1e-5:
        raise ValueError("Intro does not land on the exact Dendrite demo pose")
    if any(block.value != 0. for obj in nodes for block in list(obj.data.shape_keys.key_blocks)[1:]):
        raise ValueError("A morph displacement remains at the handoff")
    geometry_after = mesh_digest(nodes + edges)
    if geometry_before != geometry_after or connectors_before != connector_digest(connectors) or digest(source) != source_hash:
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
        "morph_anchors": 361, "morph_anchor_scope": "Illustrative grid on the departing candidate tile",
        "training_cycles": 1, "training_illustrative": True, "candidate_replacement_frame": 208,
        "source_panel_curves": len(connectors), "source_panel_curve_coordinates_sha256": connectors_before,
        "forward_emphasis_scope": "Editorial intensity envelope over captured Dendrite materials",
        "final_camera_error": camera_error, "final_rotor_error": rotor_error,
        "dendrite_exact_pose_start": LOCK_FRAME, "markers": {str(frame): name for frame, name in MARKERS.items()},
        "handoff": "Authentic demo context joins the closing Blender frames; frames 331–360 use the moving demo.",
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
