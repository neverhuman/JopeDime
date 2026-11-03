"""Cinematic JAIN overview, driven by observed native affine operations."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import bpy
import numpy as np
from mathutils import Vector

from build_scene import material, cube, text, line, interpolation, choose_layers
from validate_trace import affine_arrays, engine_evidence

CYAN = (.025, .72, 1., 1.)
VIOLET = (.58, .18, 1., 1.)
AMBER = (1., .49, .09, 1.)
WHITE = (.78, .89, 1., 1.)
STEEL = (.12, .20, .29, 1.)


def build(data, args, validation):
    """Build, save and render requested stills; no simulation or frame handler."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.max_bounces = 3
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, args.frames
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -.35
    scene.world = bpy.data.worlds.new("Night")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.001, .003, .008, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .12

    cases = data["cases"]
    selected = [choose_layers(case) for case in cases]
    identity = lambda layer: (layer["network"], layer["name"], layer["sequence"])
    if any([identity(v) for v in layers] != [identity(v) for v in selected[0]] for layers in selected[1:]):
        raise ValueError("Selected native operation identities differ between cases")
    spans = [(1 + i * args.frames // len(cases), (i + 1) * args.frames // len(cases)) for i in range(len(cases))]
    if any(end - start + 1 < 24 for start, end in spans):
        raise ValueError("Cinematic layout requires at least 24 frames per recorded case")

    ink = material("Lettering", WHITE, 1.)
    dim = material("Neutral structure", STEEL, .45)
    cyan = material("Problem encoding label", CYAN, 1.2)
    purple = material("Continuation label", VIOLET, 1.2)
    amber = material("Projection label", AMBER, 1.1)
    black = material("Opaque synthesis", (.006, .009, .016, 1), surface=True)
    ground = material("Obsidian floor", (.004, .008, .016, 1), surface=True)
    cube("Ground", (0, 0, -3.6), (60, 24, .15), ground)
    for x in range(-16, 17, 2):
        line("Architectural floor", [(x, -6, -3.5), (x, 6, -3.5)], .004, dim)

    cam = bpy.data.objects.new("Hero camera", bpy.data.cameras.new("Hero camera"))
    scene.collection.objects.link(cam)
    cam.location = (8.0, -35.0, 20.0)
    cam.rotation_euler = (Vector((0, 0, .4)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 31.5
    scene.camera = cam
    for location, energy, color, size in [((0, -6, 14), 1800, (.44, .64, 1.), 14),
                                         ((-9, 5, 6), 1400, (.20, .45, 1.), 8),
                                         ((8, 5, 8), 1500, (.48, .25, 1.), 8)]:
        light = bpy.data.lights.new("Architectural rim", "AREA")
        light.energy, light.color, light.shape, light.size = energy, color, "DISK", size
        obj = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 0)) - obj.location).to_track_quat("-Z", "Y").to_euler()

    def label(body, position, size=.20, mat=ink):
        obj = text(body, position, size, mat)
        obj.rotation_euler = cam.rotation_euler
        return obj

    def hud(body, xy, size=.20, mat=ink):
        obj = text(body, (0, 0, 0), size, mat)
        obj.parent = cam
        obj.location = (*xy, -25.)
        obj.rotation_euler = (0, 0, 0)
        return obj

    def hud_tile(name, xy, dimensions, mat):
        obj = cube(name, (0, 0, 0), (*dimensions, .025), mat)
        obj.parent = cam
        obj.location = (*xy, -25.08)
        obj.rotation_euler = (0, 0, 0)
        return obj

    def visible_case(obj, ci, onset=None):
        start, end = spans[ci]
        onset = start if onset is None else onset
        # Explicit first/last boundaries make out-of-order rendering equivalent.
        for frame, state in sorted({1: False, onset: True, end + 1: False}.items()):
            obj.hide_render = not state
            obj.keyframe_insert("hide_render", frame=frame)
        interpolation(obj, "CONSTANT")

    hud("JAIN", (-14.1, 7.15), .88)
    hud("D A T A   →   W O R K I N G   C O D E", (-10.4, 7.43), .23, dim)
    hud("RECORDED INFERENCE · TIME EXPANDED", (5.9, 7.40), .21, amber)
    footer = material("Readable evidence footer", (.35, .44, .56, 1), 1)
    hud("Recorded native connections · intervening operations omitted · separate signed-engine source capture, same input/FIT", (-14.1, -7.45), .21, footer)

    # One shared geometric primitive, with an independent exact scalar material
    # per object. There are no data images or implicit RGB-to-float conversions.
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1)
    seed = bpy.context.object
    mesh = seed.data
    mesh.materials.append(ink)
    bpy.data.objects.remove(seed, do_unlink=True)

    def bead(name, location, radius, mat):
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.location, obj.scale = location, (radius,) * 3
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mat
        return obj

    def grid(n, center, plane_x):
        columns = math.ceil(math.sqrt(n))
        rows = math.ceil(n / columns)
        spacing = .34 if n >= 100 else .45
        return [Vector((plane_x, center[1] + (i % columns - (columns - 1) / 2) * spacing,
                        center[2] + (i // columns - (rows - 1) / 2) * spacing)) for i in range(n)]

    # Centers form one continuous architectural composition. Only within-tap
    # lines are weights. Neutral rails between regions are editorial layout.
    centers = [(-7.4, .1, .4), (-3.7, .6, .65), (2.9, .3, .2), (5.6, .5, .6)]
    phases = [(0., .14), (.15, .30), (.34, .43), (.44, .53)]
    receipts = []
    coverage = []
    node_count = 0
    for li, (first, center) in enumerate(zip(selected[0], centers)):
        color = CYAN if li < 2 else VIOLET
        origin = "dope" if li < 2 else "jope"
        arrays = [affine_arrays(layers[li]) for layers in selected]
        x, w, bias, y = arrays[0]
        for _, ww, _, _ in arrays:
            if not np.array_equal(ww, w):
                raise ValueError("Observed weights changed between captured cases")
        positions = [grid(w.shape[1], center, center[0] - 1.35), grid(w.shape[0], center, center[0] + 1.35)]
        # Cover every displayed nonzero-weight coordinate with a real edge.
        # Extra strongest edges add detail without orphaning the other neurons.
        active_in = set(map(int, np.flatnonzero(np.any(w != 0, axis=0))))
        active_out = set(map(int, np.flatnonzero(np.any(w != 0, axis=1))))
        edge_set = {(int(np.argmax(np.abs(w[:, src]))), src) for src in active_in}
        edge_set.update((dst, int(np.argmax(np.abs(w[dst])))) for dst in active_out)
        for flat in np.argsort(-np.abs(w).reshape(-1), kind="stable")[:min(args.edges, w.size)]:
            dst, src = map(int, np.unravel_index(flat, w.shape))
            if w[dst, src] != 0:
                edge_set.add((dst, src))
        picked = sorted(edge_set)
        connected_in = {src for _, src in picked}
        connected_out = {dst for dst, _ in picked}
        if connected_in != active_in or connected_out != active_out:
            raise ValueError("Displayed native neuron lacks a nonzero recorded weight connection")
        coverage.append({"network": first["network"], "name": first["name"], "sequence": first["sequence"],
                         "total_input_nodes": int(w.shape[1]), "total_output_nodes": int(w.shape[0]),
                         "displayed_input_nodes": len(active_in), "displayed_output_nodes": len(active_out),
                         "connected_in": len(connected_in), "connected_out": len(connected_out),
                         "selected_edge_count": len(picked), "inactive_inputs": sorted(set(range(w.shape[1])) - active_in),
                         "inactive_outputs": sorted(set(range(w.shape[0])) - active_out), "all_displayed_nodes_connected": True})
        scale = max((abs(float(xx[0, src] * ww[dst, src])) for xx, ww, _, _ in arrays for dst, src in picked), default=0.) or 1.
        for side, vectors in enumerate(([a[0][0] for a in arrays], [a[3][0] for a in arrays])):
            normalizer = max(float(np.max(np.abs(v))) for v in vectors) or 1.
            for index, position in enumerate(positions[side]):
                if index not in (active_in if side == 0 else active_out):
                    continue
                mat = material(f"Observed node {li}/{side}/{index}", color, 0)
                obj = bead(f"Native {origin}/{first['name']}/{side}/{index}", position, .045, mat)
                socket = mat.node_tree.nodes["Emission"].inputs["Strength"]
                for ci, values in enumerate(vectors):
                    start, end = spans[ci]
                    length = end - start + 1
                    onset = start + int(phases[li][side] * (length - 1))
                    keys = {start: 0., max(start, onset - 1): 0., onset: 2.8 * abs(float(values[index])) / normalizer, end: 2.8 * abs(float(values[index])) / normalizer}
                    for frame, value in sorted(keys.items()):
                        socket.default_value = value
                        socket.keyframe_insert("default_value", frame=frame)
                interpolation(mat.node_tree, "CONSTANT")
                obj["observed_index"], obj["operand_side"] = index, "input" if side == 0 else "output"
                node_count += 1
        for ei, (dst, src) in enumerate(picked):
            begin, end_position = positions[0][src], positions[1][dst]
            points = [begin.lerp(end_position, t / 10) + Vector((0, 0, math.sin(math.pi * t / 10) * .16)) for t in range(11)]
            width = .002 + .007 * abs(float(w[dst, src])) / (float(np.max(np.abs(w))) or 1.)
            # Neutral geometry is fixed and remains visible even when x*w=0.
            structural_edge = line(f"Recorded weight structure {li}/{dst}/{src}", points, width, dim)
            structural_edge["weight_origin"] = origin
            edge_mat = material(f"Observed fixed contribution {li}/{ei}", color, 0)
            edge = line(f"Weight contribution {li}/{dst}/{src}", points, width * .74, edge_mat)
            edge["weight_origin"] = origin
            edge_socket = edge_mat.node_tree.nodes["Emission"].inputs["Strength"]
            mat = material(f"Observed term {li}/{ei}", color, 0)
            pulse = bead(f"Contribution {li}/{dst}/{src}", begin, .035, mat)
            socket = mat.node_tree.nodes["Emission"].inputs["Strength"]
            terms = []
            for ci, (xx, ww, _, _) in enumerate(arrays):
                term = float(xx[0, src] * ww[dst, src])
                terms.append(term)
                first_frame, last_frame = spans[ci]
                length = last_frame - first_frame + 1
                op_start = first_frame + int(phases[li][0] * (length - 1))
                op_end = first_frame + int(phases[li][1] * (length - 1))
                start = min(op_start + ei % 2, op_end - 1)
                duration = max(1, op_end - start)
                strength = 3.6 * (abs(term) / scale) ** .6 if term != 0 else 0.
                edge_strength = .32 * (abs(term) / scale) ** .6 if term != 0 else 0.
                for frame, value in sorted({first_frame: 0., max(first_frame, op_start - 1): 0.,
                                            op_start: edge_strength, last_frame: edge_strength}.items()):
                    edge_socket.default_value = value
                    edge_socket.keyframe_insert("default_value", frame=frame)
                keys = {first_frame: 0., max(first_frame, start - 1): 0., start: strength,
                        op_end: strength, min(last_frame, op_end + 1): 0., last_frame: 0.}
                for frame, value in sorted(keys.items()):
                    socket.default_value = value
                    socket.keyframe_insert("default_value", frame=frame)
                pulse.location = begin
                pulse.keyframe_insert("location", frame=first_frame)
                for t, position in enumerate(points):
                    pulse.location = position
                    pulse.keyframe_insert("location", frame=start + duration * t / 10)
                pulse.location = end_position
                pulse.keyframe_insert("location", frame=last_frame)
            interpolation(pulse, "LINEAR")
            interpolation(mat.node_tree, "CONSTANT")
            interpolation(edge_mat.node_tree, "CONSTANT")
            receipts.append({"network": first["network"], "name": first["name"], "sequence": first["sequence"],
                             "batch_row": 0, "out": dst, "in": src, "weight": float(w[dst, src]),
                             "case_terms": terms, "normalizer": scale, "gamma": .6})

    label("Problem encoding", (-9.2, -.8, 4.5), .38, cyan)
    label("Neural inference", (1.7, -.8, 4.2), .38, purple)
    label("ENCODE", (-9.2, -.8, -3.0), .23, cyan)
    label("INFER", (2.0, -.8, -2.8), .23, purple)

    # Token vectors are aggregate prisms, not disconnected neuron glyphs.
    # Their light is the measured mean magnitude, not a fabricated edge.
    bridge_values = [np.asarray(case["outputs"]["tokens"], dtype=float).reshape(10, 32) for case in cases]
    bridge_scale = max(float(np.max(np.abs(v))) for v in bridge_values) or 1.
    for slot in range(10):
        mat = material(f"Bridge token aggregate {slot}", AMBER, 0)
        obj = cube(f"Stage1B token magnitude aggregate {slot}", (.3 + (slot % 2) * .17, (slot // 2 - 2) * .34, .5), (.12, .22, 3.4), mat)
        socket = mat.node_tree.nodes["Emission"].inputs["Strength"]
        for ci, values in enumerate(bridge_values):
            start, end = spans[ci]
            onset = start + int((end - start) * .32)
            value = 2.6 * float(np.mean(np.abs(values[slot]))) / bridge_scale
            for frame, strength in sorted({start: 0., onset - 1: 0., onset: value, end: value}.items()):
                socket.default_value = strength
                socket.keyframe_insert("default_value", frame=frame)
        interpolation(mat.node_tree, "CONSTANT")
    label("CONDENSE", (-.65, -.9, -2.8), .20, amber)

    # Excel-style preview uses the actual recorded values, rounded for display.
    slab = material("Table slab", (.012, .024, .042, 1), surface=True)
    cube("Input table", (-12.3, 0, -.7), (3.1, 3.1, .15), slab, .05)
    excel_green = material("Spreadsheet green", (.018, .23, .10, 1), 1)
    excel_white = material("Spreadsheet cells", (.78, .85, .87, 1), 1)
    excel_alt = material("Spreadsheet alternate rows", (.66, .76, .79, 1), 1)
    excel_dark = material("Spreadsheet values", (.012, .025, .035, 1), 1)
    hud("TABULAR DATA", (-14.5, 4.4), .31)
    for ci, case in enumerate(cases):
        objects = [hud(f"{case['id'].upper()} · {case['rows']} FIT ROWS", (-14.5, 3.9), .22, amber)]
        for col, header in enumerate(("x0", "x1", "x2", "x3", "y")):
            left = -14.5 + col * .96
            objects.append(hud_tile(f"Spreadsheet header {ci}/{col}", (left + .47, 3.30), (.94, .48), excel_green))
            objects.append(hud(header, (left + .26, 3.20), .24, ink))
        preview = case.get("table_preview", [])[:5]
        for row, values in enumerate(preview):
            for col, value in enumerate(values[:5]):
                mat = material(f"Real table cell {ci}/{row}/{col}", WHITE if col == 4 else CYAN, .45 + .5 * min(1, abs(float(value))))
                obj = cube(f"Cell {ci}/{row}/{col}", (-13.5 + col * .56, .95 - row * .45, -.4), (.48, .36, .07 + .10 * min(1, abs(float(value)))), mat)
                objects.append(obj)
                left, top = -14.5 + col * .96, 2.80 - row * .50
                objects.append(hud_tile(f"Spreadsheet value cell {ci}/{row}/{col}", (left + .47, top), (.94, .48), excel_white if row % 2 == 0 else excel_alt))
                objects.append(hud(f"{float(value): .2f}", (left + .08, top - .09), .23, excel_dark))
        for obj in objects:
            visible_case(obj, ci)

    # No search topology is drawn inside this opaque boundary.
    cube("Opaque synthesis boundary", (8.1, .3, .2), (.55, 2.0, 3.5), black, .10)
    label("SYNTHESIS", (7.3, -.8, -2.7), .18, dim)
    evidence = engine_evidence(args.engine_proof, data, args.emit_metrics) if getattr(args, "engine_proof", None) else []
    rust_keyword = material("Rust keyword", (.72, .44, 1., 1), 1)
    rust_number = material("Rust numeric literal", (1., .65, .27, 1), 1)
    rust_string = material("Rust string literal", (.37, .83, .63, 1), 1)
    rust_type = material("Rust type", (.25, .79, 1., 1), 1)
    code_font = bpy.data.fonts.load("/System/Library/Fonts/Menlo.ttc") if Path("/System/Library/Fonts/Menlo.ttc").exists() else None
    source_reveals = []
    tree_receipts = []
    show_gp_tree = bool(getattr(args, "show_gp_tree", False))
    if show_gp_tree and not evidence:
        raise ValueError("GP tree debug requires verified signed-engine program evidence")

    def debug_tree(proof, ci, onset):
        """Render the verified S-expression hierarchy, without inventing a chain."""
        ast = proof.get("gp_tree")
        program_sha = proof.get("gp_program_sha256", "")
        if not isinstance(ast, list) or not ast or not re.fullmatch(r"[0-9a-f]{64}", program_sha):
            raise ValueError("GP tree debug requires a verified AST and program SHA-256")
        nodes, links = [], []
        leaf_count = 0

        def collect(value, path=(), depth=0):
            nonlocal leaf_count
            if depth > 8 or len(nodes) >= 128:
                raise ValueError("GP tree exceeds the bounded debug diagram")
            operator = isinstance(value, list)
            if operator:
                if not value or not isinstance(value[0], str):
                    raise ValueError("GP operator must be a nonempty list headed by its actual name")
                name, children = value[0], value[1:]
            else:
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("Unsupported GP parameter atom")
                name, children = str(value), []
            node = {"path": list(path), "label": name, "operator": operator, "depth": depth}
            nodes.append(node)
            if children:
                child_nodes = []
                for index, child in enumerate(children, 1):
                    child_node = collect(child, (*path, index), depth + 1)
                    links.append((node, child_node))
                    child_nodes.append(child_node)
                node["leaf_x"] = (child_nodes[0]["leaf_x"] + child_nodes[-1]["leaf_x"]) / 2
            else:
                node["leaf_x"] = leaf_count
                leaf_count += 1
            return node

        collect(ast)
        depth = max(node["depth"] for node in nodes)
        left, right = -6.45, 5.35
        step_y = 1.78 / max(1, depth)
        for node in nodes:
            node["x"] = (left + right) / 2 if leaf_count == 1 else left + node["leaf_x"] * (right - left) / (leaf_count - 1)
            node["y"] = -4.28 - node["depth"] * step_y
        objects = [hud("GP TREE · DEBUG", (-6.8, -3.68), .25, amber),
                   hud(f"PROGRAM {program_sha[:16]}", (1.9, -3.67), .17, footer)]
        for parent, child in links:
            obj = line("GP AST parent-child", [(parent["x"], parent["y"] - .17, 0),
                                                (child["x"], child["y"] + .17, 0)], .009, footer)
            obj.parent = cam
            obj.location = (0, 0, -25.04)
            obj.rotation_euler = (0, 0, 0)
            objects.append(obj)
        for node in nodes:
            size = .24 if node["depth"] == 0 else .205 if node["operator"] else .18
            if node["operator"]:
                width = max(.72, len(node["label"]) * size * .62 + .25)
                objects.append(hud_tile("Actual GP operator", (node["x"], node["y"] + .055), (width, .37), black))
            obj = hud(node["label"], (node["x"], node["y"]), size,
                      ink if node["operator"] else amber if "=" in node["label"] else footer)
            obj.data.align_x = "CENTER"
            if code_font:
                obj.data.font = code_font
            objects.append(obj)
        for obj in objects:
            visible_case(obj, ci, onset)
        tree_receipts.append({"case_id": cases[ci]["id"], "program_sha256": program_sha,
                              "source_sha256": proof["source_sha256"], "ast": ast,
                              "reveal_start": onset, "hide_at": spans[ci][1] + 1,
                              "nodes": nodes, "edges": [{"parent": parent["path"], "child": child["path"]} for parent, child in links],
                              "scope": "owner-requested internal GP AST debug; source program from this verified fitted model"})

    for ci, case in enumerate(cases):
        start, finish = spans[ci]
        onset = start + int((finish - start) * .55)
        complete = start + int((finish - start) * .88)
        items = [hud("RUST SOURCE", (7.3, 4.4), .34), hud("EXCERPT · READABLE REPLAY", (7.3, 3.92), .18, amber)]
        if evidence:
            proof = evidence[ci]
            if show_gp_tree:
                debug_tree(proof, ci, onset)
            # Exact original emitted lines, never replacement model code.
            source = "\n".join(proof.get("source_excerpt_lines", proof["source_lines"]))
            tokens = re.findall(r'//[^\n]*|"(?:\\.|[^"\\])*"|\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b|\b[A-Za-z_]\w*\b|\s+|.', source)
            words = [token for token in tokens if not token.isspace()]
            column = row = serial = 0
            for token in tokens:
                if token.isspace():
                    for char in token:
                        if char == "\n":
                            row, column = row + 1, 0
                        else:
                            column += 1
                    continue
                mat = rust_keyword if token in {"pub", "fn", "let", "mut", "if", "for", "in", "return", "const", "match", "else"} else rust_number if token[0].isdigit() else rust_string if token.startswith('"') else rust_type if token in {"f64", "f32", "Vec", "Result", "String", "Ok", "Err", "Some", "None"} else ink
                reveal = onset + (complete - onset) * serial // max(1, len(words) - 1)
                serial += 1
                # Wrapping changes layout only. Concatenated token text remains
                # exactly the selected original source, with no invented code.
                if column and column + len(token) > 48:
                    row, column = row + 1, 0
                for offset in range(0, len(token), 48):
                    part = token[offset:offset + 48]
                    if column + len(part) > 48:
                        row, column = row + 1, 0
                    obj = hud(part, (7.3 + column * .140, 3.35 - row * .30), .235, mat)
                    if code_font:
                        obj.data.font = code_font
                    visible_case(obj, ci, reveal)
                    column += len(part)
                    if offset + 48 < len(token):
                        row, column = row + 1, 0
            source_reveals.append({"case_id": case["id"], "source_sha256": proof["source_sha256"],
                                   "excerpt_text": source, "reveal_start": onset, "reveal_complete": complete,
                                   "reveal_tokens": len(words), "display_wrap_columns": 48, "layout_only_wrapping": True})
            badge = hud("Compiled · executed · checked", (7.3, -3.8), .22, cyan)
            visible_case(badge, ci, complete)
            metric = proof.get("emission_metric")
            if metric:
                rate = metric["value"]
                display = f"{rate / 1e6:.2f} M" if rate >= 1e6 else f"{rate / 1e3:.1f} k"
                rate_obj = hud(f"{display} Rust lexical tokens/s", (7.3, -4.5), .28, amber)
                visible_case(rate_obj, ci, complete)
                rate_scope = hud("Warm repeated emission · same artifact\nDev build · loop overhead included", (7.3, -5.05), .18, footer)
                visible_case(rate_scope, ci, complete)
            else:
                items.append(hud("Emission timing pending", (7.3, -4.5), .20, footer))
        else:
            items.append(hud("Verified source capture pending", (7.3, 3.2), .20, footer))
        for obj in items:
            visible_case(obj, ci, onset)

    # Broad architectural arrows express stage order, never learned edges.
    for left, right in [(-10.7, -9.1), (-2.1, -.5), (1.1, 1.5), (7.1, 7.8), (8.5, 10.6)]:
        line("Editorial stage rail", [(left, 0, -.1), (right, 0, -.1)], .025, footer)
        line("Editorial arrow head", [(right - .24, -.16, -.1), (right, 0, -.1), (right - .24, .16, -.1)], .025, footer)
    try:
        comp = bpy.data.node_groups.new(" restrained optical glow", "CompositorNodeTree")
        comp.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        render = comp.nodes.new("CompositorNodeRLayers")
        glare = comp.nodes.new("CompositorNodeGlare")
        glare.inputs["Type"].default_value = "Bloom"
        glare.inputs["Threshold"].default_value = 1.3
        glare.inputs["Strength"].default_value = .14
        out = comp.nodes.new("NodeGroupOutput")
        comp.links.new(render.outputs["Image"], glare.inputs["Image"])
        comp.links.new(glare.outputs["Image"], out.inputs[0])
        scene.compositing_node_group = comp
    except (KeyError, TypeError, AttributeError):
        pass

    receipt = {"schema": "jain.cinematic-review/v1", "trace_sha256": validation["trace_sha256"],
               "bundle_digest": data["bundle_digest"], "blender_version": bpy.app.version_string,
               "blender_build_hash": bpy.app.build_hash.decode(), "case_frames": spans,
               "observed_affine_nodes": node_count, "selected_edges": receipts, "layer_coverage": coverage,
               "selection": "union of strongest outgoing edge per active input, strongest incoming edge per active output, and top requested absolute weights; first batch row; all-zero rows/columns hidden",
               "bridge": "ten aggregate token prisms; brightness is mean absolute magnitude of each actual 32-value Stage1B token; no neuron or learned-edge claim",
               "source_reveals": source_reveals, "table_display": "five actual recorded rows, rounded to two decimal places; last target column aliased y",
               "source_evidence": evidence if show_gp_tree else [{key: value for key, value in item.items() if not key.startswith("gp_")} for item in evidence],
               "gp_tree_debug": {"enabled": show_gp_tree, "cases": tree_receipts},
               "scope": "private diagnostic; explanatory layout/reveal; separate signed-engine source capture",
               "omitted_transitions": "neutral stage rails; no invented learned connections or decoder",
               "pulse_brightness": "3.6 * (abs(input * weight) / fixed_selected_run_max)^0.6; zero maps to zero",
               "node_brightness": "2.8 * abs(observed value) / fixed_run_max; zero maps to zero"}
    scene["evidence_scope"], scene["trace_sha256"] = receipt["scope"], validation["trace_sha256"]
    bpy.data.texts.new("PRIVATE_BINDING_RECEIPT.json").write(json.dumps(receipt, indent=2))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    scene.frame_set(args.render_frame[0] if args.render_frame else 28)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    for frame in args.render_frame:
        scene.frame_set(frame)
        scene.render.filepath = str(output.parent / f"review-{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    if args.render_animation:
        scene.render.filepath = str(output.parent / "frames/frame-")
        bpy.ops.render.render(animation=True)
    print(json.dumps({"blend": str(output), "nodes": node_count, "edges": len(receipts), "stills": args.render_frame}))
