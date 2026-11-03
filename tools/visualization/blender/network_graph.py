"""Compact native affine microscopes in the actual fused dependency graph.

Weighted tubes exist only inside an observed affine operation. Thin neutral
arrows denote the documented functional dependencies, never trained weights.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np

POSITIVE = (.035, .72, .90, 1.)
NEGATIVE = (1., .31, .055, 1.)
WHITE = (.69, .78, .87, 1.)
QUIET = (.055, .075, .10, 1.)

DOPE = "dope_model.router."
DOPE_LAYOUT = {
    "backbone.0": (-6.75, 3.05, "140 → 256"),
    "backbone.3": (-4.65, 3.05, "256 → 256"),
    "inter_attn.token_proj": (-6.75, 1.60, "32 → 32 × 8"),
    "inter_attn.qkv": (-5.15, 1.60, "Q / K / V"),
    "inter_attn.bilinear_u": (-6.75, .22, "tanh branch"),
    "inter_attn.bilinear_v": (-5.15, .22, "SiLU branch"),
    "inter_attn.out_proj": (-2.55, 1.60, "384 → 256"),
    "gate_head": (-.10, 3.35, "10 gates"),
    "v3_gate_head": (1.55, 3.35, "4 gates"),
    "task_latent_head": (-.10, 2.40, "task · 128"),
    "dist_latent_head": (1.55, 2.40, "distribution · 128"),
    "landmark_recon_head": (-.10, 1.45, "landmarks · 23"),
    "bb_recon_head": (1.55, 1.45, "blocks · 16"),
}

ANCHOR = [
    "anchor.ctx_in.1", "anchor.ctx_block.fc_gate", "anchor.ctx_block.fc_val", "anchor.ctx_block.fc_out",
    "anchor.landmark_router.1", "anchor.landmark_router.3",
    "anchor.prog_in.1", "anchor.prog_block.fc_gate", "anchor.prog_block.fc_val", "anchor.prog_block.fc_out",
    "anchor.prior_head.1", "anchor.prior_head.3", "anchor.history_token_in.0",
    "anchor.cross_attn.out_proj", "anchor.traj_out.1", "anchor.traj_out.3",
    "anchor.gate_head.1", "anchor.gate_head.3",
]
JOPE_GROUPS = [
    ("Anchor", -6.45, 3.0, ANCHOR),
    ("Candidate query", -3.72, 1.85, ["prog_412_bottleneck.0", "cand_query_proj.0"]),
    ("Context interaction", -1.35, 2.45, ["mm_cross_attn.out_proj", "prog_bilinear_v", "ctx_bilinear_u", "mm_prior_out.0", "mm_prior_out.2"]),
    ("Trajectory", 1.35, 2.45, ["hist_token_proj.0", "traj_cross_attn.out_proj", "traj_delta_head.0", "traj_delta_head.2"]),
]


def _flat(record, key):
    return np.asarray(record[key], dtype=np.float64)


def _arrays(record):
    w = _flat(record, "weights").reshape(record["weight_shape"])
    x = _flat(record, "input").reshape(-1, w.shape[1])
    y = _flat(record, "output").reshape(-1, w.shape[0])
    if not all(np.isfinite(v).all() for v in (x, w, y)):
        raise ValueError("Nonfinite native affine observation")
    return x, w, y


def _calibrate(values):
    values = np.asarray(values, dtype=np.float64)
    nonzero = np.abs(values[values != 0])
    if not nonzero.size:
        return np.zeros(values.shape, dtype=np.float32), {"scale": 1., "bound": 0., "all_zero": True}
    scale, bound = float(np.median(nonzero)), float(np.percentile(nonzero, 99))
    result = np.sign(values) * np.minimum(1., np.log1p(np.abs(values) / scale) / math.log1p(bound / scale))
    return result.astype(np.float32), {"scale": scale, "bound": bound, "all_zero": False,
        "transform": "sign(v)*min(1,log1p(abs(v)/scale)/log1p(bound/scale)); fixed across all cases"}


def _atlas(name, values):
    height, width = values.shape
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[..., 0], pixels[..., 1], pixels[..., 3] = values, 1., 1.
    image = bpy.data.images.new(name, width=width, height=height, alpha=True,
                                float_buffer=True, is_data=True)
    image.colorspace_settings.is_data = True
    image.file_format = "OPEN_EXR"
    image.pixels.foreach_set(pixels.ravel())
    readback = np.empty(pixels.size, dtype=np.float32)
    image.pixels.foreach_get(readback)
    if not np.array_equal(readback, pixels.ravel()):
        raise ValueError("FP32 numeric atlas binding changed values")
    image.pack()
    return image, hashlib.sha256(pixels.tobytes()).hexdigest()


def _material(name, image, spans, edge=False):
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    tree.nodes.clear()
    new, link = tree.nodes.new, tree.links.new
    uv = new("ShaderNodeTexCoord")
    separate = new("ShaderNodeSeparateXYZ")
    link(uv.outputs["UV"], separate.inputs[0])
    coordinates = new("ShaderNodeCombineXYZ")
    link(separate.outputs["X"], coordinates.inputs["X"])
    row = new("ShaderNodeValue")
    row.label = "Exact captured case; constant interpolation"
    for i, (start, _) in enumerate(spans):
        row.outputs[0].default_value = (i + .5) / len(spans)
        row.outputs[0].keyframe_insert("default_value", frame=start)
    _constant(tree)
    link(row.outputs[0], coordinates.inputs["Y"])
    texture = new("ShaderNodeTexImage")
    texture.image, texture.interpolation, texture.extension = image, "Closest", "EXTEND"
    link(coordinates.outputs[0], texture.inputs["Vector"])
    red = new("ShaderNodeSeparateColor")
    red.mode = "RGB"
    link(texture.outputs["Color"], red.inputs["Color"])
    magnitude = new("ShaderNodeMath"); magnitude.operation = "ABSOLUTE"
    link(red.outputs["Red"], magnitude.inputs[0])
    positive = new("ShaderNodeMath"); positive.operation = "GREATER_THAN"
    positive.inputs[1].default_value = 0.
    link(red.outputs["Red"], positive.inputs[0])
    colors = new("ShaderNodeMixRGB")
    colors.inputs[1].default_value, colors.inputs[2].default_value = NEGATIVE, POSITIVE
    link(positive.outputs[0], colors.inputs[0])
    gain = new("ShaderNodeMath"); gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 2.4 if edge else 2.8
    link(magnitude.outputs[0], gain.inputs[0])
    strength = gain.outputs[0]
    if edge:
        phase = new("ShaderNodeValue")
        phase.label = "Editorial travel on the actual weighted connection"
        for start, end in spans:
            phase.outputs[0].default_value = 0.
            phase.outputs[0].keyframe_insert("default_value", frame=start)
            phase.outputs[0].default_value = 1.
            phase.outputs[0].keyframe_insert("default_value", frame=min(end, start+24))
        path = phase.outputs[0].path_from_id("default_value")
        action = tree.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(tree.animation_data.action_slot)
                if bag:
                    for curve in bag.fcurves:
                        if curve.data_path == path:
                            for point in curve.keyframe_points:
                                point.interpolation = "LINEAR"
        difference = new("ShaderNodeMath"); difference.operation = "SUBTRACT"
        link(separate.outputs["Y"],difference.inputs[0]);link(phase.outputs[0],difference.inputs[1])
        distance = new("ShaderNodeMath");distance.operation = "ABSOLUTE"
        link(difference.outputs[0],distance.inputs[0])
        pulse = new("ShaderNodeMath");pulse.operation = "LESS_THAN";pulse.inputs[1].default_value=.12
        link(distance.outputs[0],pulse.inputs[0])
        wave = new("ShaderNodeMath");wave.operation="MULTIPLY_ADD"
        wave.inputs[1].default_value=2.8;wave.inputs[2].default_value=.28
        link(pulse.outputs[0],wave.inputs[0])
        lit = new("ShaderNodeMath");lit.operation="MULTIPLY"
        link(gain.outputs[0],lit.inputs[0]);link(wave.outputs[0],lit.inputs[1])
        strength=lit.outputs[0]
    emission = new("ShaderNodeEmission")
    link(colors.outputs[0], emission.inputs["Color"])
    link(strength, emission.inputs["Strength"])
    base = new("ShaderNodeEmission")
    base.inputs["Color"].default_value = QUIET
    base.inputs["Strength"].default_value = .06 if edge else .035
    add = new("ShaderNodeAddShader")
    link(base.outputs[0], add.inputs[0]); link(emission.outputs[0], add.inputs[1])
    output = new("ShaderNodeOutputMaterial")
    link(add.outputs[0], output.inputs["Surface"])
    return mat


def _constant(tree):
    ad = tree.animation_data
    if ad is None or ad.action is None:
        return
    for layer in ad.action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(ad.action_slot)
            if bag:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "CONSTANT"


ICO_VERTICES = [(0,-1,-1.618),(0,-1,1.618),(0,1,-1.618),(0,1,1.618),
                (-1,-1.618,0),(-1,1.618,0),(1,-1.618,0),(1,1.618,0),
                (-1.618,0,-1),(-1.618,0,1),(1.618,0,-1),(1.618,0,1)]
# Compute the twenty faces once from the equal shortest edge length.
ICO = np.asarray(ICO_VERTICES, dtype=float)
ICO /= np.linalg.norm(ICO, axis=1)[:, None]
ICO_FACES = [(a,b,c) for a in range(12) for b in range(a+1,12) for c in range(b+1,12)
             if max(np.linalg.norm(ICO[a]-ICO[b]), np.linalg.norm(ICO[a]-ICO[c]),
                    np.linalg.norm(ICO[b]-ICO[c])) < 1.06]


def _mesh(collection, name, vertices, faces, face_uv, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="Native scalar index")
    for polygon, coordinates in zip(mesh.polygons, face_uv):
        for loop, coordinate in zip(polygon.loop_indices, coordinates):
            uv.data[loop].uv = coordinate
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def _node_geometry(positions, radius):
    vertices, faces, uv = [], [], []
    for index, position in enumerate(positions):
        offset = len(vertices)
        vertices.extend(tuple(Vector(position) + Vector(point) * radius) for point in ICO)
        for face in ICO_FACES:
            faces.append(tuple(offset + v for v in face))
            uv.append([((index + .5) / len(positions), .5)] * 3)
    return vertices, faces, uv


def _edge_geometry(edges):
    vertices, faces, uv = [], [], []
    for index, (begin, end, width) in enumerate(edges):
        begin, end = Vector(begin), Vector(end)
        direction = (end - begin).normalized()
        u = direction.cross(Vector((0, 1, 0))).normalized()
        v = direction.cross(u).normalized()
        offset = len(vertices)
        for step in range(7):
            center = begin.lerp(end, step / 6) + Vector((0, math.sin(math.pi * step / 6) * .045, 0))
            for side in range(4):
                vertices.append(tuple(center + width * (u * math.cos(side * math.pi / 2) + v * math.sin(side * math.pi / 2))))
        for step in range(6):
            for side in range(4):
                a, b = offset + step*4 + side, offset + step*4 + (side+1)%4
                faces.append((a,b,b+4,a+4))
                uv.append([((index+.5)/len(edges), step/6), ((index+.5)/len(edges), step/6),
                           ((index+.5)/len(edges), (step+1)/6), ((index+.5)/len(edges), (step+1)/6)])
    return vertices, faces, uv


def _select(w, channels, edge_limit):
    picked, inputs, outputs = [], set(), set()
    for flat in np.argsort(-np.abs(w).ravel(), kind="stable"):
        out, source = map(int, np.unravel_index(flat, w.shape))
        if w[out,source] == 0:
            continue
        if len(inputs | {source}) > channels or len(outputs | {out}) > channels:
            continue
        inputs.add(source); outputs.add(out); picked.append((out,source))
        if len(picked) >= edge_limit:
            break
    return picked, sorted(inputs), sorted(outputs)


def build_network(data, spans, camera, bounds=None):
    """Create a static branch graph. Case values change; geometry never changes.

    bounds optionally supplies (xmin,xmax,zmin,zmax), scaling the default region.
    The caller owns its scene, camera, lights, captions, table and source panels.
    """
    if len(spans) != len(data["cases"]):
        raise ValueError("One frame interval per captured case is required")
    region = (-8., 3., -2., 4.) if bounds is None else tuple(bounds)
    if len(region) != 4 or region[1] <= region[0] or region[3] <= region[2]:
        raise ValueError("bounds must be xmin,xmax,zmin,zmax")
    collection = bpy.data.collections.new("JAIN native branch graph")
    bpy.context.scene.collection.children.link(collection)
    objects, receipts, dependencies = [], [], []
    def pos(x, z, y=0.):
        return (region[0]+(x+8)/11*(region[1]-region[0]), y,
                region[2]+(z+2)/6*(region[3]-region[2]))
    plain = bpy.data.materials.new("JAIN neutral labels")
    tree = plain.node_tree; tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission"); emission.inputs["Color"].default_value = WHITE
    emission.inputs["Strength"].default_value = .75
    output = tree.nodes.new("ShaderNodeOutputMaterial"); tree.links.new(emission.outputs[0], output.inputs[0])
    route = bpy.data.materials.new("JAIN quiet functional routes")
    route_tree = route.node_tree; route_tree.nodes.clear()
    route_emission = route_tree.nodes.new("ShaderNodeEmission")
    route_emission.inputs["Color"].default_value = QUIET
    route_emission.inputs["Strength"].default_value = .22
    route_output = route_tree.nodes.new("ShaderNodeOutputMaterial")
    route_tree.links.new(route_emission.outputs[0], route_output.inputs[0])
    def label(body, x, z, size=.13):
        curve = bpy.data.curves.new("Native stage label", "FONT"); curve.body, curve.size = body, size
        obj = bpy.data.objects.new("Stage · " + body, curve); collection.objects.link(obj)
        obj.location = pos(x,z,-.22); obj.rotation_euler = camera.rotation_euler
        curve.materials.append(plain); objects.append(obj)
        return obj
    def rail(source, target, caption=None):
        # These are structural routes, not connections backed by a learned matrix.
        curve = bpy.data.curves.new("Unweighted functional dependency", "CURVE")
        curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", .003, 0
        spline = curve.splines.new("POLY"); spline.points.add(2)
        a,b = Vector(pos(*source,.16)),Vector(pos(*target,.16))
        for point,value in zip(spline.points,(a,a.lerp(b,.5),b)):
            point.co=(*value,1)
        obj=bpy.data.objects.new("Unweighted functional dependency",curve); collection.objects.link(obj)
        curve.materials.append(route);objects.append(obj)
        dependencies.append({"from":source,"to":target,"role":"unweighted documented dependency","caption":caption})
        if caption:
            label(caption,(source[0]+target[0])/2,(source[1]+target[1])/2+.08,.085)
    by_case=[]
    for case in data["cases"]:
        found={}
        for op in sorted(case["trace"]["affine"],key=lambda op:op["sequence"]):
            found.setdefault((op["network"],op["name"]),op)
        by_case.append(found)
    layout={}
    for short,(x,z,title) in DOPE_LAYOUT.items():
        layout[("dope",DOPE+short)]=(x,z,title,6,18,1.10,.035)
    layout[("dope","dope_model.target_net.col_gate_head")]=(1.55,.50,"column gate · 145 → 1",6,12,1.10,.035)
    label("Neural inference",-7.8,-.58,.22)
    label("Selected real channels · signed activations / weighted terms",-7.8,-1.99,.105)
    for title,cx,width,names in JOPE_GROUPS:
        present=[name for name in names if ("jope",name) in by_case[0]]
        columns=5 if title=="Anchor" else 2
        row_count=math.ceil(len(present)/columns)
        label(f"{title} · {len(present)} affine ops",cx-width/2,-.89,.11)
        for index,name in enumerate(present):
            x=cx-width/2+(index%columns+.5)*width/columns
            z=-1.13-(index//columns)*min(.20,.60/max(1,row_count-1))
            layout[("jope",name)]=(x,z,None,3,7,min(.32,width/columns*.7),.020)
    for key,(cx,cz,title,channels,edge_limit,width,radius) in layout.items():
        if key not in by_case[0]:
            continue
        records=[case[key] for case in by_case]
        arrays=[_arrays(record) for record in records]
        w=arrays[0][1]
        if any(not np.array_equal(w,other[1]) for other in arrays):
            raise ValueError("Native weights changed between displayed cases")
        picked,inputs,outputs=_select(w,channels,edge_limit)
        if title:
            label(title,cx-width*.58,cz+.34,.12)
            label(f"{len(inputs)}/{w.shape[1]} in · {len(outputs)}/{w.shape[0]} out",cx-width*.58,cz-.42,.075)
        if not picked:
            label("zero W",cx-width*.35,cz,.07)
            receipts.append({"network":key[0],"name":key[1],"native_sequences":[r["sequence"] for r in records],
                "input_dimension":int(w.shape[1]),"output_dimension":int(w.shape[0]),
                "displayed_units":[],"selected_edges":[],"reason":"all weights exactly zero; no nonzero-edge neuron glyphs fabricated",
                "native_output_first_row":[y[0].tolist() for _,_,y in arrays],"weights_fixed_across_cases":True})
            continue
        spacing=.11 if key[0]=="dope" else .055
        left={n:pos(cx-width/2,cz+(i-(len(inputs)-1)/2)*spacing,-.06) for i,n in enumerate(inputs)}
        right={n:pos(cx+width/2,cz+(i-(len(outputs)-1)/2)*spacing,.06) for i,n in enumerate(outputs)}
        nodes=[("input",n) for n in inputs]+[("output",n) for n in outputs]
        positions=list(left.values())+list(right.values())
        raw_nodes=np.asarray([[float(x[0,n] if side=="input" else y[0,n]) for side,n in nodes] for x,_,y in arrays])
        node_values,node_scale=_calibrate(raw_nodes)
        node_image,node_sha=_atlas("Native nodes · "+key[1],node_values)
        node_mat=_material("Native nodes · "+key[1],node_image,spans)
        objects.append(_mesh(collection,"Units · "+key[1],*_node_geometry(positions,radius),node_mat))
        terms=np.asarray([[float(x[0,source]*ww[out,source]) for out,source in picked] for x,ww,_ in arrays])
        term_values,term_scale=_calibrate(terms)
        term_image,term_sha=_atlas("Native terms · "+key[1],term_values)
        term_mat=_material("Native terms · "+key[1],term_image,spans,True)
        maximum=float(np.max(np.abs(w))) or 1.
        edges=[(left[source],right[out],.0025+.004*abs(float(w[out,source]))/maximum) for out,source in picked]
        objects.append(_mesh(collection,"Weighted terms · "+key[1],*_edge_geometry(edges),term_mat))
        unit_records=[{"side":side,"channel":n,"batch_row":0,"values":[float(v) for v in raw_nodes[:,i]]} for i,(side,n) in enumerate(nodes)]
        edge_records=[{"input":source,"output":out,"weight":float(w[out,source]),"terms":[float(v) for v in terms[:,i]]} for i,(out,source) in enumerate(picked)]
        receipts.append({"network":key[0],"name":key[1],"native_sequences":[r["sequence"] for r in records],
            "input_dimension":int(w.shape[1]),"output_dimension":int(w.shape[0]),"batch_row":0,
            "displayed_units":unit_records,"selected_edges":edge_records,"all_units_have_real_nonzero_incident_weight":True,
            "output_scope":"full observed affine output; selected displayed terms do not by themselves sum to it",
            "node_calibration":node_scale,"term_calibration":term_scale,"node_atlas_f32_sha256":node_sha,
            "term_atlas_f32_sha256":term_sha,"weights_fixed_across_cases":True})
    # This surface is an observed probability matrix, not neuron glyphs or a
    # trained-weight layer. Its 64 cells are counted separately in the receipt.
    attention=[]
    for case in data["cases"]:
        taps=[t for t in case["trace"]["tensors"] if t["network"]=="dope" and t["name"]=="attention_probabilities"]
        if len(taps)!=1 or taps[0]["shape"]!=[8,8]:
            raise ValueError("The actual 8×8 attention probability readout is required")
        matrix=np.asarray(taps[0]["values"],dtype=np.float64).reshape(8,8)
        if not np.isfinite(matrix).all() or np.any(matrix<0) or not np.allclose(matrix.sum(axis=1),1,atol=1e-6,rtol=0):
            raise ValueError("Invalid native attention probabilities")
        attention.append(matrix.ravel())
    attention_values,attention_scale=_calibrate(np.asarray(attention)-1/8)
    attention_image,attention_sha=_atlas("Native attention probabilities",attention_values)
    attention_material=_material("Actual attention probabilities",attention_image,spans)
    positions=[pos(-3.83+(column-3.5)*.065,1.60+(row-3.5)*.065,-.10) for row in range(8) for column in range(8)]
    objects.append(_mesh(collection,"Observed probability surface",*_node_geometry(positions,.024),attention_material))
    label("Attention · A − 1/8",-4.33,2.03,.12)
    # Actual parallel branches and functional merges from FusedForward::stages.
    rail((-7.85,1.6),(-7.32,3.05),"LN140")
    rail((-7.85,1.6),(-7.32,1.6),"LN + reshape")
    rail((-7.85,1.6),(-7.32,.22),"x256")
    rail((-7.85,1.6),(-5.72,.22))
    rail((-6.20,3.05),(-5.20,3.05),"SiLU / LN")
    rail((-6.20,1.6),(-5.70,1.6),"SiLU")
    rail((-4.60,1.6),(-3.55,1.6))
    rail((-6.20,1.6),(-3.55,1.6),"token residual bypass")
    rail((-6.20,.22),(-3.55,.48),"×")
    rail((-4.60,.22),(-3.55,.48))
    rail((-3.55,.48),(-3.10,1.6),"concat")
    rail((-3.55,1.6),(-3.10,1.6))
    rail((-4.10,3.05),(-1.35,2.85),"level +")
    rail((-2.00,1.6),(-1.35,2.85),"LN / add")
    for cx,cz in [(-.10,3.35),(1.55,3.35),(-.10,2.40),(1.55,2.40),(-.10,1.45),(1.55,1.45)]:
        rail((-1.35,2.85),(cx-.55,cz))
    rail((.45,2.40),(1.,.50),"tanh + col17")
    label("Attention / residual / norm routes are neutral, unweighted",-3.0,.08,.085)
    # JOPE stages are grouped, not drawn as a false chain of four affine layers.
    rail((-3.72,-1.0),(-1.35,-1.0),"query → attention")
    rail((-1.35,-1.0),(1.35,-1.0),"context + history")
    rail((-6.45,-1.8),(1.35,-1.8),"anchor / kernel / residual combination")
    used={(r["network"],r["name"]) for r in receipts}
    omitted=[{"network":n,"name":name} for n,name in by_case[0] if (n,name) not in used]
    receipt={"schema":"jain.native-branch-network/v1","native_source_commit":data.get("source_commit"),
        "bundle_digest":data.get("bundle_digest"),"bounds":region,"operations":receipts,"functional_dependencies":dependencies,
        "unique_native_affines_available":len(by_case[0]),"unique_native_affines_displayed":len(receipts),
        "omitted_affines":omitted,"displayed_neurons":sum(len(r["displayed_units"]) for r in receipts),
        "displayed_weighted_connections":sum(len(r["selected_edges"]) for r in receipts),
        "attention_probability_cells":{"count":64,"role":"observed softmax probabilities, not neurons or learned weights",
            "display_transform":"probability minus uniform 1/8; signed fixed log magnitude",
            "native_shape":[8,8],"values":[v.tolist() for v in attention],"fixed_calibration":attention_scale,
            "atlas_f32_sha256":attention_sha,"row_sum_check_tolerance":1e-6},
        "selection":"fixed strongest absolute nonzero weights with bounded input/output channels; all cases share topology; first observed named invocation, first batch row",
        "scope":"private diagnostic selected channels; complete native affine values, partial channel/connection coverage; neutral functional topology is documented, not measured signal travel",
        "atlas":"FP32 Non-Color, Closest, SeparateColor.Red; signed fixed per-operation log scale across all cases",
        "zero_policy":"zero data emission; constant quiet base below 1 percent of maximum data emission",
        "pulse_motion":"linear editorial source-to-destination travel; measured term magnitude gates every pulse; not physical transit time",
        "branch_source_paths":["crates/jope-service/src/ranker/cuda/forward.rs::FusedForward::stages",
                               "crates/jope-service/src/ranker/adapter.rs::CpuAdapter::forward_row/anchor"],
        "geometry_changes_between_cases":False}
    return {"receipt":receipt,"objects":objects}
