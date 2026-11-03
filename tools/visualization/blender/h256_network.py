"""Recorded matrix connections for saved H256 Stage 1 and Stage 2 snapshots.

Geometry is editorial. Every colored connection is one actual W[out,in] or
bias entry. Neutral routes come only from the capture's typed dependencies.
Repeated token rows use their signed maximum-absolute observation, with the
winning row retained in the receipt. No missing query or readout is invented.
"""
from __future__ import annotations

import hashlib
import math
import re

import bpy
import numpy as np
from mathutils import Vector

from build_scene import interpolation, line, material, text
from network_graph import _calibrate, _node_geometry

CYAN = (.025, .70, .95, 1.)
VIOLET = (.47, .18, 1., 1.)
NEGATIVE = (1., .33, .075, 1.)
QUIET = (.035, .055, .09, 1.)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def arrays(record):
    # Round the direct serde decimal representation back to its native type.
    w = np.asarray(record["weights"], dtype=np.float32).astype(np.float64).reshape(record["weight_shape"])
    x = np.asarray(record["input"], dtype=np.float32).astype(np.float64).reshape(-1, w.shape[1])
    y = np.asarray(record["output"], dtype=np.float32).astype(np.float64).reshape(-1, w.shape[0])
    b = np.asarray(record["bias"], dtype=np.float32).astype(np.float64)
    if not all(np.isfinite(a).all() for a in (x, w, y, b)):
        raise ValueError("Nonfinite H256 matrix observation")
    return x, w, b, y


def representative(values):
    row = np.argmax(np.abs(values), axis=0)
    return values[row, np.arange(values.shape[1])], row


def output_value(case, name, fold=0):
    """Use a real fold readout; never replace missing Stage2 values by Stage1."""
    if name in case["outputs"]:
        return case["outputs"][name]
    return case["outputs"][f"fold{fold}.{name}"]


def fold_index(name):
    match=re.search(r"(?:^|\.)fold([0-4])\.",name)
    return int(match[1]) if match else None


def fold_position(position, fold):
    if fold is None:
        return tuple(position)
    x,y,z=position
    return ((x-1.5)*.78+1.35,y*.27+(fold-2)*1.45,z*.60)


def _curves(tree):
    ad = tree.animation_data
    if not ad or not ad.action:
        return []
    curves = []
    for layer in ad.action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(ad.action_slot)
            if bag:
                curves.extend(bag.fcurves)
    return curves


def scalar_material(name, values, spans, color, edge=False, delay=0, power_exponent=None):
    """A tiled FP32 atlas avoids the maximum texture width for 500k edges."""
    normalized, calibration = _calibrate(values)
    count = normalized.shape[1]
    width = min(4096, count)
    block = math.ceil(count / width)
    height = block * len(spans)
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[..., 3] = 1.
    for case in range(len(spans)):
        pixels[case * block:(case + 1) * block, :, 0].flat[:count] = normalized[case]
    image = bpy.data.images.new(name + " · FP32", width=width, height=height,
                                alpha=True, float_buffer=True, is_data=True)
    image.colorspace_settings.is_data = True
    image.file_format = "OPEN_EXR"
    image.pixels.foreach_set(pixels.ravel())
    check = np.empty(pixels.size, dtype=np.float32)
    image.pixels.foreach_get(check)
    if not np.array_equal(check, pixels.ravel()):
        raise ValueError("Native H256 scalar atlas changed values")
    image.pack()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    new, link = tree.nodes.new, tree.links.new
    uv = new("ShaderNodeUVMap"); uv.uv_map = "Scalar"
    split = new("ShaderNodeSeparateXYZ"); link(uv.outputs["UV"], split.inputs[0])
    row = new("ShaderNodeValue")
    for i, (first, _) in enumerate(spans):
        row.outputs[0].default_value = i * block / height
        row.outputs[0].keyframe_insert("default_value", frame=first)
    add = new("ShaderNodeMath"); add.operation = "ADD"
    link(split.outputs["Y"], add.inputs[0]); link(row.outputs[0], add.inputs[1])
    coord = new("ShaderNodeCombineXYZ")
    link(split.outputs["X"], coord.inputs["X"]); link(add.outputs[0], coord.inputs["Y"])
    tex = new("ShaderNodeTexImage"); tex.image = image
    tex.interpolation, tex.extension = "Closest", "EXTEND"
    link(coord.outputs[0], tex.inputs["Vector"])
    channel = new("ShaderNodeSeparateColor"); channel.mode = "RGB"
    link(tex.outputs["Color"], channel.inputs["Color"])
    positive = new("ShaderNodeMath"); positive.operation = "GREATER_THAN"
    link(channel.outputs["Red"], positive.inputs[0])
    colors = new("ShaderNodeMixRGB")
    colors.inputs[1].default_value = NEGATIVE; colors.inputs[2].default_value = color
    link(positive.outputs[0], colors.inputs[0])
    absolute = new("ShaderNodeMath"); absolute.operation = "ABSOLUTE"
    link(channel.outputs["Red"], absolute.inputs[0])
    power = new("ShaderNodeMath"); power.operation = "POWER"
    exponent=(5. if edge else 1.8) if power_exponent is None else float(power_exponent)
    power.inputs[1].default_value = exponent
    link(absolute.outputs[0], power.inputs[0])
    gain = new("ShaderNodeMath"); gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 3.8 if edge else 3.
    link(power.outputs[0], gain.inputs[0])
    for curve in _curves(tree):
        for point in curve.keyframe_points:
            point.interpolation = "CONSTANT"
    strength = gain.outputs[0]
    if edge:
        phase = new("ShaderNodeValue")
        for first, last in spans:
            phase.outputs[0].default_value = 0.
            phase.outputs[0].keyframe_insert("default_value", frame=first)
            phase.outputs[0].keyframe_insert("default_value", frame=min(last-1,first+delay))
            phase.outputs[0].default_value = 1.
            phase.outputs[0].keyframe_insert("default_value", frame=min(last, first + delay + 24))
        path = phase.outputs[0].path_from_id("default_value")
        for curve in _curves(tree):
            if curve.data_path == path:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
        flow = new("ShaderNodeUVMap"); flow.uv_map = "Flow"
        axis = new("ShaderNodeSeparateXYZ"); link(flow.outputs["UV"], axis.inputs[0])
        delta = new("ShaderNodeMath"); delta.operation = "SUBTRACT"
        link(axis.outputs["Y"], delta.inputs[0]); link(phase.outputs[0], delta.inputs[1])
        distance = new("ShaderNodeMath"); distance.operation = "ABSOLUTE"
        link(delta.outputs[0], distance.inputs[0])
        pulse = new("ShaderNodeMath"); pulse.operation = "LESS_THAN"; pulse.inputs[1].default_value = .13
        link(distance.outputs[0], pulse.inputs[0])
        wave = new("ShaderNodeMath"); wave.operation = "MULTIPLY_ADD"
        wave.inputs[1].default_value, wave.inputs[2].default_value = 3., .15
        link(pulse.outputs[0], wave.inputs[0])
        lit = new("ShaderNodeMath"); lit.operation = "MULTIPLY"
        link(gain.outputs[0], lit.inputs[0]); link(wave.outputs[0], lit.inputs[1]); strength = lit.outputs[0]
    emission = new("ShaderNodeEmission")
    link(colors.outputs[0], emission.inputs["Color"]); link(strength, emission.inputs["Strength"])
    base = new("ShaderNodeEmission")
    base.inputs["Color"].default_value, base.inputs["Strength"].default_value = QUIET, .006
    summed = new("ShaderNodeAddShader")
    link(base.outputs[0], summed.inputs[0]); link(emission.outputs[0], summed.inputs[1])
    output = new("ShaderNodeOutputMaterial"); link(summed.outputs[0], output.inputs["Surface"])
    uv_values = np.column_stack(((np.arange(count) % width + .5) / width,
                                 (np.arange(count) // width + .5) / height))
    return mat, uv_values, {"atlas_f32_sha256": sha_bytes(pixels.tobytes()),
                            "calibration": calibration, "width": width, "height": height,
                            "shader_power":exponent,"data_gain":3.8 if edge else 3.,"quiet_base":.006}


def channel_positions(count, x, y, z, spacing=.073):
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    ids = np.arange(count)
    return np.column_stack((np.full(count, x), y + (ids % columns - (columns - 1) / 2) * spacing,
                            z + (ids // columns - (rows - 1) / 2) * spacing))


def ribbons(name, begin, end, widths, coordinates, mat, camera):
    """Four vertices per scalar, one mesh per observed matrix, no edge objects."""
    count = len(begin)
    direction = end - begin
    eye = np.asarray(camera.location, dtype=np.float64) - (begin + end) / 2
    lateral = np.cross(direction, eye)
    norm = np.linalg.norm(lateral, axis=1)
    lateral[norm < 1e-12] = (0., 0., 1.)
    lateral /= np.linalg.norm(lateral, axis=1)[:, None]
    offset = lateral * widths[:, None]
    vertices = np.stack((begin - offset, begin + offset, end + offset, end - offset), axis=1).astype(np.float32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(4 * count); mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(4 * count); mesh.loops.foreach_set("vertex_index", np.arange(4 * count, dtype=np.int32))
    mesh.polygons.add(count)
    mesh.polygons.foreach_set("loop_start", np.arange(count, dtype=np.int32) * 4)
    mesh.polygons.foreach_set("loop_total", np.full(count, 4, dtype=np.int32))
    uv = mesh.uv_layers.new(name="Scalar")
    uv.data.foreach_set("uv", np.repeat(coordinates, 4, axis=0).astype(np.float32).ravel())
    flow = mesh.uv_layers.new(name="Flow")
    flow.data.foreach_set("uv", np.tile(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), (count, 1)).astype(np.float32).ravel())
    mesh.materials.append(mat); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    return obj


def neurons(name, positions, coordinates, mat, radius=.024):
    vertices, faces, _ = _node_geometry(positions, radius)
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(vertices, [], faces); mesh.update()
    uv = mesh.uv_layers.new(name="Scalar")
    for face in mesh.polygons:
        coordinate = coordinates[face.index // 20]
        for loop in face.loop_indices:
            uv.data[loop].uv = coordinate
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    return obj


def _position(name, role, fallback):
    """Editorial positions keyed only to operations present in the real trace."""
    if role == "problem_encoder":
        known = {"backbone.0": (-7.3, 0., 2.3), "backbone.3": (-5.4, 0., 2.3),
                 "inter_attn.token_proj": (-7.3, -.2, .1), "inter_attn.qkv": (-5.4, -.2, .1),
                 "inter_attn.bilinear_u": (-7.3, 1.7, -2.0), "inter_attn.bilinear_v": (-5.4, -1.5, -2.0),
                 "inter_attn.out_proj": (-3.2, 0., .2), "gate_head": (-2.0, 2.3, 3.1),
                 "v3_gate_head": (-.4, -2.3, 3.1), "task_latent_head": (-1.9, 0., 1.6),
                 "dist_latent_head": (-.4, 0., -.1), "landmark_recon_head": (-1.9, 2.3, -1.8),
                 "bb_recon_head": (-.4, -2.3, -1.8), "col_gate_head": (.1, 0., -3.0)}
    else:
        known = {"desc_proj": (1.5, 2.1, 2.8), "task_proj": (1.5, 1.1, 1.2),
                 "dist_proj": (1.5, -1.1, 1.2), "heads_proj": (1.5, -2.1, -.2),
                 "colgate_proj": (1.5, 2.1, -1.1), "hidden_proj": (1.5, -1.7, 3.0),
                 "col_encoder.0": (1.5, 0., -2.8), "col_encoder.2": (3.3, 0., -2.8),
                 "scale_side_proj": (4.7, 2.0, -2.8), "ctx_cross_attn.k_proj": (6.0, 1.2, 2.4),
                 "ctx_cross_attn.v_proj": (6.0, -1.2, .3), "ctx_bilinear_u": (7.4, 0., -2.2),
                 "cand_query_proj.0":(2.5,-4.,3.5),"ctx_cross_attn.q_proj":(4.7,-2.5,3.4),
                 "ctx_cross_attn.out_proj":(6.6,0.,1.5),"prog_bilinear_v":(6.0,2.7,-1.4),
                 "prior_residual_mlp.0":(7.1,0.,2.6),"prior_residual_out":(8.5,0.,2.6),
                 "hist_proj.0":(4.5,2.,-3.0),"traj_cross_attn.q_proj":(5.7,-2.,-2.5),
                 "traj_cross_attn.k_proj":(5.7,1.,-3.7),"traj_cross_attn.v_proj":(5.7,-1.,-3.7),
                 "traj_cross_attn.out_proj":(7.,0.,-3.0),"traj_residual_mlp.0":(7.2,0.,-1.6),
                 "traj_residual_out":(8.6,0.,-1.6)}
    matches = [(key, value) for key, value in known.items() if name.endswith(key)]
    if matches:
        return fold_position(max(matches, key=lambda item: len(item[0]))[1],fold_index(name))
    return fold_position(((-6. + fallback % 3 * 2.) if role == "problem_encoder" else (2. + fallback % 3 * 2.),
                          (fallback // 3 % 3 - 1) * 1.7, -3.4),fold_index(name))


def build_network(data, spans, camera, all_weights=True, max_connections=600_000, *, layout="technical"):
    if layout=="layered":
        return _build_layered_network(data,spans,camera,all_weights,max_connections)
    if layout!="technical":
        raise ValueError("Unknown H256 network layout")
    cases = data["cases"]
    inventories=[];functional_proofs=[]
    for case in cases:
        records=list(case["trace"]["affine"])
        if data["schema"]=="jain.h256-stage2-capture/v1" and any(p["id"]=="base.mm_cross_attn.in_proj_weight" for p in data["parameters"]):
            from h256_functional import derived_mha_affines
            derived,proof=derived_mha_affines(data,case)
            records.extend(derived);functional_proofs.append(proof)
        order={operation["id"]:i for i,operation in enumerate(case["operations"])}
        records.sort(key=lambda record:order[record["operation_id"]])
        inventories.append({(record["network"],record["name"]):record for record in records})
    keys = list(inventories[0])
    if any(set(inv) != set(keys) for inv in inventories):
        raise ValueError("Checkpoint matrix inventory differs across datasets")
    total = sum(np.count_nonzero(arrays(inventories[0][key])[1]) + np.count_nonzero(arrays(inventories[0][key])[2]) for key in keys)
    if all_weights and total > max_connections:
        raise ValueError(f"{total} scalar connections exceed explicit {max_connections} limit; refuse hidden sampling")
    receipts, ports, active_nodes, drawn = [], {}, 0, 0
    label_mat = material("H256 stage labels", (.52, .64, .78, 1), .8)
    base_ordinals={key:i for i,key in enumerate(key for key in keys if key[1].startswith("base."))}
    for index, key in enumerate(keys):
        records = [inv[key] for inv in inventories]
        observed = [arrays(record) for record in records]
        w, b = observed[0][1:3]
        if any(not np.array_equal(w, v[1]) or not np.array_equal(b, v[2]) for v in observed):
            raise ValueError("Saved checkpoint parameters changed across cases")
        role = "problem_encoder" if key[0].lower() == "dope" else "solution_encoder"
        fold=fold_index(key[1]);geometry_scale=.72 if fold is not None else 1.
        cx, cy, cz = _position(key[1], role, index)
        if key in base_ordinals:
            # The actually executed frozen base is a separate depth lane; its
            # matrices cannot share coordinates with new fold parameters.
            ordinal=base_ordinals[key];geometry_scale=.65
            cx,cy,cz=(-.2+(ordinal%5)*1.6,4.5+(ordinal//5)*.4,1.8-(ordinal//5)*1.2)
        ports[key[1]] = {"input": (cx - .56*geometry_scale, cy, cz), "output": (cx + .56*geometry_scale, cy, cz)}
        out, source = np.nonzero(w)
        if not all_weights and len(out) > 2048:
            selected = np.argsort(-np.abs(w[out, source]), kind="stable")[:2048]
            out, source = out[selected], source[selected]
        bias_ids = np.flatnonzero(b)
        # Hide genuinely inactive endpoints rather than fabricate incident edges.
        inputs = np.unique(source); outputs = np.union1d(out, bias_ids)
        if not len(outputs):
            receipts.append({"network": key[0], "name": key[1], "role": role,
                             "reason": "all W and bias entries zero", "connections": 0})
            continue
        left_all = channel_positions(w.shape[1], cx - .56*geometry_scale, cy, cz,.073*geometry_scale)
        right_all = channel_positions(w.shape[0], cx + .56*geometry_scale, cy, cz,.073*geometry_scale)
        bias_position = np.asarray((cx - .56*geometry_scale, cy, cz - geometry_scale))
        begin = np.concatenate((left_all[source], np.repeat(bias_position[None, :], len(bias_ids), axis=0)))
        end = np.concatenate((right_all[out], right_all[bias_ids]))
        scalars = np.concatenate((w[out, source], b[bias_ids]))
        terms, node_values, winners = [], [], []
        for x, ww, bb, y in observed:
            xv, xr = representative(x); yv, yr = representative(y)
            terms.append(np.concatenate((xv[source] * ww[out, source], bb[bias_ids])))
            node_values.append(np.concatenate((xv[inputs], yv[outputs], [1.] if len(bias_ids) else [])))
            winners.append({"input_rows": xr[inputs].tolist(), "output_rows": yr[outputs].tolist()})
        color = CYAN if role == "problem_encoder" else VIOLET
        delay = round(index / max(1,len(keys)-1) * 32)
        edge_mat, edge_uv, edge_atlas = scalar_material("Actual W·x · " + key[1], np.asarray(terms), spans, color, True, delay)
        widths = .00055 + .0022 * np.sqrt(np.abs(scalars) / max(float(np.max(np.abs(scalars))), 1e-30))
        edge_object=ribbons("Every recorded weight · " + key[1], begin, end, widths, edge_uv, edge_mat, camera)
        edge_object["h256_matrix_name"]=key[1];edge_object["h256_geometry_kind"]="edges"
        positions = np.concatenate((left_all[inputs], right_all[outputs], bias_position[None, :] if len(bias_ids) else np.empty((0, 3))))
        node_mat, node_uv, node_atlas = scalar_material("Actual activation · " + key[1], np.asarray(node_values), spans, color)
        node_object=neurons("Connected channels · " + key[1], positions, node_uv, node_mat,.024*geometry_scale)
        node_object["h256_matrix_name"]=key[1];node_object["h256_geometry_kind"]="nodes"
        short_name=key[1].removeprefix("dope.router.").removeprefix("dope.target_net.").removeprefix("h256.")
        if fold is None or fold==0:
            label = text(short_name.removeprefix("fold0."),
                         (cx - .58*geometry_scale, cy - .1, cz + .91*geometry_scale), .09, label_mat)
            label.rotation_euler = camera.rotation_euler
        active_nodes += len(positions); drawn += len(scalars)
        receipts.append({"network": key[0], "name": key[1], "role": role,
            "parameter_ids": records[0].get("parameter_ids", []), "weight_shape": list(w.shape),
            "execution_scope":records[0].get("execution_scope"),"call_source":records[0].get("call_source"),
            "functional_kind":records[0].get("functional_kind"),
            "source_parameter_slices":records[0].get("source_parameter_slices"),
            "operation_id":records[0].get("operation_id"),"editorial_pulse_delay_frames":delay,
            "fold_index":fold,"geometry_scale":geometry_scale,
            "weight_layout": "out_in", "weight_f32_sha256": sha_bytes(w.astype("<f4").tobytes()),
            "bias_f32_sha256": sha_bytes(b.astype("<f4").tobytes()),
            "nonzero_matrix_entries": int(np.count_nonzero(w)), "drawn_matrix_entries": len(out),
            "drawn_nonzero_biases": len(bias_ids), "connections": len(scalars),
            "input_channels": inputs.tolist(), "output_channels": outputs.tolist(),
            "all_displayed_neural_nodes_have_actual_incident_weight": True,
            "aggregation": "signed maximum-absolute actual token/batch row per channel; edge term uses winning input row",
            "winning_rows_by_case": winners, "input_shapes": [r["input_shape"] for r in records],
            "edge_coordinate_order": "np.nonzero(W) row-major (output,input), followed by nonzero bias indices",
            "term_atlas": edge_atlas, "node_atlas": node_atlas,
            "positions": ports[key[1]]})
    neutral = material("Typed functional routes", (.14, .20, .28, 1), .42)
    dependencies = cases[0].get("functional_dependencies", [])
    for case in cases[1:]:
        if case.get("functional_dependencies", []) != dependencies:
            raise ValueError("Functional architecture differs across cases")
    # Names such as aten.add.Tensor recur. Only immutable invocation IDs can
    # establish the real graph; merging equal display names invents routes.
    op_ports = {inventories[0][key]["operation_id"]:ports[key[1]] for key in keys}
    operation_ids = {operation["id"] for operation in cases[0]["operations"]}
    adjacency = {oid:[] for oid in operation_ids}
    raw_edges = {}
    for dependency in dependencies:
        a,b=dependency["from_operation_id"],dependency["to_operation_id"]
        if a not in operation_ids or b not in operation_ids:
            raise ValueError("Functional route references an unobserved invocation")
        if not dependency.get("kind") or not dependency.get("source_ref"):
            raise ValueError("Functional route requires a typed captured/source declaration")
        adjacency[a].append(b);raw_edges[(a,b)]=dependency
    tensor_lookup={tensor["id"]:tensor for tensor in cases[0]["tensors"]}
    # Keep real output junctions, including the hidden bypass, as graph anchors.
    output_locations={"dope629":(.25,0.,1.25),"hidden256":(-2.8,0.,2.0),
                      "context_tokens_92x32":(4.25,0.,.65),"context_mask_92":(4.25,0.,-1.0),
                      "block_log_norms_11":(.25,1.,-2.0),"context_summary_64":(6.2,0.,-1.6)}
    if data["schema"]=="jain.h256-stage2-capture/v1":
        for fold in range(5):
            for name,position in list(output_locations.items()):
                if not name.startswith("fold") and name not in {"dope629","hidden256"}:
                    output_locations[f"fold{fold}.{name}"]=fold_position(position,fold)
        output_locations.update({"ensemble_delta_prior":(7.7,0.,2.1),"ensemble_delta_trajectory":(7.7,0.,-1.),
                                 "ensemble_weighted_residual":(8.2,0.,.7)})
    for name,position in output_locations.items():
        tid=cases[0].get("outputs_tensor_ids",{}).get(name)
        producer=tensor_lookup.get(tid,{}).get("producer_op_id")
        if producer and producer not in op_ports:
            op_ports[producer]={"input":position,"output":position}
    # Root tensors are real observed inputs, not additional learned neurons.
    roots={}
    for operation in cases[0]["operations"]:
        for tid in operation["input_tensor_ids"]:
            tensor=tensor_lookup.get(tid)
            if tensor and tensor.get("producer_op_id") is None:
                root="input:"+tid
                if root not in roots:
                    name=tensor.get("name","")
                    position=(-8.65,0.,.5 if "v256" in name else -2.1)
                    roots[root]={"tensor_id":tid,"name":name}
                    op_ports[root]={"input":position,"output":position};adjacency[root]=[]
                adjacency[root].append(operation["id"])
                raw_edges[(root,operation["id"]) ]={"kind":"observed input tensor","source_ref":operation.get("call_source")}
    # Collapse only paths that truly exist in the captured DAG. Every bundle
    # retains complete operation-ID paths, including real forks and joins.
    bundles={};path_count=0
    def walk(source,node,path):
        nonlocal path_count
        if node in path:
            raise ValueError("Cycle in captured functional route")
        next_path=[*path,node]
        if node in op_ports and node!=source:
            bundles.setdefault((source,node),[]).append(next_path)
            path_count+=1
            if path_count>20000:
                raise ValueError("Functional bundle path budget exceeded")
            return
        for target in sorted(set(adjacency.get(node,[]))):
            walk(source,target,next_path)
    for source in op_ports:
        for target in sorted(set(adjacency.get(source,[]))):
            walk(source,target,[source])
    bundle_receipts=[];represented=set()
    for (source,target),paths in sorted(bundles.items()):
        a,b=Vector(op_ports[source]["output"]),Vector(op_ports[target]["input"])
        if (b-a).length<.01:
            continue
        edge_ids=set(zip(paths[0],paths[0][1:]))
        for path in paths[1:]:edge_ids.update(zip(path,path[1:]))
        represented.update(edge_ids)
        kinds=sorted({raw_edges[edge]["kind"] for edge in edge_ids})
        curve=bpy.data.curves.new("Contracted actual functional chain","CURVE")
        curve.dimensions="3D";curve.bevel_depth=.007;curve.bevel_resolution=1;curve.resolution_u=8
        spline=curve.splines.new("BEZIER");spline.bezier_points.add(2)
        midpoint=(a+b)/2+Vector((0.,.15,.16))
        for point,position in zip(spline.bezier_points,(a,midpoint,b)):
            point.co=position;point.handle_left_type=point.handle_right_type="AUTO"
        curve.materials.append(neutral)
        obj=bpy.data.objects.new("Actual dependency bundle",curve);bpy.context.collection.objects.link(obj)
        obj["h256_route_from"]=source;obj["h256_route_to"]=target
        bundle_receipts.append({"from_operation_id":source,"to_operation_id":target,
                                "operation_id_paths":paths,"typed_operations":kinds,
                                "role":"contracted recorded functional dependencies; never a learned weight"})
    # A tensor surface is kept distinct from neural glyphs: 92 recorded slots,
    # 32 coordinates each, with the capture's actual padding mask applied only
    # as an explicitly receipted display transform.
    tokens = np.asarray([output_value(c,"context_tokens_92x32") for c in cases], dtype=np.float32).reshape(len(cases), 92, 32)
    masks = np.asarray([output_value(c,"context_mask_92") for c in cases], dtype=np.float32).reshape(len(cases), 92)
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Context token mask must contain actual binary values")
    token_mat, token_uv, token_atlas = scalar_material("Actual 92-token surface", (tokens*masks[...,None]).reshape(len(cases), -1), spans, VIOLET)
    vertices, faces, uv = [], [], []
    right=camera.rotation_euler.to_quaternion()@Vector((1,0,0))
    up=camera.rotation_euler.to_quaternion()@Vector((0,1,0))
    normal=camera.rotation_euler.to_quaternion()@Vector((0,0,1))
    for slot in range(92):
        group, local = (0, slot) if slot < 20 else ((1, slot-20) if slot < 28 else (2, slot-28))
        cols = (5, 4, 8)[group]
        centre=Vector((4.2,0.,(2.05,.70,-.6)[group]))+normal*group*.15
        centre+=right*((local%cols-(cols-1)/2)*.27)+up*((local//cols-((4,2,8)[group]-1)/2)*.22)
        for channel in range(32):
            cell=centre+right*((channel%8-3.5)*.027)+up*((channel//8-1.5)*.04)
            offset = len(vertices)
            vertices.extend(tuple(cell+right*x+up*y) for x,y in ((-.011,-.017),(.011,-.017),(.011,.017),(-.011,.017)))
            faces.append((offset,offset+1,offset+2,offset+3))
            uv.append([token_uv[slot*32+channel]]*4)
    token_mesh = bpy.data.meshes.new("92 recorded token slots")
    token_mesh.from_pydata(vertices, [], faces); token_mesh.update()
    token_coordinates = token_mesh.uv_layers.new(name="Scalar")
    for polygon, coordinates in zip(token_mesh.polygons, uv):
        for loop, coordinate in zip(polygon.loop_indices, coordinates):
            token_coordinates.data[loop].uv = coordinate
    token_mesh.materials.append(token_mat)
    token_object = bpy.data.objects.new("Actual context token tensor surface", token_mesh)
    bpy.context.collection.objects.link(token_object)
    token_label = text("92 context tokens · fold0" if data["schema"]=="jain.h256-stage2-capture/v1" else "92 context tokens", (3.65,-.2,3.03), .14, label_mat)
    token_label.rotation_euler = camera.rotation_euler
    return {"schema": "jain.h256-full-matrix-scene/v1", "operations": receipts,
            "affine_operations": len(keys), "available_nonzero_connections": int(total),
            "drawn_nonzero_connections": int(drawn), "all_recorded_affine_connections_drawn": bool(drawn == total),
            "displayed_neural_nodes": int(active_nodes), "functional_dependencies": dependencies,
            "fold_indices":sorted({fold_index(key[1]) for key in keys if fold_index(key[1]) is not None}),
            "derived_functional_affine_proofs":functional_proofs,
            "raw_observed_linear_calls":len(cases[0]["trace"]["affine"]),
            "derived_actual_functional_matrix_calls":len(keys)-len(cases[0]["trace"]["affine"]),
            "ensemble_geometry":"separate saved-fold parameter volumes; never parameter averaging",
            "functional_bundles":bundle_receipts,"observed_root_tensors":roots,
            "functional_edges_represented":len(represented),"functional_edges_available":len(raw_edges),
            "omitted_functional_edges":[{"from":a,"to":b,"kind":value["kind"]} for (a,b),value in raw_edges.items() if (a,b) not in represented],
            "context_token_surface": {"slots":92,"coordinates_per_slot":32,"kind":"measured tensor surface, not neural glyphs",
                "fold_index":0 if data["schema"]=="jain.h256-stage2-capture/v1" else None,
                "active_slots_by_case":masks.sum(axis=1).astype(int).tolist(),"atlas":token_atlas,
                "display_transform":"recorded token values times recorded padding mask; not an additional inference operation",
                "raw_f32_sha256_by_case":[sha_bytes(t.astype("<f4").tobytes()) for t in tokens]},
            "scope": "actual observed affine parameters; typed neutral functional routes; no missing query/readout or old compression bridge",
            "pulse_motion": "editorial linear travel; signed measured W*x magnitude gates every colored pulse",
            "zero_policy": "zero scalar has zero data emission; neutral structural base is constant",
            "parameter_coverage": "all recorded affine W entries and nonzero biases; full checkpoint inventory remains in capture, not a claim that every parameter is affine"}


def weighted_causal_depths(case, weighted_operation_ids):
    """Pure producer-DAG depth; functional operations never invent a layer."""
    depths={};inputs={};weighted=set(weighted_operation_ids)
    for operation in case["operations"]:
        parents=operation.get("parent_op_ids",[])
        if any(parent not in depths for parent in parents):
            raise ValueError("Captured operations are not in producer order")
        incoming=max((depths[parent] for parent in parents),default=0)
        inputs[operation["id"]]=incoming
        depths[operation["id"]]=incoming+int(operation["id"] in weighted)
    if not weighted<=depths.keys():
        raise ValueError("Weighted operation is missing from the actual DAG")
    return inputs,depths


def _continuous_layer_material(mat,spans,color,stage_fraction,edge):
    """Retain the signed measured atlas, display a continuous section palette."""
    tree=mat.node_tree;new,link=tree.nodes.new,tree.links.new
    source=next(node for node in tree.nodes if node.bl_idname=="ShaderNodeSeparateColor")
    actual=next(node for node in tree.nodes if node.bl_idname=="ShaderNodeEmission" and node.inputs["Strength"].is_linked)
    absolute=new("ShaderNodeMath");absolute.operation="ABSOLUTE";link(source.outputs["Red"],absolute.inputs[0])
    ramp=new("ShaderNodeValToRGB");ramp.color_ramp.interpolation="LINEAR"
    palette=([(0.,(.003,.018,.035,1)),(.30,(.005,.15,.36,1)),(.65,(.015,.55,.79,1)),(1.,(.12,.91,1.,1))]
             if color==CYAN else [(0.,(.012,.005,.032,1)),(.30,(.12,.025,.32,1)),(.65,(.38,.10,.75,1)),(1.,(.72,.38,1.,1))])
    first=ramp.color_ramp.elements[0];first.position,first.color=palette[0]
    last=ramp.color_ramp.elements[1];last.position,last.color=palette[-1]
    for position,value in palette[1:-1]:ramp.color_ramp.elements.new(position).color=value
    link(absolute.outputs[0],ramp.inputs["Fac"])
    for old in list(actual.inputs["Color"].links):tree.links.remove(old)
    link(ramp.outputs["Color"],actual.inputs["Color"])
    power=new("ShaderNodeMath");power.operation="POWER";power.inputs[1].default_value=2.8 if edge else .8
    link(absolute.outputs[0],power.inputs[0]);strength=power.outputs[0]
    if edge:
        progress=new("ShaderNodeValue");progress.label="Repeated editorial playback of one recorded case"
        for first,last in spans:
            progress.outputs[0].default_value=0.;progress.outputs[0].keyframe_insert("default_value",frame=first)
            progress.outputs[0].default_value=(last-first)/48.;progress.outputs[0].keyframe_insert("default_value",frame=last)
        path=progress.outputs[0].path_from_id("default_value")
        for curve in _curves(tree):
            if curve.data_path==path:
                for point in curve.keyframe_points:point.interpolation="LINEAR"
        delay=new("ShaderNodeMath");delay.operation="SUBTRACT";delay.inputs[1].default_value=stage_fraction*.65
        link(progress.outputs[0],delay.inputs[0])
        phase=new("ShaderNodeMath");phase.operation="FRACT";link(delay.outputs[0],phase.inputs[0])
        flow=new("ShaderNodeUVMap");flow.uv_map="Flow"
        axis=new("ShaderNodeSeparateXYZ");link(flow.outputs["UV"],axis.inputs[0])
        difference=new("ShaderNodeMath");difference.operation="SUBTRACT"
        link(axis.outputs["Y"],difference.inputs[0]);link(phase.outputs[0],difference.inputs[1])
        distance=new("ShaderNodeMath");distance.operation="ABSOLUTE";link(difference.outputs[0],distance.inputs[0])
        pulse=new("ShaderNodeMath");pulse.operation="LESS_THAN";pulse.inputs[1].default_value=.10
        link(distance.outputs[0],pulse.inputs[0])
        wave=new("ShaderNodeMath");wave.operation="MULTIPLY_ADD";wave.inputs[1].default_value=.82;wave.inputs[2].default_value=.035
        link(pulse.outputs[0],wave.inputs[0])
        gain=new("ShaderNodeMath");gain.operation="MULTIPLY";link(power.outputs[0],gain.inputs[0]);link(wave.outputs[0],gain.inputs[1]);strength=gain.outputs[0]
    else:
        gain=new("ShaderNodeMath");gain.operation="MULTIPLY";gain.inputs[1].default_value=2.3
        link(power.outputs[0],gain.inputs[0]);strength=gain.outputs[0]
    for old in list(actual.inputs["Strength"].links):tree.links.remove(old)
    link(strength,actual.inputs["Strength"])
    for node in tree.nodes:
        if node.bl_idname=="ShaderNodeEmission" and node!=actual and not node.inputs["Strength"].is_linked:
            node.inputs["Strength"].default_value=.0004 if edge else .008


def _build_layered_network(data,spans,camera,all_weights,max_connections):
    """Reposition exact technical meshes; never construct cross-branch weights."""
    before=set(bpy.data.objects)
    receipt=build_network(data,spans,camera,all_weights,max_connections,layout="technical")
    created=[obj for obj in bpy.data.objects if obj not in before]
    operations=receipt["operations"]
    case=data["cases"][0]
    op_by_id={operation["id"]:operation for operation in case["operations"]}
    incoming,depths=weighted_causal_depths(case,{record["operation_id"] for record in operations if record.get("operation_id")})
    maximum=max(depths.values());stage_entries={stage:[] for stage in range(maximum+1)}
    by_name={record["name"]:record for record in operations}
    def family(record):
        if record["role"]=="problem_encoder":return 0
        fold=record.get("fold_index")
        return 1 if fold is None else 2+fold
    for record in operations:
        if not record.get("operation_id"):continue
        oid=record["operation_id"];record["input_stage"]=incoming[oid];record["output_stage"]=depths[oid]
        for side,channels,stage in (("input",record["input_channels"],incoming[oid]),("output",record["output_channels"],depths[oid])):
            stage_entries[stage].extend((family(record),record["name"],side,int(channel)) for channel in channels)
        if record["drawn_nonzero_biases"]:
            stage_entries[incoming[oid]].append((family(record),record["name"],"bias",0))
    positions={};radii={};stage_receipts=[]
    for stage,entries in stage_entries.items():
        x=-7.8+15.6*stage/max(1,maximum);entries.sort();count=len(entries);start=-2.4;bands=[]
        for group in sorted({entry[0] for entry in entries}):
            subset=[entry for entry in entries if entry[0]==group]
            width=4.8*len(subset)/max(1,count)
            rows=max(1,math.ceil(math.sqrt(len(subset)*6.8/max(width,.01))))
            columns=max(1,math.ceil(len(subset)/rows));dy=width/columns;dz=6.8/rows
            for index,entry in enumerate(subset):
                _,name,side,channel=entry
                positions[(name,side,channel)]=np.asarray((x,start+(index//rows+.5)*dy,-3.4+(index%rows+.5)*dz))
                radii[(name,side,channel)]=min(.052,min(dy,dz)*.36)
            bands.append({"family":group,"nodes":len(subset),"depth_interval":[start,start+width],"rows":rows,"columns":columns})
            start+=width
        stage_receipts.append({"stage":stage,"x":x,"nodes":count,"parallel_family_bands":bands})
    records={record["name"]:record for record in case["trace"]["affine"]}
    if receipt["derived_actual_functional_matrix_calls"]:
        from h256_functional import derived_mha_affines
        derived,_=derived_mha_affines(data,case);records.update({record["name"]:record for record in derived})
    from network_graph import ICO
    actor_ports={}
    for obj in created:
        name=obj.get("h256_matrix_name");kind=obj.get("h256_geometry_kind")
        if not name:continue
        record=by_name[name];x,w,b,y=arrays(records[name])
        out,source=np.nonzero(w)
        if not all_weights and len(out)>2048:
            selected=np.argsort(-np.abs(w[out,source]),kind="stable")[:2048];out,source=out[selected],source[selected]
        bias=np.flatnonzero(b)
        if kind=="nodes":
            identities=[(name,"input",int(channel)) for channel in record["input_channels"]]
            identities.extend((name,"output",int(channel)) for channel in record["output_channels"])
            if len(bias):identities.append((name,"bias",0))
            centres=np.asarray([positions[key] for key in identities]);sizes=np.asarray([radii[key] for key in identities])
            vertices=(centres[:,None,:]+np.asarray(ICO)[None,:,:]*sizes[:,None,None]).astype(np.float32)
            obj.data.vertices.foreach_set("co",vertices.ravel());obj.data.update()
            actor_ports[record["operation_id"]]={}
            for side in ("input","output"):
                values=[positions[(name,side,int(channel))] for channel in record[side+"_channels"]]
                if not values and side=="input" and len(bias):values=[positions[(name,"bias",0)]]
                if not values:raise ValueError("Displayed matrix endpoint has no actual incident channel")
                actor_ports[record["operation_id"]][side]=np.mean(values,axis=0)
            record["positions"]={side:value.tolist() for side,value in actor_ports[record["operation_id"]].items()}
            record["layered_node_radius_range"]=[float(np.min(sizes)),float(np.max(sizes))]
            record["node_atlas"].update(shader_power=.8,data_gain=2.3,quiet_base=.008)
        else:
            begin=np.asarray([positions[(name,"input",int(channel))] for channel in source])
            end=np.asarray([positions[(name,"output",int(channel))] for channel in out])
            if len(bias):
                begin=np.concatenate((begin,np.repeat(positions[(name,"bias",0)][None,:],len(bias),axis=0)))
                end=np.concatenate((end,[positions[(name,"output",int(channel))] for channel in bias]))
            scalars=np.concatenate((w[out,source],b[bias]));widths=.00015+.00075*np.sqrt(np.abs(scalars)/max(float(np.max(np.abs(scalars))),1e-30))
            direction=end-begin;eye=np.asarray(camera.location)-(begin+end)/2;lateral=np.cross(direction,eye)
            norms=np.linalg.norm(lateral,axis=1);lateral[norms<1e-12]=(0.,0.,1.);lateral/=np.linalg.norm(lateral,axis=1)[:,None]
            offset=lateral*widths[:,None]
            vertices=np.stack((begin-offset,begin+offset,end+offset,end-offset),axis=1).astype(np.float32)
            obj.data.vertices.foreach_set("co",vertices.ravel());obj.data.update()
            record["term_atlas"].update(shader_power=2.8,data_gain=.855,quiet_base=.0004)
        _continuous_layer_material(obj.data.materials[0],spans,CYAN if record["role"]=="problem_encoder" else VIOLET,
                                   record["input_stage"]/max(1,maximum),kind=="edges")
    # Generic aten names do not identify a fold. Keep parallel functional
    # junctions separate using actual parameter and producer ancestry. Only a
    # real multi-fold operation may reconverge in the central shared lane.
    weighted_folds={record["operation_id"]:record.get("fold_index") for record in operations if record.get("operation_id")}
    fold_owners={}
    for operation in case["operations"]:
        oid=operation["id"]
        direct={fold_index(name) for name in [operation["name"],*operation.get("parameter_ids",[])]}
        direct.discard(None)
        if weighted_folds.get(oid) is not None:direct.add(weighted_folds[oid])
        inherited=set().union(*(fold_owners[parent] for parent in operation.get("parent_op_ids",[])))
        if direct and inherited and not inherited<=direct:
            raise ValueError("Functional fold ownership conflicts with recorded producers")
        fold_owners[oid]=direct or inherited
    junction_receipts={}
    def port(oid,side):
        if oid in actor_ports:return Vector(actor_ports[oid][side])
        if oid.startswith("input:"):return Vector((-7.8,0.,0.))
        stage=depths[oid];owners=fold_owners[oid]
        fold=next(iter(owners)) if len(owners)==1 else None
        band=next((band for band in stage_receipts[stage]["parallel_family_bands"] if fold is not None and band["family"]==2+fold),None)
        depth=0. if fold is None else sum(band["depth_interval"])/2 if band else (fold-2)*.85
        position=(-7.8+15.6*stage/max(1,maximum),depth,0.)
        junction_receipts[oid]={"fold_owners":sorted(owners),"fold_index":fold,"position":list(position),
                               "ownership_source":"actual parameter IDs and producer invocation ancestry"}
        return Vector(position)
    for obj in created:
        if obj.type=="FONT" or obj.name.startswith("Actual context token tensor surface"):
            obj.hide_render=True;obj.hide_viewport=True
        if obj.get("h256_route_from"):
            a=port(obj["h256_route_from"],"output");b=port(obj["h256_route_to"],"input")
            midpoint=(a+b)/2+Vector((0.,.05,.08))
            for point,position in zip(obj.data.splines[0].bezier_points,(a,midpoint,b)):point.co=position
            obj.data.bevel_depth=.002
    # Technical mode supplied the immutable scalar atlas and identities. Its
    # editorial shader/position metadata must describe the active layered view.
    for record in operations:
        if not record.get("operation_id"):
            continue
        record["technical_editorial_pulse_delay_frames"]=record["editorial_pulse_delay_frames"]
        record["editorial_pulse_delay_frames"]=record["input_stage"]/max(1,maximum)*.65*48.
        record["editorial_pulse_cycle_frames"]=48
        record["technical_positions"]=record["positions"]
        record["positions"]={side:actor_ports[record["operation_id"]][side].tolist() for side in ("input","output")}
        record["technical_geometry_scale"]=record.pop("geometry_scale")
        record["node_radius_policy"]="min(0.052, 0.36*min(stage-band cell width,cell height))"
        for field,power,gain,quiet in (("term_atlas",2.8,.855,.0004),("node_atlas",.8,2.3,.008)):
            atlas=record[field]
            atlas["technical_shader_power"]=atlas["shader_power"]
            atlas["technical_data_gain"]=atlas["data_gain"]
            atlas["shader_power"],atlas["data_gain"],atlas["quiet_base"]=power,gain,quiet
            atlas["color_mapping"]="linear continuous section ramp of abs(signed normalized measured value)"
            if field=="term_atlas":atlas["data_gain_range"]=[.035,.855]
    receipt.update({"layout":"layered","stage_planes":stage_receipts,"maximum_weighted_causal_depth":maximum,
                    "geometry_bounds":{"x":[-8.,8.],"depth":[-2.5,2.5],"z":[-3.6,3.6]},
                    "objects_are_unparented":True,"token_surface_displayed":False,"operation_labels_displayed":False,
                    "functional_junctions":junction_receipts,
                    "node_identity":"separate actual module input/output channels; shared x plane does not equate tensors",
                    "shared_rows":"same signed max-absolute token/candidate-row observations as technical mode",
                    "colors":"continuous absolute measured strength in cyan/violet section palettes; signed data retained in trace and atlas",
                    "pulse_motion":"48-frame repeated editorial travel across every full case; delay follows actual weighted producer depth",
                    "geometry_sampling":False if all_weights else "explicit 2048-edge per-matrix cap"})
    receipt["context_token_surface"].update(displayed=False,hide_reason="shared channel stage view; recorded tensor surface omitted for clarity")
    return receipt
