"""Clean layered display sampled from actual recorded neural matrix operations.

This is a representative display, not a full architecture or parameter census.
Every internal weighted strand is a real recorded W[out,in] multiplied by its
actual input. Shared x planes are editorial; distinct tensors remain distinct.
Ingress/egress ports are conceptual display ports for the surrounding scene.
"""
from __future__ import annotations

import hashlib
import math

import bpy
import numpy as np

from h256_network import (arrays, representative, ribbons, scalar_material,
                          weighted_causal_depths, _continuous_layer_material)
from network_graph import ICO, ICO_FACES, _select


CHANNELS = 6
MAX_CANDIDATE_ROWS = 12


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _hue(fraction):
    stops = [(0., np.array((.015, .80, .96))), (.48, np.array((.075, .38, 1.))),
             (1., np.array((.68, .16, 1.)))]
    for (left, a), (right, b) in zip(stops, stops[1:]):
        if fraction <= right:
            rgb = a + (b-a)*(fraction-left)/(right-left)
            return (*rgb.tolist(), 1.)
    return (*stops[-1][1].tolist(), 1.)


def _shade(mat, spans, fraction, edge, pulse_frames):
    # The existing helper keeps the signed FP32 atlas, fixed case switches and
    # 48-frame editorial pulses. Only the display palette/backbone changes.
    _continuous_layer_material(mat, spans, (), fraction, edge)
    if edge:
        progress = next(node for node in mat.node_tree.nodes
                        if node.bl_idname == "ShaderNodeValue" and node.label.startswith("Repeated editorial"))
        from h256_network import _curves
        path = progress.outputs[0].path_from_id("default_value")
        for curve in _curves(mat.node_tree):
            if curve.data_path == path:
                for point in curve.keyframe_points:
                    point.co.y *= 48. / pulse_frames
    ramp = next(node for node in reversed(list(mat.node_tree.nodes))
                if node.bl_idname == "ShaderNodeValToRGB")
    hue = np.asarray(_hue(fraction)[:3])
    for element, level in zip(ramp.color_ramp.elements, (.035, .25, .65, 1.)):
        element.color = (*(hue*level).tolist(), 1.)
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeEmission" and not node.inputs["Strength"].is_linked:
            node.inputs["Color"].default_value = (*(hue*.20).tolist(), 1.)
            node.inputs["Strength"].default_value = .06 if edge else .16


def _unit_sphere():
    vertices = [point.copy() for point in np.asarray(ICO)]
    mids = {}
    def midpoint(a, b):
        key = tuple(sorted((a,b)))
        if key not in mids:
            point = (vertices[a]+vertices[b])/2
            point /= np.linalg.norm(point)
            mids[key] = len(vertices); vertices.append(point)
        return mids[key]
    faces = []
    for a,b,c in ICO_FACES:
        ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
        faces.extend(((a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)))
    return np.asarray(vertices), faces


SPHERE, SPHERE_FACES = _unit_sphere()


def _orbs(name, positions, uv_values, mat, radius):
    vertices = (np.asarray(positions)[:,None,:]+SPHERE[None,:,:]*radius).reshape(-1,3)
    faces = [tuple(index*len(SPHERE)+v for v in face)
             for index in range(len(positions)) for face in SPHERE_FACES]
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(vertices.tolist(), [], faces); mesh.update()
    uv = mesh.uv_layers.new(name="Scalar")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        coordinate = uv_values[polygon.index//len(SPHERE_FACES)]
        for loop in polygon.loop_indices: uv.data[loop].uv = coordinate
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    return obj


def _records(data):
    cases = data["cases"]; result = []; proofs = []
    for case in cases:
        records = list(case["trace"]["affine"])
        if any(parameter["id"] == "base.mm_cross_attn.in_proj_weight" for parameter in data["parameters"]):
            from h256_functional import derived_mha_affines
            extra, proof = derived_mha_affines(data, case)
            records.extend(extra); proofs.append(proof)
        order = {operation["id"]: i for i, operation in enumerate(case["operations"])}
        records.sort(key=lambda record: order[record["operation_id"]])
        result.append({record["name"]: record for record in records})
    if any(list(case) != list(result[0]) for case in result):
        raise ValueError("Representative display requires the same recorded operation inventory")
    return result, proofs


def _family(name):
    if name.startswith("dope."): return (0, name)
    if name.startswith("base."): return (1, name)
    return (2, name)


def build_display_network(data, spans, camera, *, width=16.2, pulse_frames=48):
    """Build sampled real strands; return JSON-safe receipt and display ports."""
    if not math.isfinite(width) or not 4 <= width <= 20:
        raise ValueError("Display width must be4..20 world units")
    if not isinstance(pulse_frames, int) or pulse_frames < 2:
        raise ValueError("Pulse period must be at least two frames")
    half_width = width/2
    if len(spans) != len(data["cases"]):
        raise ValueError("One full-case interval is required per captured dataset")
    inventories, functional_proofs = _records(data)
    case = data["cases"][0]
    incoming, depths = weighted_causal_depths(case,
        {record["operation_id"] for record in inventories[0].values()})
    maximum = max(depths.values())
    plans = []; stage_nodes = {stage: [] for stage in range(maximum+1)}
    available_edges = 0
    for name in inventories[0]:
        records = [inventory[name] for inventory in inventories]
        observed = [arrays(record) for record in records]
        w,b = observed[0][1:3]
        if any(not np.array_equal(w, other[1]) or not np.array_equal(b, other[2]) for other in observed):
            raise ValueError("Recorded weights changed between display cases")
        available_edges += int(np.count_nonzero(w)+np.count_nonzero(b))
        input_bound = np.max(np.stack([np.max(np.abs(x),axis=0) for x,_,_,_ in observed]),axis=0)
        score = np.abs(w)*input_bound[None,:]
        if not np.any(score): score = np.abs(w)
        picked, selected_in, selected_out = _select(score, CHANNELS, CHANNELS*CHANNELS)
        # If an input was zero in every capture, the frozen parameter is still
        # structural evidence, but its actual contribution remains exactly zero.
        if not picked and np.any(w):
            picked, selected_in, selected_out = _select(np.abs(w), CHANNELS, CHANNELS*CHANNELS)
        if not picked and np.any(b):
            selected_out = sorted(int(index) for index in np.argsort(-np.abs(b),kind="stable") if b[index]!=0)[:CHANNELS]
        bias_out = [int(out) for out in selected_out if b[out] != 0]
        if not selected_out:
            plans.append({"name":name,"operation_id":records[0]["operation_id"],"inactive":True})
            continue
        # Unroll real candidate rows only for shared scalar heads. This produces
        # genuine score glyphs without inventing independently weighted outputs.
        rows = (list(range(min(MAX_CANDIDATE_ROWS,min(x.shape[0] for x,_,_,_ in observed))))
                if w.shape[0] == 1 and min(x.shape[0] for x,_,_,_ in observed)>1 else [None])
        node_specs = [("input",row,int(channel)) for row in rows for channel in selected_in]
        node_specs += [("output",row,int(channel)) for row in rows for channel in selected_out]
        if bias_out: node_specs.append(("bias",None,0))
        local = {spec: i for i,spec in enumerate(node_specs)}
        edge_specs = []
        for row in rows:
            for out,source in picked:
                edge_specs.append({"from":local[("input",row,source)],"to":local[("output",row,out)],
                                   "input":source,"output":out,"row":row,"bias":False,"weight":float(w[out,source])})
            for out in bias_out:
                edge_specs.append({"from":local[("bias",None,0)],"to":local[("output",row,out)],
                                   "input":None,"output":out,"row":row,"bias":True,"weight":float(b[out])})
        oid = records[0]["operation_id"]
        for index,(side,row,channel) in enumerate(node_specs):
            stage = depths[oid] if side == "output" else incoming[oid]
            stage_nodes[stage].append((name,index,side,row,channel))
        # A displayed affine has one actual row, shared by every input and
        # output endpoint. Choose it once across cases by input-contribution
        # energy; explicit scalar heads still unroll their actual candidate rows.
        row_count = min(x.shape[0] for x,_,_,_ in observed)
        if any(x.shape[0] != y.shape[0] for x,_,_,y in observed):
            raise ValueError("Actual affine input/output rows differ")
        row_energy = sum(np.sum(np.abs(x[:row_count])*np.abs(w).sum(axis=0),axis=1)
                         for x,_,_,_ in observed)
        display_row = int(np.argmax(row_energy))
        node_values = []; terms = []; winners = []
        for x,ww,bb,y in observed:
            xv, yv = x[display_row], y[display_row]
            xr = np.full(x.shape[1],display_row,dtype=int)
            yr = np.full(y.shape[1],display_row,dtype=int)
            values = [1. if side=="bias" else float((xv if side=="input" else yv)[channel]) if row is None
                      else float((x if side=="input" else y)[row,channel]) for side,row,channel in node_specs]
            node_values.append(values)
            terms.append([edge["weight"] if edge["bias"] else values[edge["from"]]*edge["weight"] for edge in edge_specs])
            winners.append({"input_rows":{str(channel):int(xr[channel]) for channel in selected_in},
                            "output_rows":{str(channel):int(yr[channel]) for channel in selected_out}})
        plans.append({"name":name,"operation_id":oid,"records":records,"nodes":node_specs,"edges":edge_specs,
                      "node_values":np.asarray(node_values),"terms":np.asarray(terms),"winning_rows":winners,
                      "input_stage":incoming[oid],"output_stage":depths[oid],"shared_candidate_rows":rows,
                      "display_row":display_row})
    positions = {}; stage_receipts = []
    for stage, nodes in stage_nodes.items():
        nodes.sort(key=lambda node:(_family(node[0]),node[1]))
        columns = max(1,math.ceil(len(nodes)/32)); x = -half_width+width*stage/max(1,maximum)
        for index,(name,local,*_) in enumerate(nodes):
            column = index//32; row = index%32; actual_rows = min(32,len(nodes)-column*32)
            y = 0. if columns==1 else -1.1+2.2*column/(columns-1)
            z = 0. if actual_rows==1 else -4.35+8.7*row/(actual_rows-1)
            positions[(name,local)] = np.asarray((x,y,z))
        stage_receipts.append({"stage":stage,"x":x,"displayed_actual_endpoint_nodes":len(nodes),
                               "vertical_rows":32,"depth_columns":columns})
    root = bpy.data.objects.new("JAIN display network",None);bpy.context.collection.objects.link(root)
    receipts = []; edge_count = 0; node_count = 0
    for plan in plans:
        if plan.get("inactive"):
            receipts.append({key:plan[key] for key in ("name","operation_id","inactive")});continue
        name = plan["name"]; fraction = plan["output_stage"]/max(1,maximum)
        node_mat,node_uv,node_atlas = scalar_material("Display measured activations · "+name,
            plan["node_values"],spans,_hue(fraction),power_exponent=.8)
        _shade(node_mat,spans,fraction,False,pulse_frames)
        centres = np.asarray([positions[(name,index)] for index in range(len(plan["nodes"]))])
        orbs = _orbs("JAIN measured neuron samples",centres,node_uv,node_mat,.052);orbs.parent=root
        edge_mat,edge_uv,edge_atlas = scalar_material("Display measured contributions · "+name,
            plan["terms"],spans,_hue(fraction),power_exponent=2.8)
        _shade(edge_mat,spans,plan["input_stage"]/max(1,maximum),True,pulse_frames)
        begin = np.asarray([centres[edge["from"]] for edge in plan["edges"]])
        end = np.asarray([centres[edge["to"]] for edge in plan["edges"]])
        weight = np.abs([edge["weight"] for edge in plan["edges"]]); bound=max(float(np.max(weight)),1e-30)
        widths = .0015+.0020*np.sqrt(weight/bound)
        strands = ribbons("JAIN actual sampled weight strands",begin,end,widths,edge_uv,edge_mat,camera);strands.parent=root
        node_count += len(centres);edge_count += len(plan["edges"])
        native = plan["records"][0]
        receipts.append({"name":name,"operation_id":plan["operation_id"],"parameter_ids":native.get("parameter_ids",[]),
            "functional_kind":native.get("functional_kind"),"source_parameter_slices":native.get("source_parameter_slices"),
            "input_shape_by_case":[record["input_shape"] for record in plan["records"]],"weight_shape":native["weight_shape"],
            "input_stage":plan["input_stage"],"output_stage":plan["output_stage"],
            "actual_node_identities":[{"side":side,"row":None if side=="bias" else plan["display_row"] if row is None else row,"channel":channel}
                                      for side,row,channel in plan["nodes"]],
            "fixed_display_row":plan["display_row"],
            "actual_weight_strands":plan["edges"],"winning_rows_by_case":plan["winning_rows"],
            "node_values_f64_sha256_by_case":[_sha(value) for value in plan["node_values"]],
            "signed_terms_f64_sha256_by_case":[_sha(value) for value in plan["terms"]],
            "shared_candidate_rows":plan["shared_candidate_rows"],"node_atlas":node_atlas,"term_atlas":edge_atlas,
            "all_displayed_nodes_have_real_incident_weight":True})
    input_nodes=[positions[(name,index)].tolist() for name,index,*_ in stage_nodes[0]]
    output_nodes=[positions[(name,index)].tolist() for name,index,*_ in stage_nodes[maximum]]
    return {"schema":"jain.clean-measured-display/v1","root_object_name":root.name,
        "input_port":[-half_width,0.,0.],"output_port":[half_width,0.,0.],"input_node_ports":input_nodes,"output_node_ports":output_nodes,
        "bounds":{"x":[-half_width-.06,half_width+.06],"depth":[-1.2,1.2],"z":[-4.41,4.41]},
        "stage_planes":stage_receipts,"maximum_weighted_causal_depth":maximum,
        "recorded_matrix_operations":len(inventories[0]),"available_nonzero_scalar_connections":available_edges,
        "displayed_nodes":node_count,"displayed_strands":edge_count,"operations":receipts,
        "affine_operations":len(inventories[0]),"drawn_nonzero_connections":edge_count,
        "displayed_neural_nodes":node_count,"all_recorded_affine_connections_drawn":False,
        "functional_projection_proofs":functional_proofs,
        "selection":"fixed strongest measured-contribution channels across all cases; at most6 input/output channels per recorded matrix; one actual shared row per nonscalar affine",
        "coverage":"representative actual neural channels/weights; full raw architecture and parameter coverage are intentionally not drawn",
        "node_semantics":"distinct actual module endpoints; one fixed actual common input/output row per affine, except explicitly unrolled candidate rows",
        "shared_head_semantics":"one saved1×N matrix reused on actual candidate rows, never a fabricated N×candidate parameter matrix",
        "palette":"continuous cyan/blue/violet across actual causal depth; intensity is calibrated measured magnitude; sign retained in receipts",
        "pulse":f"{pulse_frames}-frame repeating editorial playback over each entire case; not measured physical transit time",
        "pulse_frames":pulse_frames,
        "ingress_egress":"conceptual scene ports; internal weighted strands retain actual parameter coordinates",
        "labels":False,"token_surfaces":False,"functional_spiderweb":False}
