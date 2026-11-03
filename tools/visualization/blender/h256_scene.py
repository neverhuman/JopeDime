"""Standalone private 3D rendering of the actual saved H256 Stage-1 encoder.

Blender --background --python h256_scene.py -- --trace ... --output ...
Only stills and a saved scene are produced. No old GP/code proof is spliced in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import bpy
import numpy as np
from mathutils import Vector

from build_scene import interpolation, line, material, text
from h256_network import build_network, neurons, scalar_material
from code_pages import reconstruct_source
from validate_trace import gp_sexpression


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_source_pages(data,args,validation):
    """Require the root validator's actual new-model-to-GP causal proof."""
    if args.code_pages is None and args.gp_proof is None:
        return None
    if args.code_pages is None or args.gp_proof is None:
        raise ValueError("--code-pages and --gp-proof are required together")
    if data["schema"]!="jain.h256-stage2-capture/v1":
        raise ValueError("Stage1 encoder diagnostics cannot acquire a GP source claim")
    verified=validation.get("gp_sources",{})
    result=[];trace_sha=digest(args.trace)
    for case in data["cases"]:
        source=case.get("source");cid=case["id"]
        if not isinstance(source,dict) or source.get("case_id")!=cid:
            raise ValueError("Whole-source view needs a source bound to every actual case")
        proof=verified.get(cid,{})
        if not (proof.get("status")=="PASS" and proof.get("causal_link_verified") is True
                and proof.get("trace_sha256")==trace_sha
                and all(proof.get(key)==source.get(key) and bool(source.get(key))
                        for key in ("rust_sha256","proof_sha256","selected_program_sha256"))):
            raise ValueError("Root verifier has not established the new checkpoint's GP/source causality")
        relative=Path(source["proof_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GP proof path must remain inside its supplied root")
        proof_path=(args.gp_proof/relative).resolve()
        if not proof_path.is_relative_to(args.gp_proof.resolve()) or digest(proof_path)!=source["proof_sha256"]:
            raise ValueError("Verified causal GP proof changed")
        manifest_path=args.code_pages/cid/"manifest.json"
        manifest=json.loads(manifest_path.read_text());raw=reconstruct_source(manifest)
        if hashlib.sha256(raw).hexdigest()!=source["rust_sha256"]:
            raise ValueError("Whole Rust pages differ from the causally verified source")
        for page in manifest["pages"]:
            filename=page["file"]
            if Path(filename).name!=filename or not filename.endswith(".png") or digest(manifest_path.parent/filename)!=page["sha256"]:
                raise ValueError("Whole-source PNG path or bytes changed")
        result.append({"manifest":manifest,"manifest_path":manifest_path,"source":raw.decode("utf-8"),
                       "proof_path":proof_path,"binding":source})
    return result


def build(data, args, validation):
    if data["schema"] not in {"jain.h256-stage1-capture/v1","jain.h256-stage2-capture/v1"}:
        raise ValueError("H256 scene requires the new saved-encoder capture")
    stage2=data["schema"]=="jain.h256-stage2-capture/v1"
    if stage2 and data["checkpoint"].get("scaler_mode")!="canonical_full_train":
        raise ValueError("Full Stage2 scene requires the original shared full-train scaler")
    cases = data["cases"]
    pages=full_source_pages(data,args,validation);code_mode=pages is not None
    if args.show_gp_tree and not code_mode:
        raise ValueError("GP tree requires the actual causally verified selected program and source proof")
    has_trajectory=stage2 and any(name.endswith(".delta_trajectory") for name in cases[0]["outputs"])
    spans=[];nn_spans=[];cues={};cursor=1
    for i,case in enumerate(cases):
        first=cursor
        if code_mode:
            manifest=pages[i]["manifest"];source_start=first+36
            cursor=source_start+manifest["page_count"]*args.code_hold_frames+6
            nn_spans.append((first,first+35))
            cue={"neural":first+18,"source_first":source_start,
                 "source_last":source_start+(manifest["page_count"]-1)*args.code_hold_frames}
            for name,pattern in (("training",r"(?m)^    fn fit\("),("prediction",r"(?m)^    fn predict\("),("fitted_state",r"(?m)^fn fitted_model\(")):
                match=re.search(pattern,pages[i]["source"])
                if match:
                    page_index=next(index for index,page in enumerate(manifest["pages"])
                                    if page["source_char_start"]<=match.start()<page["source_char_end"])
                    cue[name]=source_start+page_index*args.code_hold_frames
            cues[case["id"]]=cue
        else:
            cursor=first+args.frames_per_case;nn_spans.append((first,cursor-1))
        spans.append((first,cursor-1))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples, scene.cycles.max_bounces = 24, 2
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGB"
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, spans[-1][1]
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    scene.world = bpy.data.worlds.new("H256 deep space"); scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.001, .002, .008, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .15
    camera = bpy.data.objects.new("Continuous encoder camera", bpy.data.cameras.new("Continuous encoder camera"))
    scene.collection.objects.link(camera)
    camera.location = (3.2, -35, 14)
    camera.rotation_euler = (Vector((0., 0., .3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 38. if code_mode else 31.5
    scene.camera = camera
    ink = material("H256 clear typography", (.78, .85, .94, 1), 1)
    muted = material("H256 secondary typography", (.35, .44, .58, 1), .8)
    cyan = material("Problem encoder heading", (.035, .72, .95, 1), 1.2)
    violet = material("Solution encoder heading", (.61, .36, 1, 1), 1.2)
    paper = material("Table backing", (.009, .018, .032, 1), 1)
    rule = material("Table header", (.035, .085, .12, 1), 1)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    mono = bpy.data.fonts.load(str(font_path)) if font_path.exists() else None

    def hud(body, x, y, size=.25, mat=ink, mono_font=False):
        if code_mode and x < -9:x-=3.2
        obj = text(body, (0, 0, 0), size, mat)
        obj.parent, obj.location, obj.rotation_euler = camera, (x, y, -6), (0, 0, 0)
        if mono_font and mono:
            obj.data.font = mono
        return obj

    def tile(name, x, y, width, height, mat):
        if code_mode and x < -9:x-=3.2
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(-width/2,-height/2,0),(width/2,-height/2,0),(width/2,height/2,0),(-width/2,height/2,0)],[],[(0,1,2,3)])
        obj = bpy.data.objects.new(name, mesh); scene.collection.objects.link(obj)
        mesh.materials.append(mat); obj.parent, obj.location = camera, (x, y, -6.05)
        return obj

    def show(obj, first, last):
        for frame, hidden in sorted({1: True, first: False, last + 1: True}.items()):
            obj.hide_render = hidden; obj.keyframe_insert("hide_render", frame=frame)
        interpolation(obj, "CONSTANT")

    hud("JAIN", -14.7, 7.3, .72)
    hud("RECORDED INFERENCE · TIME EXPANDED", 4.1, 7.45, .22, muted)
    hud("problem encoder", -8.4, 5.95, .5, cyan)
    hud("solution encoder", 1.0, 5.95, .5, violet)
    hud("DOPE", -8.4, 5.43, .22, muted)
    hud("SAVED H256 · FIVE FOLD PREDICTION MEAN" if stage2 else "SAVED H256 · STAGE 1", 1.0, 5.43, .19, muted)
    hud("TABULAR INPUT", -14.7, 5.9, .3)
    if code_mode:
        hud("RUST · COMPLETE EMITTED FILE",8.8,6.8,.32)
    else:
        hud("MEASURED RESIDUAL" if stage2 else "MEASURED CONTEXT", 9.1, 5.9, .3)
        hud("92 × 32 tokens · five folds" if stage2 else "92 × 32 tokens", 9.1, 5.34, .24, violet)
        hud("20 Full629 · 8 Hidden256", 9.1, 4.91, .19, muted)
        hud("64 column tokens · 11 → 8 scale", 9.1, 4.55, .19, muted)
    parameter_rows = data.get("parameters", [])
    if isinstance(parameter_rows, dict):
        parameter_rows = [{"id": key, **value} for key,value in parameter_rows.items()]
    used = {pid for case in cases for operation in case["trace"]["affine"] for pid in operation.get("parameter_ids", [])}
    inactive = [row for row in parameter_rows if row.get("executed") is False or
                row.get("id") in data.get("coverage",{}).get("unused_parameter_ids", [])]
    inactive.sort(key=lambda row:(row.get("role")!="solution_encoder",str(row.get("name",row.get("id","")))))
    if inactive and not code_mode:
        hud(f"{len(inactive)} saved tensors unexecuted", 9.1, -3.19, .22, muted)
        hud("Stored weights · no activity inferred", 9.1, -3.62, .18, muted)
        names = [str(row.get("name",row.get("id",""))).removeprefix("h256.") for row in inactive]
        for i, name in enumerate(names[:6]):
            hud(name[:39], 9.1, -4.04-i*.29, .15, muted, True)
    # Custody stamps come from capture metadata, never from a guessed latest file.
    stamps = []
    def checkpoint_stamps(node, path=""):
        if isinstance(node,dict):
            for key,value in node.items():
                child=f"{path}.{key}" if path else key
                if isinstance(value,str) and "checkpoint" in child.lower() and ("sha" in key.lower()) and len(value)==64:
                    stamps.append(value)
                elif key not in {"cases","parameters"}:
                    checkpoint_stamps(value,child)
    checkpoint_stamps(data)
    if stamps:
        hud("checkpoint "+stamps[0][:16], 1.0, 5.03, .18, muted, True)
    if data.get("checkpoint",{}).get("epochs") is not None:
        hud(f"saved epoch {data['checkpoint']['epochs']}", 1.0, 4.71, .18, muted)
    for index, (case, span) in enumerate(zip(cases, spans)):
        objects = [hud(f"{index+1}/{len(cases)}  {case['id'].upper()}", -14.7, 5.25, .28)]
        preview = case.get("table_preview")
        if preview is None:
            raise ValueError("Actual table_preview is required; no fabricated table fallback")
        header = case.get("input_header", ["x0", "x1", "x2", "x3", "y"])
        if len(header) != 5 or len(preview) < 5:
            raise ValueError("First scene supports five actual columns and five preview rows")
        row_ids = case.get("table_preview_row_ids")
        widths = [.45, 1., 1., 1., 1., 1.] if row_ids is not None else [1.]*5
        for row in range(6):
            cells = header if row == 0 else [f"{float(value):.2f}".replace("-", "−") for value in preview[row-1]]
            if row_ids is not None:
                cells = ["row" if row == 0 else str(row_ids[row-1]), *cells]
            for col, value in enumerate(cells):
                x, y = -14.7 + sum(widths[:col]), 4.40 - row * .46
                objects.append(tile(f"Actual cell {index}/{row}/{col}", x + widths[col]/2, y, widths[col]-.02, .44, rule if row == 0 else paper))
                label = hud(str(value), x + widths[col]-.09, y - .095, .22, ink, True)
                label.data.align_x = "RIGHT"; objects.append(label)
        objects += [hud(f"{len(case['fit_row_ids'])} FIT / {len(case['holdout_row_ids'])} HOLD rows", -14.7, 1.2, .23, muted),
                    hud("Synthetic regression",-14.7,1.60,.20,muted),
                    hud("Changing data · fixed learned weights", -14.7, .74, .19, muted)]
        if not code_mode:
            objects += [hud(("Prior + trajectory · ensemble" if has_trajectory else "K=0 prior · ensemble") if stage2 else "Token mean / max · diagnostic", 9.1, .45, .20),
                        hud("Actual candidate score residuals" if stage2 else "32 masked means + 32 maxima", 9.1, .05, .19, muted)]
        for obj in objects:
            show(obj, *span)
    # Measured tensor cells are deliberately distinct from neural glyphs. The
    # saved graph already establishes the destinations; these grids expose the
    # actual landmark coordinates that entered that graph.
    widget_receipts=[]
    def landmark_grid(name,values,columns,title_y,top_y,cell_width,cell_height,destination,
                      source_refs,source_shapes,display_indices,raw_sources,color):
        cells=np.asarray(values,dtype=np.float32)
        if cells.ndim!=2 or cells.shape[0]!=len(cases) or not np.isfinite(cells).all():
            raise ValueError("Invalid measured landmark cells")
        mat,uv,atlas=scalar_material("Measured landmark · "+name,cells,spans,color,power_exponent=.7)
        # Zero has no data glow and no emitted structural base in these grids.
        for node in mat.node_tree.nodes:
            if node.bl_idname=="ShaderNodeEmission" and not node.inputs["Strength"].is_linked:
                node.inputs["Strength"].default_value=0.
        vertices=[];faces=[]
        x_offset=-3.2 if code_mode else 0.
        for index in range(cells.shape[1]):
            x=-14.7+x_offset+(index%columns)*cell_width
            y=top_y-(index//columns)*cell_height
            offset=len(vertices)
            vertices.extend(((x,y,-5.85),(x+cell_width*.87,y,-5.85),
                             (x+cell_width*.87,y-cell_height*.83,-5.85),(x,y-cell_height*.83,-5.85)))
            faces.append((offset,offset+1,offset+2,offset+3))
        mesh=bpy.data.meshes.new("Measured landmark cells · "+name)
        mesh.from_pydata(vertices,[],faces);mesh.update();coordinates=mesh.uv_layers.new(name="Scalar")
        for polygon in mesh.polygons:
            for loop in polygon.loop_indices:coordinates.data[loop].uv=uv[polygon.index]
        mesh.materials.append(mat);obj=bpy.data.objects.new("Measured tensor · "+name,mesh)
        scene.collection.objects.link(obj);obj.parent=camera
        hud(name,-14.7,title_y,.19,ink)
        hud(destination,-14.7,title_y-.28,.15,muted)
        widget_receipts.append({"name":name,"kind":"measured tensor cells, not neural neuron glyphs",
            "source_refs_by_case":source_refs,"source_shapes_by_case":source_shapes,
            "displayed_source_indices_by_case":display_indices,
            "raw_source_f32_sha256_by_case":[hashlib.sha256(np.asarray(value,dtype="<f4").tobytes()).hexdigest() for value in raw_sources],
            "displayed_f32_sha256_by_case":[hashlib.sha256(value.astype("<f4").tobytes()).hexdigest() for value in cells],
            "displayed_cells":int(cells.shape[1]),"columns":columns,"atlas":atlas,
            "zero_policy":"zero emits no light; structural emission disabled",
            "calibration_scope":"fixed signed log calibration across all captured cases; power0.7"})
    recorded_v=[];recorded_col=[];valid_columns=[];vrefs=[];crefs=[]
    for case in cases:
        v=next(tensor for tensor in case["tensors"] if tensor.get("name")=="fit.v256")
        col=next(tensor for tensor in case["tensors"] if tensor.get("name")=="fit.col64x17")
        vv=np.asarray(v["values"],dtype=np.float32).reshape(256)
        cc=np.asarray(col["values"],dtype=np.float32).reshape(64,17)
        if vv.tobytes()!=np.asarray(case["outputs"]["dope629"],dtype=np.float32).reshape(-1)[:256].tobytes():
            raise ValueError("Displayed v256 differs from actual Full629 source")
        mask_name="fold0.context_mask_92" if stage2 else "context_mask_92"
        mask=np.asarray(case["outputs"][mask_name]).reshape(92)[28:]
        recorded_v.append(vv);recorded_col.append(cc);valid_columns.append(np.flatnonzero(mask==1))
        vrefs.append(v["id"]);crefs.append(col["id"])
    landmark_grid("v256 · 256 values",recorded_v,32,.16,-.42,.145,.10,"desc_proj → 8 descriptor tokens",
                  vrefs,[[1,256]]*len(cases),[list(range(256)) for _ in cases],recorded_v,(.025,.70,.95,1))
    visible_columns=min(4,min(len(indices) for indices in valid_columns))
    if visible_columns:
        selected=[indices[:visible_columns] for indices in valid_columns]
        landmark_grid(f"{visible_columns} valid columns × 17 · capacity64",
                      [value[indices].reshape(-1) for value,indices in zip(recorded_col,selected)],17,-1.55,-2.12,.273,.14,
                      "col_encoder.0 → 64 tokens (masked)",crefs,[[1,64,17]]*len(cases),
                      [[int(row*17+column) for row in indices for column in range(17)] for indices in selected],recorded_col,(.025,.70,.95,1))
    else:
        hud("0 valid column landmark rows",-14.7,-1.55,.19,muted)
    dope_values=[np.asarray(case["outputs"]["dope629"],dtype=np.float32).reshape(629) for case in cases]
    landmark_grid("predicted lm23 · 23 values",[value[526:549] for value in dope_values],23,-2.98,-3.49,.20,.14,
                  "heads_proj → semantic head tokens",["outputs.dope629"]*len(cases),[[629]]*len(cases),
                  [list(range(526,549)) for _ in cases],dope_values,(.47,.18,1.,1))
    if all("fresh_descriptor_projection_lm48" in case["outputs"] for case in cases):
        lm48=[np.asarray(case["outputs"]["fresh_descriptor_projection_lm48"],dtype=np.float32).reshape(48) for case in cases]
        landmark_grid("lm48 · 48 values",lm48,16,-3.94,-4.46,.29,.14,"lm48 → parallel base input",
                      ["outputs.fresh_descriptor_projection_lm48"]*len(cases),[[48]]*len(cases),
                      [list(range(48)) for _ in cases],lm48,(.47,.18,1.,1))
    hud("Measured tensor cells · signed relative activity",-14.7,-5.25,.14,muted)
    output_atlas=None
    if not code_mode:
        output_name="ensemble_weighted_residual" if stage2 else "context_summary_64"
        raw_outputs=[np.asarray(case["outputs"][output_name],dtype=np.float32).reshape(-1) for case in cases]
        # Case candidate counts may differ. Separate measured glyph sets avoid
        # padding invented zero candidates into a shared rectangular output.
        output_atlas={"name":output_name,"cases":[]}
        for case,raw,span in zip(cases,raw_outputs,spans):
            displayed_count=min(64,len(raw))
            mat,uv,atlas=scalar_material("Actual "+output_name+case["id"],raw[:displayed_count][None,:],[span],(.58,.30,1.,1),power_exponent=.7)
            # Markers show actual observed outputs, never padded candidates.
            positions=np.asarray([(9.35+(j%8)*.59,3.75-(j//8)*.34,-5.9) for j in range(displayed_count)])
            glyphs=neurons("Measured output coordinates · "+case["id"],positions,uv,mat,.10);glyphs.parent=camera
            show(glyphs,span[0]+min(30,span[1]-span[0]),span[1])
            if stage2:
                note=hud(f"{displayed_count}/{len(raw)} actual candidate residuals",9.1,3.12-(math.ceil(displayed_count/8)-1)*.34,.17,muted)
                show(note,*span)
            output_atlas["cases"].append({"case_id":case["id"],"available_coordinates":len(raw),
                                           "displayed_indices":list(range(displayed_count)),"atlas":atlas})
        hud("Shared full-train scaler" if stage2 else "Stage 1 snapshot · context only",9.1,-1.48,.22,muted)
        hud("Five saved models · prediction mean" if stage2 else "Stage 1 encoder + partial projections",9.1,-1.91,.19,muted)
    receipt = build_network(data, nn_spans, camera, args.all_weights, args.max_connections)
    landmark_paths=[
        {"name":"v256","source":"dope629[0:256]","destination":"desc_proj input[8,32] inside Full629 tokens0..7",
         "coordinates":[{"source":j,"input_token":j//32,"input_channel":j%32} for j in range(256)]},
        {"name":"column landmarks","source":"col64x17 + dope629[565:629] gates",
         "destination":"col_encoder input[64,18], context token slots28..91",
         "coordinates":[{"column":j,"token_slot":28+j,"landmark_input_channels":list(range(17)),"gate_input_channel":17} for j in range(64)]},
        {"name":"predicted landmarks23","source":"dope629[526:549]","destination":"heads_proj rawsemantic inputs then Full629 head token slots16,17",
         "coordinates":[{"source":526+j,"input_token":16+(14+j)//32,"input_channel":(14+j)%32} for j in range(23)]}]
    for path,start,end in ((landmark_paths[0],0,256),(landmark_paths[2],526,549)):
        path["source_f32_sha256_by_case"]=[hashlib.sha256(np.asarray(case["outputs"]["dope629"],dtype="<f4").reshape(-1)[start:end].tobytes()).hexdigest() for case in cases]
        path["transform"]="recorded saved scaler and learned projection; coordinates identify projection inputs, not equal-valued outputs"
    landmark_paths[1]["gate_f32_sha256_by_case"]=[hashlib.sha256(np.asarray(case["outputs"]["dope629"],dtype="<f4").reshape(-1)[565:629].tobytes()).hexdigest() for case in cases]
    landmark_paths[1]["column17_f32_sha256_by_case"]=[]
    for case in cases:
        tensor=next((tensor for tensor in case["tensors"] if tensor.get("name")=="fit.col64x17"),None)
        if tensor is None:raise ValueError("Landmark coordinate ports require the actual recorded 64x17 input")
        landmark_paths[1]["column17_f32_sha256_by_case"].append(hashlib.sha256(np.asarray(tensor["values"],dtype="<f4").tobytes()).hexdigest())
    hud("v256 [0:256] → 8 descriptor tokens",-8.4,-4.42,.21,muted)
    hud("64×17 column landmarks + 64 gates → 64×18 → 64 tokens",-8.4,-4.82,.19,muted)
    hud("Predicted landmarks23 [526:549] → semantic head tokens",-8.4,-5.22,.19,muted)
    legacy=all(any(name in case["outputs"] for name in ("lm48","base.lm48","fresh_descriptor_projection_lm48","fresh_lm48_z")) and
               any(name in case["outputs"] for name in ("base_tokens_10x32","base.tokens_10x32")) and
               any(any(part in op["name"] for part in ("anchor.","mm_cross_attn")) for op in case["trace"]["affine"]) for case in cases)
    if legacy:
        hud("lm48 → recorded legacy 10-token ranker path · parallel",-8.4,-5.62,.19,muted)
        landmark_paths.append({"name":"lm48","destination":"actually captured parallel legacy base path","displayed":True,
                               "readouts":[name for name in ("fresh_descriptor_projection_lm48","fresh_lm48_z","base_tokens_10x32","base_mask_10") if name in cases[0]["outputs"]],
                               "source_f32_sha256_by_case":[hashlib.sha256(np.asarray(next(case["outputs"][name] for name in ("fresh_descriptor_projection_lm48","lm48","base.lm48","fresh_lm48_z") if name in case["outputs"]),dtype="<f4").tobytes()).hexdigest() for case in cases]})
    else:
        landmark_paths.append({"name":"lm48","displayed":False,"reason":"parallel legacy base network is not captured; no invented route"})
    hud(f"{receipt['drawn_nonzero_connections']:,} actual scalar connections",-8.4,-6.30,.25)
    hud("Width: |weight| · firing: signed relative |weight × input|",-8.4,-6.75,.20,muted)
    hud("Recorded weights · token-row aggregates · neutral functional bundles",-8.4,-7.13,.18,muted)
    hud("Saved Stage 2 inference" if stage2 else "Saved Stage 1 encoder inference",-14.7,-10.0 if args.show_gp_tree else -7.95,.22,muted)
    page_receipts=[]
    if code_mode:
        for case,info,(first,last) in zip(cases,pages,spans):
            manifest=info["manifest"];source_start=cues[case["id"]]["source_first"]
            width=9.4;height=width*manifest["height"]/manifest["width"]
            for index,page in enumerate(manifest["pages"]):
                image=bpy.data.images.load(str(info["manifest_path"].parent/page["file"]),check_existing=True)
                if tuple(image.size)!=(manifest["width"],manifest["height"]):raise ValueError("Whole-source image dimensions differ")
                image.pack();mat=bpy.data.materials.new(f"Whole new-model Rust {case['id']}/{index}");mat.use_nodes=True
                tree=mat.node_tree;tree.nodes.clear();texture=tree.nodes.new("ShaderNodeTexImage");texture.image=image;texture.interpolation="Closest"
                emit=tree.nodes.new("ShaderNodeEmission");emit.inputs["Strength"].default_value=1.
                out=tree.nodes.new("ShaderNodeOutputMaterial");tree.links.new(texture.outputs["Color"],emit.inputs["Color"]);tree.links.new(emit.outputs[0],out.inputs["Surface"])
                mesh=bpy.data.meshes.new("Complete Rust page");mesh.from_pydata([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,.5,0)],[],[(0,1,2,3)])
                uv=mesh.uv_layers.new(name="UVMap")
                for loop,coordinate in zip(uv.data,((0,0),(1,0),(1,1),(0,1))):loop.uv=coordinate
                obj=bpy.data.objects.new(f"Complete Rust {case['id']} page{index+1}",mesh);scene.collection.objects.link(obj);mesh.materials.append(mat)
                obj.parent=camera;obj.location=(8.8+width/2,6.4-height/2,-5.8);obj.scale=(width,height,1.)
                begin=source_start+index*args.code_hold_frames
                finish=begin+args.code_hold_frames-1 if index+1<len(manifest["pages"]) else last
                show(obj,begin,finish)
            for obj in (hud(f"ALL {manifest['source_lines']} lines · {manifest['source_bytes']:,} bytes",8.8,-3.5,.23,muted),
                        hud("Verified new-model selection → native GP → Rust",8.8,-3.95,.20,muted)):
                show(obj,source_start,last)
            verdict=validation["gp_sources"][case["id"]]
            badge=hud(f"{verdict['prediction_rows']} source rows · max |Δ| {verdict['max_abs_error']:.2e}",8.8,-4.4,.21,muted)
            show(badge,source_start,last)
            rate=verdict.get("rate_rust_lexical_tokens_per_s")
            if isinstance(rate,(int,float)) and math.isfinite(rate) and rate>0:
                labels=[hud(f"{rate/1e6:.2f} M Rust lexical tokens/s",8.8,-4.95,.26),
                        hud("Release · warm repeated Rust source emission",8.8,-5.40,.18,muted),
                        hud("lexical tokens/s · allocation + loop included",8.8,-5.74,.17,muted)]
                for label in labels:show(label,source_start,last)
            page_receipts.append({"case_id":case["id"],"manifest_sha256":digest(info["manifest_path"]),
                                  "rust_sha256":info["binding"]["rust_sha256"],"proof_sha256":info["binding"]["proof_sha256"],
                                  "page_count":manifest["page_count"],"all_original_bytes_reconstructed":True,
                                  "verified_rate_rust_lexical_tokens_per_s":rate,
                                  "emit_metric_sha256":info["binding"].get("emit_metric_sha256"),
                                  "frames":[source_start,last]})
    gp_trees=[]
    if args.show_gp_tree:
        for case,(first,last) in zip(cases,spans):
            source=case["source"];program=source["selected_program"]
            if hashlib.sha256(program.encode()).hexdigest()!=source["selected_program_sha256"]:
                raise ValueError("GP tree differs from the actually selected program")
            tree=gp_sexpression(program);objects=[]
            objects.append(hud("GP TREE · TEMPORARY OPERATOR PREVIEW",-8.4,-7.40,.24))
            objects.append(hud("Actual selected program · parameter counts; full syntax retained",-8.4,-7.76,.16,muted))
            def weight(node):
                return max(1.,len(str(node[0]))*.35) if isinstance(node,list) else 1.
            def draw(node,left,right,depth,parent=None):
                x=(left+right)/2;y=-8.12-depth*.76
                name=str(node[0]) if isinstance(node,list) else str(node)
                obj=hud(name,x,y,.21,ink,True);obj.data.align_x="CENTER";objects.append(obj)
                if parent is not None:
                    edge=line("Literal GP operator parent-child",[(parent[0],parent[1]-.08,-6.02),(x,y+.25,-6.02)],.008,muted)
                    edge.parent=camera;objects.append(edge)
                if isinstance(node,list):
                    children=[child for child in node[1:] if isinstance(child,list) or child=="X"]
                    parameters=[child for child in node[1:] if isinstance(child,str) and child!="X"]
                    if parameters:
                        note=hud(f"{len(parameters)} params",x,y-.30,.15,muted,True);note.data.align_x="CENTER";objects.append(note)
                    denominator=sum(weight(child) for child in children);position=left
                    for child in children:
                        width=(right-left)*weight(child)/max(1.,denominator)
                        draw(child,position,position+width,depth+1,(x,y));position+=width
            draw(tree,-8.4,7.8,0)
            for obj in objects:show(obj,cues[case["id"]]["source_first"],last)
            gp_trees.append({"case_id":case["id"],"selected_program_sha256":source["selected_program_sha256"],
                             "literal_tree":tree,"projection":"actual operator parent-child tree with parameter atom counts"})
    compositor = bpy.data.node_groups.new("H256 measured signal glow", "CompositorNodeTree")
    compositor.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    layer = compositor.nodes.new("CompositorNodeRLayers")
    glow = compositor.nodes.new("CompositorNodeGlare")
    glow.inputs["Type"].default_value = "Bloom"; glow.inputs["Threshold"].default_value = 1.6
    glow.inputs["Strength"].default_value = .22
    final = compositor.nodes.new("NodeGroupOutput")
    compositor.links.new(layer.outputs["Image"], glow.inputs["Image"])
    compositor.links.new(glow.outputs["Image"], final.inputs[0]); scene.compositing_node_group = compositor
    parameter_inventory = [{key:value for key,value in row.items() if key in {"id","name","shape","dtype","sha256","origin","role","source_sha256","executed"}}
                           for row in parameter_rows]
    result = {"schema": "jain.h256-scene-receipt/v1", "trace_sha256": digest(args.trace),
              "capture_schema": data["schema"], "capture_metadata": {k:v for k,v in data.items() if k not in {"cases","parameters"}},
              "parameter_inventory":parameter_inventory,
              "unused_parameter_ids":data.get("coverage",{}).get("unused_parameter_ids",[]),
              "renderer_source_sha256": {Path(__file__).name: digest(__file__), "h256_network.py": digest(HERE/"h256_network.py")},
              "validation": validation, "network": receipt, "case_frames": spans,"nn_frames":nn_spans,"source_cues":cues,
              "landmark_paths":landmark_paths,"whole_source_pages":page_receipts,
              "measured_landmark_widgets":widget_receipts,
              "gp_operator_trees":gp_trees,
              "context_output_atlas": output_atlas, "blender_version": bpy.app.version_string,
              "scope":data.get("inference_scope"),
              "source": "root-validated causal whole-source pages" if code_mode else None,"tokens_per_second": None,"video_rendered":False}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    bpy.data.texts.new("H256_PRIVATE_RECEIPT.json").write(json.dumps(result, indent=2))
    default_frames=([cues[cases[0]["id"]]["source_first"],
                     cues[cases[len(cases)//2]["id"]].get("training",cues[cases[len(cases)//2]["id"]]["source_first"]),
                     cues[cases[-1]["id"]].get("prediction",cues[cases[-1]["id"]]["source_last"])] if code_mode else
                    [min(spans[0][1],18),spans[len(spans)//2][0]+35,spans[-1][0]+55])
    frames=args.render_frame or default_frames
    scene.frame_set(frames[0]); bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    for frame in frames:
        if not scene.frame_start <= frame <= scene.frame_end:
            raise ValueError("Review frame lies outside captured cases")
        scene.frame_set(frame); scene.render.filepath = str(destination.parent/f"h256-{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
    print(json.dumps({"blend": str(destination), "case_frames": spans, "stills": frames,
                      "connections": receipt["drawn_nonzero_connections"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--saved-validation", type=Path, required=True)
    parser.add_argument("--code-pages",type=Path)
    parser.add_argument("--gp-proof",type=Path)
    parser.add_argument("--code-hold-frames",type=int,default=6)
    parser.add_argument("--show-gp-tree",action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--frames-per-case", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--all-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-connections", type=int, default=600000)
    parser.add_argument("--render-frame", type=int, action="append", default=[])
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if args.frames_per_case < 64 or args.max_connections < 1 or args.code_hold_frames<1:
        parser.error("at least64 frames per case and a positive explicit connection budget required")
    schema=json.loads(args.trace.read_text())["schema"]
    if schema=="jain.h256-stage2-capture/v1":
        from h256_stage2_validate import validate
        data,validation=validate(args.trace,saved_receipt=args.saved_validation,gp_root=args.gp_proof)
    else:
        from h256_validate import validate
        data,validation=validate(args.trace,saved_receipt=args.saved_validation)
    build(data, args, validation)


if __name__ == "__main__":
    main()
