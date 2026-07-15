
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import argparse
import inspect
import matplotlib.pyplot as plt
import json
import numpy
import os
import ovito
from ovito.io import import_file, export_file
import shlex
import sys
import uuid
import os
import gsd.hoomd
import time
from ovito.modifiers import *
from ovito.vis import *
import numpy as np

def wait_until_ready(output_file_traj, max_attempts=10, delay=5):
    for attempt in range(max_attempts):
        try:
            pipeline = import_file(output_file_traj)
            frame_count = pipeline.source.num_frames
            print(f"Attempt {attempt+1}: Found {frame_count} frame(s).")
            if frame_count > 0:
                return pipeline
        except Exception as e:
            print(f"Attempt {attempt+1}: Error reading file {output_file_traj} - {e}")
        time.sleep(delay)
    raise RuntimeError("File not ready after multiple attempts.")

def CNA_Classification(input_path_full, parameters):
    # parser = argparse.ArgumentParser(description="performs common neighbor analysis on two-dimensional binary systems to identify and classify crystals")
    # parser.add_argument("-i", "--input-path", action="store", type=str, default=None, help="path to the input file")
    # parser.add_argument("--f",dest="input_file",type=str)
    # arguments = parser.parse_args()
    input_path = input_path_full

    # Define constants
    DEFAULT_CONF = 'crystal.conf'
    SS = "ss"  # for amorphous or unclassified structures
    SSS_bin = "single_strand_strip_bin"
    
    BTr = 'binary_tr'
    BTr_bin = 'binary_tr_bin'
    SL_bin = 'sl_bin'
    OHC_bin = 'ohc_bin'

    # Converts labels into set/tuple format for faster lookup
    def convert_labels(labels):

        return {tuple(label) for label in labels}

    # Read configuration file
    with open(DEFAULT_CONF, "r") as configuration_file:
        labels = json.load(configuration_file)
        conv = convert_labels
     
        
        label_btr_bin, label_sss_bin, label_sl_bin, label_ohc_bin = conv(labels.get(BTr_bin, [])), conv(
            labels.get(SSS_bin, [])), conv(labels.get(SL_bin, [])), conv(
            labels.get(OHC_bin, []))  # Binary classification for binary triangles
        structure_factors_cfg = labels.get("structure_factors", {})
        # labels_hl = conv(labels[HL])

    d = 1

    # Per-structure hyperparameters (each structure has both overall and binary factor).
    structure_factors = {
        SL_bin: {
            "overall": structure_factors_cfg.get("sl", {}).get("factor_overall", 2.8),
            "binary": structure_factors_cfg.get("sl", {}).get("factor_binary", 2.8),
        },
        OHC_bin: {
            "overall": structure_factors_cfg.get("ohc", {}).get("factor_overall", 2.8),
            "binary": structure_factors_cfg.get("ohc", {}).get("factor_binary", 2.8),
        },
        BTr_bin: {
            "overall": structure_factors_cfg.get("btr", {}).get("factor_overall", 2.56),
            "binary": structure_factors_cfg.get("btr", {}).get("factor_binary", 2.56),
        },
        SSS_bin: {
            "overall": structure_factors_cfg.get("sss", {}).get("factor_overall", 1.22),
            "binary": structure_factors_cfg.get("sss", {}).get("factor_binary", 1.22),
        }
    }

    # Histogram construction
    def row_histogram(a):
        if len(a) == 0:
            return numpy.array([]).reshape(0, 3), numpy.array([])
        ca = numpy.ascontiguousarray(a).view([('', a.dtype)] * a.shape[1])
        unique, indices, inverse = numpy.unique(ca, return_index=True, return_inverse=True)
        counts = numpy.bincount(inverse)
        return a[indices], counts

  
    # Binary structure identification
    def identify_binary(lab):
        if lab in label_btr_bin: return BTr_bin
        if lab in label_sss_bin: return SSS_bin
        if lab in label_sl_bin: return SL_bin
        if lab in label_ohc_bin: return OHC_bin
        return SS

    # Load the input data
    pipeline = import_file(input_path)

    # frame_count = pipeline.source.num_frames

    # Create simulation cell modifier
    def cell_modifier(frame, data):
        coor_origin = data.cell[0, 3]
        data.cell_[2, :] = numpy.array([0, 0, -2 * coor_origin, coor_origin])
        data.cell_.is2D = False
        data.cell_.pbc = (True, True, False)

    # Determine the number of particles
    data = pipeline.compute()
    N = data.particles.count

    flag = 0

    # Binary structure counters (one list per structure, appended each frame)
    NBTr_bin_ = []
    NSSS_bin_ = []
    NSL_bin_ = []
    NOHC_bin_ = []

    # Each structure uses its own (factor_overall, factor_binary) pair from structure_factors.
    # Classification: binary CNA with factor_binary cutoff matched against the crystal.conf signature.
    # factor_overall is stored per-structure for reference / future overall-signature matching.
    def build_binary_labels_for_cutoff(cutoff_value, frame_index):
        while len(pipeline.modifiers):
            del pipeline.modifiers[-1]
        pipeline.modifiers.append(ovito.modifiers.PythonScriptModifier(function=cell_modifier))
        pipeline.modifiers.append(ovito.modifiers.CreateBondsModifier(cutoff=cutoff_value))
        pipeline.modifiers.append(ovito.modifiers.CommonNeighborAnalysisModifier(
            mode=ovito.modifiers.CommonNeighborAnalysisModifier.Mode.BondBased))

        data_binary_local = pipeline.compute(frame_index)
        cna_indices_binary_local = data_binary_local.particles.bonds["CNA Indices"].array
        bond_pairs_binary_local = data_binary_local.particles.bonds['Topology'].array
        bond_enumerator_binary_local = ovito.data.BondsEnumerator(data_binary_local.particles.bonds)
        types_local = data_binary_local.particles.particle_types

        labels_local = []
        for particle_index in range(N):
            particle_type = int(types_local[particle_index])
            bond_indices = list(bond_enumerator_binary_local.bonds_of_particle(particle_index))
            bonds_to_A = []
            bonds_to_B = []

            for bond_idx in bond_indices:
                bond = bond_pairs_binary_local[bond_idx]
                neighbor_idx = bond[1] if bond[0] == particle_index else bond[0]
                neighbor_type = int(types_local[neighbor_idx])

                if neighbor_type == 0:
                    bonds_to_A.append(bond_idx)
                else:
                    bonds_to_B.append(bond_idx)

            if len(bonds_to_A) > 0:
                cna_A_indices = cna_indices_binary_local[bonds_to_A]
                unique_A, counts_A = row_histogram(cna_A_indices)
            else:
                unique_A, counts_A = numpy.array([]).reshape(0, 3), numpy.array([])

            if len(bonds_to_B) > 0:
                cna_B_indices = cna_indices_binary_local[bonds_to_B]
                unique_B, counts_B = row_histogram(cna_B_indices)
            else:
                unique_B, counts_B = numpy.array([]).reshape(0, 3), numpy.array([])

            label_parts = [particle_type, len(bonds_to_A)]

            if len(unique_A) > 0:
                zipexpr_A = zip(unique_A, ((x,) for x in counts_A))
                for z in zipexpr_A:
                    for y in z:
                        for x in y:
                            label_parts.append(x)

            label_parts.append(len(bonds_to_B))

            if len(unique_B) > 0:
                zipexpr_B = zip(unique_B, ((x,) for x in counts_B))
                for z in zipexpr_B:
                    for y in z:
                        for x in y:
                            label_parts.append(x)

            labels_local.append(tuple(label_parts))
        return labels_local

    try:
        for frame in range(pipeline.source.num_frames):
            NSL_bin = 0
            NOHC_bin = 0
            NSSS_bin = 0
            NBTr_bin = 0
            Ntotal = N

            print(f"Processing frame {frame + 1}/{pipeline.source.num_frames}...")

            # Each structure is classified independently with its own factor pair and signature.
            # A CNA failure for one structure does not affect the others — it contributes 0.
            try:
                binary_labels_sl = build_binary_labels_for_cutoff(structure_factors[SL_bin]["binary"] * d, frame)
                NSL_bin = sum(1 for label in binary_labels_sl if label in label_sl_bin)
            except Exception as e:
                print(f"[WARNING] Frame {frame}: SL CNA failed: {e}")

            try:
                binary_labels_ohc = build_binary_labels_for_cutoff(structure_factors[OHC_bin]["binary"] * d, frame)
                NOHC_bin = sum(1 for label in binary_labels_ohc if label in label_ohc_bin)
            except Exception as e:
                print(f"[WARNING] Frame {frame}: OHC CNA failed: {e}")

            try:
                binary_labels_sss = build_binary_labels_for_cutoff(structure_factors[SSS_bin]["binary"] * d, frame)
                NSSS_bin = sum(1 for label in binary_labels_sss if label in label_sss_bin)
            except Exception as e:
                print(f"[WARNING] Frame {frame}: SSS CNA failed: {e}")

            try:
                binary_labels_btr = build_binary_labels_for_cutoff(structure_factors[BTr_bin]["binary"] * d, frame)
                NBTr_bin = sum(1 for label in binary_labels_btr if label in label_btr_bin)
            except Exception as e:
                print(f"[WARNING] Frame {frame}: BTr CNA failed: {e}")

            NBTr_bin_.append(NBTr_bin / Ntotal)
            NSSS_bin_.append(NSSS_bin / Ntotal)
            NSL_bin_.append(NSL_bin / Ntotal)
            NOHC_bin_.append(NOHC_bin / Ntotal)

        # Order: [SL, OHC, SSS, BTr]
        x_pairs = numpy.column_stack((NSL_bin_, NOHC_bin_, NSSS_bin_, NBTr_bin_))
    except Exception as e:
        print(f"[ERROR] An error occurred inside CNA_Classification: {e}")
        x_pairs = numpy.zeros((1, 4))  # 4 structure types: SL, OHC, SSS, BTr
        flag = 1
        print("[ERROR] Returning default zero array due to critical error")

    # Normalize by goal values if available
    goal_dic_vector = list(parameters.goal_dic.values())
    if len(goal_dic_vector) == x_pairs.shape[1]:
        x_ = x_pairs / goal_dic_vector  # Normalize each column by its goal value
    else:
        # If goal_dic doesn't match, return unnormalized
        print(f"[WARNING] goal_dic has {len(goal_dic_vector)} entries but x_pairs has {x_pairs.shape[1]} columns. Returning unnormalized.")
        x_ = x_pairs
        
    return x_, flag



def PTM_Classification(state_history, output_file_traj, parameters):
    structure_index = parameters.str_index
    print("We're in!")
    pipeline = wait_until_ready(output_file_traj, max_attempts=12, delay=5)

    modifier = PolyhedralTemplateMatchingModifier()
    modifier.rmsd_cutoff = 0.15
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = False
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
    pipeline.modifiers.append(modifier)
    flag = 0

    structure_to_key = {
        'FCC':      'FCC',
        'HCP':      'HCP',
        'BCC':      'BCC',
        'ICO':      'ICO',
        'SC':       'SC',
        'Cub_Diam': 'CUBIC_DIAMOND',
        'Hex_Diam': 'HEX_DIAMOND',
    }

    structure_lists = {name: [] for name in parameters.goal_dic.keys()}

    for frame_index in range(pipeline.source.num_frames):
        try: 
            data = pipeline.compute(frame_index)
            counts = data.attributes
            Ntotal = data.particles.count

            cubic_diam_count = counts.get('PolyhedralTemplateMatching.counts.CUBIC_DIAMOND', 0)
            hex_diam_count   = counts.get('PolyhedralTemplateMatching.counts.HEX_DIAMOND', 0)
            combined_diamond_count = cubic_diam_count + hex_diam_count

            for struct_name in parameters.goal_dic.keys():
                if struct_name in ['Cub_Diam', 'Hex_Diam']:
                    count = combined_diamond_count
                else:
                    count_key = structure_to_key.get(struct_name, struct_name.upper())
                    count = counts.get(f'PolyhedralTemplateMatching.counts.{count_key}', 0)
                fraction = count / Ntotal if Ntotal > 0 else 0
                structure_lists[struct_name].append(fraction)

        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
            for struct_name in parameters.goal_dic.keys():
                structure_lists[struct_name].append(0)

    columns = [structure_lists[name] for name in parameters.goal_dic.keys()]
    x_pairs = np.column_stack(columns)

    goal_dic_vector = list(parameters.goal_dic.values())
    x_ = x_pairs / goal_dic_vector

    return x_, flag




