import os
import json
import argparse
import re
from collections import defaultdict
import numpy as np

# --- Stricter metric functions ---
def parse_material_properties(text: str):
    """Parse material properties from annotation text. Returns dict or None."""
    patterns = {
        'atomic_count': (r"Atomic Count:\s*(\d+)", int),
        'cell_volume': (r"Cell Volume:\s*([\d.]+)\s*Å³", float),
        'lattice_a': (r"Lattice Parameters:\s*a\s*=\s*([\d.]+)\s*Å", float),
        'lattice_b': (r"Lattice Parameters:.*b\s*=\s*([\d.]+)\s*Å", float),
        'lattice_c': (r"Lattice Parameters:.*c\s*=\s*([\d.]+)\s*Å", float),
        'nn_distance': (r"Average NN Distance:\s*([\d.]+)\s*Å", float),
        'density': (r"Density:\s*([\d.]+)\s*g/cm³", float),
        'primitive_a': (r"Primitive Cell Parameters:\s*a\s*=\s*([\d.]+)\s*Å", float),
        'primitive_b': (r"Primitive Cell Parameters:.*b\s*=\s*([\d.]+)\s*Å", float),
        'primitive_c': (r"Primitive Cell Parameters:.*c\s*=\s*([\d.]+)\s*Å", float),
        'primitive_alpha': (r"Primitive Cell Angles:\s*α\s*=\s*([\d.]+)°", float),
        'primitive_beta': (r"Primitive Cell Angles:.*β\s*=\s*([\d.]+)°", float),
        'primitive_gamma': (r"Primitive Cell Angles:.*γ\s*=\s*([\d.]+)°", float),
        'space_group': (r"Space Group:\s*([^'\n]+)", str)
    }
    values = {}
    for key, (pattern, conv) in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                values[key] = conv(match.group(1))
            except Exception:
                values[key] = None
        else:
            values[key] = None
    # Required fields
    required = ['atomic_count', 'cell_volume', 'lattice_a', 'lattice_b', 'lattice_c', 'nn_distance', 'density']
    if any(values[k] is None for k in required):
        return None
    return values

def physical_law_compliance(ref, gen):
    """User rule: ≤10% deviation=1.0, >10% and ≤25%=0.5, >25%=0.0, average over all checks."""
    if ref is None or gen is None:
        return 0.0
    scores = []
    # Density
    try:
        if ref['density'] > 0 and gen['density'] > 0:
            rel = abs(gen['density'] - ref['density']) / ref['density']
            if rel <= 0.10:
                scores.append(1.0)
            elif rel <= 0.25:
                scores.append(0.5)
            else:
                scores.append(0.0)
    except Exception:
        scores.append(0.0)
    # Lattice ratios
    try:
        for p1, p2 in [('lattice_b', 'lattice_a'), ('lattice_c', 'lattice_a')]:
            if ref[p2] > 0 and gen[p2] > 0:
                ref_ratio = ref[p1] / ref[p2]
                gen_ratio = gen[p1] / gen[p2]
                rel = abs(gen_ratio - ref_ratio) / ref_ratio if ref_ratio != 0 else 1.0
                if rel <= 0.10:
                    scores.append(1.0)
                elif rel <= 0.25:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
    except Exception:
        scores.append(0.0)
    # Primitive cell ratios
    try:
        if all(ref.get(f'primitive_{p}') for p in ['a','b','c']) and all(gen.get(f'primitive_{p}') for p in ['a','b','c']):
            for p1, p2 in [('primitive_b', 'primitive_a'), ('primitive_c', 'primitive_a')]:
                if ref[p2] > 0 and gen[p2] > 0:
                    ref_ratio = ref[p1] / ref[p2]
                    gen_ratio = gen[p1] / gen[p2]
                    rel = abs(gen_ratio - ref_ratio) / ref_ratio if ref_ratio != 0 else 1.0
                    if rel <= 0.10:
                        scores.append(1.0)
                    elif rel <= 0.25:
                        scores.append(0.5)
                    else:
                        scores.append(0.0)
    except Exception:
        scores.append(0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def hallucination_score(ref, gen):
    """User rule: impossible (≤0) or >25% error=1.0, >10% and ≤25%=0.5, ≤10%=0.0, average over all checked properties."""
    if ref is None or gen is None:
        return 1.0
    scores = []
    for k in ['density','cell_volume','atomic_count','lattice_a','lattice_b','lattice_c','nn_distance','primitive_a','primitive_b','primitive_c']:
        gv = gen.get(k)
        rv = ref.get(k)
        if gv is not None and isinstance(gv, (int, float)) and gv <= 0:
            scores.append(1.0)
            continue
        if rv is not None and gv is not None and isinstance(rv, (int, float)) and rv != 0:
            rel = abs(gv - rv) / abs(rv)
            if rel <= 0.10:
                scores.append(0.0)
            elif rel <= 0.25:
                scores.append(0.5)
            else:
                scores.append(1.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))

# --- File utilities ---
def find_json_files(root_dir, prefix):
    """Recursively find all result JSONs with given prefix in root_dir."""
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.startswith(prefix) and f.endswith('.json'):
                result.append(os.path.join(dirpath, f))
    return result

# --- Main aggregation ---
def aggregate_metrics(json_files, mode):
    """Aggregate over all models and 5 runs per material per R."""
    # Dict: (material, R) -> list of (phys, halluc)
    scores = defaultdict(list)
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        # Try to find the results list
        if 'results' in data:
            results = data['results']
        elif 'outputs' in data:
            # LORO/LOMO model_outputs format
            results = []
            for mat in data['outputs']:
                m = mat['material']
                for r, rdata in mat['r_values'].items():
                    for rot, rotdata in rdata['rotations'].items():
                        results.append({
                            'material': m,
                            'held_out_r': r,
                            'rotation_index': int(rot),
                            'reference_annotation': rotdata['reference_annotation'],
                            'generated_annotation': rotdata['generated_annotation'],
                        })
        else:
            continue
        for res in results:
            mat = res.get('material')
            r = res.get('held_out_r') or res.get('r_value')
            # Load annotation texts
            ref = res.get('reference_annotation')
            gen = res.get('generated_annotation')
            # If these are file paths, load the file
            if ref and os.path.isfile(ref):
                with open(ref, 'r') as f:
                    ref = f.read()
            if gen and os.path.isfile(gen):
                with open(gen, 'r') as f:
                    gen = f.read()
            ref_props = parse_material_properties(ref) if ref else None
            gen_props = parse_material_properties(gen) if gen else None
            phys = physical_law_compliance(ref_props, gen_props)
            halluc = hallucination_score(ref_props, gen_props)
            scores[(mat, r)].append({'physical_law_compliance': phys, 'hallucination_score': halluc})
    # Average over all models and 5 runs
    agg = {}
    for (mat, r), vals in scores.items():
        phys = [v['physical_law_compliance'] for v in vals]
        halluc = [v['hallucination_score'] for v in vals]
        agg[(mat, r)] = {
            'material': mat,
            'R': r,
            'n': len(vals),
            'physical_law_compliance_mean': float(np.mean(phys)) if phys else None,
            'physical_law_compliance_std': float(np.std(phys)) if phys else None,
            'hallucination_score_mean': float(np.mean(halluc)) if halluc else None,
            'hallucination_score_std': float(np.std(halluc)) if halluc else None,
        }
    return agg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute stricter hallucination and physical law compliance metrics for LOMO and LORO results.")
    parser.add_argument('--lomo_dir', type=str, required=False, default="lomo_annotation_results", help='Path to LOMO results directory')
    parser.add_argument('--loro_dir', type=str, required=False, default="loro_annotation_results", help='Path to LORO results directory')
    parser.add_argument('--output_dir', type=str, required=False, default="metric_results", help='Where to save new aggregated results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # LOMO
    lomo_jsons = find_json_files(args.lomo_dir, 'lomo_annotations_')
    lomo_agg = aggregate_metrics(lomo_jsons, mode='lomo')
    with open(os.path.join(args.output_dir, 'lomo_strict_metrics.json'), 'w') as f:
        json.dump(list(lomo_agg.values()), f, indent=2)

    # LORO
    loro_jsons = find_json_files(args.loro_dir, 'loro_annotations_')
    loro_agg = aggregate_metrics(loro_jsons, mode='loro')
    with open(os.path.join(args.output_dir, 'loro_strict_metrics.json'), 'w') as f:
        json.dump(list(loro_agg.values()), f, indent=2)

    print(f"Done. Results saved in {args.output_dir}") 