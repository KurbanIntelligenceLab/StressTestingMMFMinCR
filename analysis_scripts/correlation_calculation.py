import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# ─── DEFAULT ROOTS (edit if you move the folders) ──────────────────────────
DEFAULT_LORO = Path("loro_annotation_results")
DEFAULT_LOMO = Path("lomo_annotation_results")

# ─── property whitelist (14 canonical error metrics) ──────────────────────
PHYS_PROPS = [
    "atomic_count", "cell_volume",
    "lattice_a", "lattice_b", "lattice_c",
    "nn_distance", "density",
    "primitive_a", "primitive_b", "primitive_c",
    "primitive_alpha", "primitive_beta", "primitive_gamma",
    "average"
]

LATEX = {
    "atomic_count"  : r"N_\mathrm{atoms}",
    "cell_volume"   : r"V",
    "lattice_a"     : r"a",
    "lattice_b"     : r"b",
    "lattice_c"     : r"c",
    "density"       : r"\rho",
    "nn_distance"   : r"\text{NN}",
    "primitive_a"   : r"a_p",
    "primitive_b"   : r"b_p",
    "primitive_c"   : r"c_p",
    "primitive_alpha": r"\alpha_p",
    "primitive_beta" : r"\beta_p",
    "primitive_gamma": r"\gamma_p",
    "average"       : r"\bar\varepsilon",
}

# ─── helpers ───────────────────────────────────────────────────────────────
def collect(root: Path) -> tuple[np.ndarray, int]:
    """Return (sum_matrix, count) over all correlation_data.json below root."""
    files = list(root.rglob("correlation_data.json"))
    if not files:
        print(f"⚠  no correlation_data.json under {root}")
        return np.zeros((len(PHYS_PROPS),)*2), 0

    acc = np.zeros((len(PHYS_PROPS),)*2)
    n = 0
    for fp in files:
        with open(fp) as f:
            data = json.load(f)

        props = data["property_names"]
        mat   = np.asarray(data["correlation_matrix"], dtype=float)

        # keep only the 14 defined props (LoMO has extra "*_abs" columns)
        try:
            idx = [props.index(p) for p in PHYS_PROPS]
        except ValueError:
            # if any canonical prop is missing we skip this file
            continue

        M = mat[np.ix_(idx, idx)]
        if M.shape == (14, 14):
            acc += M
            n  += 1
    return acc, n

def format_val(val):
    return f"\\pos{{{val:+.2f}}}" if val >= 0 else f"\\nego{{{val:+.2f}}}"

# ─── main ──────────────────────────────────────────────────────────────────
if len(sys.argv) == 3:
    root_loro = Path(sys.argv[1]).expanduser()
    root_lomo = Path(sys.argv[2]).expanduser()
else:
    print("ℹ  Using default paths (no command-line arguments provided).")
    root_loro, root_lomo = DEFAULT_LORO, DEFAULT_LOMO

sum_loro, n_loro = collect(root_loro)
sum_lomo, n_lomo = collect(root_lomo)

if n_loro == 0 or n_lomo == 0:
    sys.exit("❌ could not aggregate correlation matrices — check the paths.")

print(f"✓ aggregated {n_loro} LoRO matrices   from {root_loro}")
print(f"✓ aggregated {n_lomo} LoMO matrices   from {root_lomo}")

avg_loro = sum_loro / n_loro
avg_lomo = sum_lomo / n_lomo
delta    = avg_lomo - avg_loro

# build tidy DataFrame of pairwise deltas (upper triangle)
rows = []
for i in range(len(PHYS_PROPS)):
    for j in range(i+1, len(PHYS_PROPS)):
        rows.append(dict(
            p1   = PHYS_PROPS[i],
            p2   = PHYS_PROPS[j],
            rhoL = avg_loro[i, j],
            rhoM = avg_lomo[i, j],
            d    = delta[i, j],
            absd = abs(delta[i, j]),
        ))
df = (pd.DataFrame(rows)
        .sort_values("absd", ascending=False)
        .head(14))   # top-8 shifts

# build reverse DataFrame for LoMO→LoRO (delta reversed)
delta_rev = avg_loro - avg_lomo
rows_rev = []
for i in range(len(PHYS_PROPS)):
    for j in range(i+1, len(PHYS_PROPS)):
        rows_rev.append(dict(
            p1   = PHYS_PROPS[i],
            p2   = PHYS_PROPS[j],
            rhoL = avg_lomo[i, j],  # LoMO first
            rhoM = avg_loro[i, j],  # then LoRO
            d    = delta_rev[i, j],
            absd = abs(delta_rev[i, j]),
        ))
df_rev = (pd.DataFrame(rows_rev)
          .sort_values("absd", ascending=False)
          .head(14))

# Alphabetical ordering by LaTeX pair name
def get_pair_latex(r):
    return f"{LATEX[r.p1]}\\!\\leftrightarrow {LATEX[r.p2]}"

df = df.copy()
df['pair_latex'] = [get_pair_latex(r) for _, r in df.iterrows()]
df = df.sort_values('pair_latex')
df = df.reset_index(drop=True)

df_rev = df_rev.copy()
df_rev['pair_latex'] = [get_pair_latex(r) for _, r in df_rev.iterrows()]
df_rev = df_rev.sort_values('pair_latex')
df_rev = df_rev.reset_index(drop=True)
