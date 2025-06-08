from pathlib import Path
import glob, json, sys, warnings
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
ROOT_LORO = Path("loro_annotation_results")
ROOT_LOMO = Path("lomo_annotation_results")

NAME_MAP = {
    "anthropic_claude-opus-4":        "Claude Opus 4 (Anthropic)",
    "anthropic_claude-sonnet-4":      "Claude Sonnet 4 (Anthropic)",
    "google_gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash (Google)",
    "meta-llama_llama-4-maverick":    "Llama-4 Maverick (Meta)",
    "mistralai_mistral-medium-3":     "Mistral Medium 3 (Mistral AI)",
    "openai_gpt-4.1-mini":            "GPT-4.1 Mini (OpenAI)",
    "x-ai_grok-2-1212":               "Grok 2 (1212) (X.ai)",
    "x-ai_grok-2-vision-1212":        "Grok 2 Vision (1212) (X.ai)",
}
pretty = lambda slug: NAME_MAP.get(slug, slug.replace("_", r"\_"))


# --------------------------------------------------------------------------
# Crawler: returns DataFrame[model, material, R, prop, err]
# Non-finite or unparsable values become NaN
# --------------------------------------------------------------------------
def collect(root: Path, tag: str) -> pd.DataFrame:
    rows, patt = [], root / "**" / f"*{tag}_*.json"
    for fp in glob.iglob(str(patt), recursive=True):
        parts = Path(fp).parts
        try:
            model, material, R_val = parts[-4:-1]
        except ValueError:
            continue

        data = json.load(open(fp))
        for res in data.get("results", []):
            for k, v in (res.get("metrics") or {}).items():
                if not (k.endswith("_error") and k != "space_group_match"):
                    continue
                # robust numeric conversion
                try:
                    val = float(v) if v is not None else np.nan
                    if not np.isfinite(val):
                        val = np.nan
                except (TypeError, ValueError):
                    val = np.nan

                rows.append(dict(
                    model=model,
                    material=material,
                    R=res.get("held_out_r", R_val),
                    prop=k.replace("_error", ""),
                    err=val,
                    split=tag[:4],
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Collect splits
# --------------------------------------------------------------------------
loro_df = collect(ROOT_LORO, "loro_annotations")
lomo_df = collect(ROOT_LOMO, "lomo_annotations")
if loro_df.empty or lomo_df.empty:
    sys.exit("❌ One of the splits is empty – check ROOT paths.")

common_models = sorted(set(loro_df.model) & set(lomo_df.model))
print("✅ models kept:", common_models)

loro_df = loro_df[loro_df.model.isin(common_models)]
lomo_df = lomo_df[lomo_df.model.isin(common_models)]

# --------------------------------------------------------------------------
# Merge & compute per-sample metrics
# --------------------------------------------------------------------------
both = loro_df.merge(
    lomo_df,
    on=["model", "material", "R", "prop"],
    suffixes=("_loro", "_lomo"),
    how="inner",
)
both["gap"]   = both.err_lomo - both.err_loro
both["ratio"] = both.err_lomo / both.err_loro.replace(0, np.nan)

both.to_csv("knowledge_transfer_metrics.csv", index=False)
print("💾 knowledge_transfer_metrics.csv written")

# --------------------------------------------------------------------------
# Per-model aggregates (NaN-aware)
# --------------------------------------------------------------------------
safe_mean   = lambda s: float(np.nanmean(s))
safe_median = lambda s: float(np.nanmedian(s))

agg = (both.groupby("model", sort=False)
          .agg(LORO=("err_loro", safe_mean),
               LOMO=("err_lomo", safe_mean),
               T   =("ratio",    safe_mean))
          .reset_index())

agg["Worst_prop"] = (both.groupby("model")
                        .apply(lambda df: df.groupby("prop")["ratio"]
                                     .median().idxmax())
                        .values)

agg["Worst_mat"]  = (both.groupby("model")
                        .apply(lambda df: df.groupby("material")["gap"]
                                     .mean().idxmax())
                        .values)

agg["Gmax"]       = both.groupby("model")["gap"].max().values

# Sort for presentation
agg = agg.sort_values("T", na_position="last")

# --------------------------------------------------------------------------
# Format helpers
# --------------------------------------------------------------------------
def fmt(x, prec=3):
    return "--" if np.isnan(x) else f"{x:,.{prec}f}".replace(",", r"\,")

def fmt_ratio(x):
    return "--" if np.isnan(x) else f"{x:,.2f}".replace(",", r"\,")
