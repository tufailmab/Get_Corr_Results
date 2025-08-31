"""
Get_Corr_Results.py
-------------------

Author: Engr. Tufail Mabood, MSc Structural Engineering, UET Peshawar
Contact: https://wa.me/+923440907874
License: MIT License

Description:
This Python script performs advanced correlation analysis on numeric datasets.
It computes Pearson and Spearman correlation coefficients and p-values,
generates high-resolution heatmaps, publication-ready tables (CSV, Excel, LaTeX, PNG),
and produces a predictor-target summary report for machine learning workflows.

Developed by Engr. Tufail Mabood for reproducible statistical analysis and 
data preprocessing in research and machine learning projects.
"""


# Import all required libraries (This is developed in 3.12.3 Python)
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# User configuration
# Ensure your cleaned dataset has no special characters in the header row
# The script automatically treats the first row as column headers
# All columns except the last non-empty column are considered predictor variables
# The last non-empty column is automatically treated as the target variable

INPUT_FILENAME = "Your Cleaned Dataset.xlsx" # In the current directory, keep your cleaned dataset and change this name
OUTPUT_DIR = Path("Correlation Analyses")
IMAGE_DPI = 1200 #Highly recommended for Q1 journal
FONT_NAME = "Times New Roman"
# ---------------------------------------------------------------------

def ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_non_empty_columns(df: pd.DataFrame):
    """Return list of columns that contain at least one non-NA value (in original column order)."""
    non_empty = [col for col in df.columns if df[col].notna().any()]
    return non_empty

def compute_corr_and_pvalues(df: pd.DataFrame, method="pearson"):
    """
    Returns (corr_df, pval_df) for columns in df.
    method: "pearson" or "spearman"
    """
    cols = df.columns
    n = len(cols)
    corr = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    pvals = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            # Use pairwise complete observations
            x = df[ci]
            y = df[cj]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                # Not enough data for correlation
                corr.loc[ci, cj] = np.nan
                pvals.loc[ci, cj] = np.nan
                continue
            xi = x[valid]
            yi = y[valid]
            if method == "pearson":
                r, p = stats.pearsonr(xi, yi)
            elif method == "spearman":
                r, p = stats.spearmanr(xi, yi)
            else:
                raise ValueError("method must be 'pearson' or 'spearman'")
            corr.loc[ci, cj] = r
            pvals.loc[ci, cj] = p
    return corr, pvals

def set_font_tnr():
    # Set matplotlib font to Times New Roman globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = FONT_NAME
    plt.rcParams['font.size'] = 10
    # For table text smaller
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    sns.set_style("white")

def save_heatmap(df_corr, title, outpath, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_format=".2f"):
    set_font_tnr()
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(df_corr.columns)), max(4, 0.5*len(df_corr.columns))))
    sns.heatmap(df_corr, ax=ax, cmap=cmap, annot=annot, fmt=annot_format if annot else '',
                vmin=vmin, vmax=vmax, linewidths=0.25, linecolor='gray',
                cbar_kws={"shrink": 0.5})
    ax.set_title(title, fontname=FONT_NAME)
    plt.tight_layout()
    fig.savefig(outpath, dpi=IMAGE_DPI)
    plt.close(fig)

def save_table_image(df, title, outpath, max_cols=8):
    """
    Save a DataFrame as an image (matplotlib table). If too many columns, split into parts.
    """
    set_font_tnr()
    rows, cols = df.shape
    # If too many columns for a single image, create multiple images
    if cols <= max_cols:
        _save_table_image_single(df, title, outpath)
    else:
        # split into column chunks
        for i in range(0, cols, max_cols):
            sub = df.iloc[:, i:i+max_cols]
            p = outpath.with_name(outpath.stem + f"_part{i//max_cols+1}" + outpath.suffix)
            _save_table_image_single(sub, title + f" (cols {i+1}-{min(cols,i+max_cols)})", p)

def _save_table_image_single(df, title, outpath):
    set_font_tnr()
    nrows, ncols = df.shape
    cell_text = []
    # format numbers nicely
    for r in range(nrows):
        row = []
        for c in range(ncols):
            val = df.iat[r, c]
            if pd.isna(val):
                row.append("")
            elif isinstance(val, (float, np.floating)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        cell_text.append(row)

    fig_w = max(6, 0.9 * ncols)
    fig_h = max(2 + 0.25*nrows, 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_title(title, fontname=FONT_NAME)

    the_table = ax.table(cellText=cell_text,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         loc='center',
                         cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    # Force Times New Roman for each cell
    for key, cell in the_table.get_celld().items():
        cell.set_text_props(fontname=FONT_NAME)
        cell.set_edgecolor("gray")
    plt.tight_layout()
    fig.savefig(outpath, dpi=IMAGE_DPI, bbox_inches='tight')
    plt.close(fig)

def main():
    print("Starting correlation analysis...")

    INPUT = Path.cwd() / INPUT_FILENAME
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT} -- place the Excel file in the current working directory or change INPUT_FILENAME.")

    print(f"Reading Excel file: {INPUT}")
    df_raw = pd.read_excel(INPUT, header=0)  # first row as headers

    # Identify non-empty columns (in original order)
    non_empty_cols = find_non_empty_columns(df_raw)
    if len(non_empty_cols) < 2:
        raise ValueError("File must contain at least two non-empty columns to run predictor vs target correlation.")

    target_col = non_empty_cols[-1]
    predictor_cols = non_empty_cols[:-1]

    print("Detected target column (last non-empty):", target_col)
    print("Detected predictor columns (all non-empty columns before last):")
    for c in predictor_cols:
        print("  -", c)

    # Work with the subset: predictors + target
    df = df_raw[predictor_cols + [target_col]].copy()

    # Option: drop rows that are completely NaN for the selected cols
    df.dropna(axis=0, how='all', subset=df.columns, inplace=True)

    ensure_output_dir(OUTPUT_DIR)

    # Save the subset as CSV for reference
    subset_csv = OUTPUT_DIR / "selected_predictors_and_target.csv"
    df.to_csv(subset_csv, index=False)
    print("Saved selected subset to", subset_csv)

    # Convert non-numeric columns to numeric where possible (coerce)
    # Keep original non-numeric columns because correlation only meaningful on numeric
    df_num = df.select_dtypes(include=[np.number]).copy()
    # For columns that are not numeric, try to coerce
    non_numeric_cols = [c for c in df.columns if c not in df_num.columns]
    if non_numeric_cols:
        print("Trying to coerce non-numeric columns to numeric (if applicable):", non_numeric_cols)
        for c in non_numeric_cols:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().sum() > 0:
                df_num[c] = coerced

    # Final numeric columns to use
    numeric_cols = df_num.columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("Not enough numeric columns found for correlation analysis. At least 2 numeric columns required.")

    print("Numeric columns used for correlation:", numeric_cols)

    df_num = df_num[numeric_cols]

    # Compute Pearson and Spearman correlations + p-values
    print("Computing Pearson correlations and p-values...")
    pearson_corr, pearson_p = compute_corr_and_pvalues(df_num, method="pearson")
    print("Computing Spearman correlations and p-values...")
    spearman_corr, spearman_p = compute_corr_and_pvalues(df_num, method="spearman")

    # Save correlation matrices to CSV / Excel / LaTeX
    pearson_corr.to_csv(OUTPUT_DIR / "pearson_correlation_coefficients.csv")
    pearson_p.to_csv(OUTPUT_DIR / "pearson_p_values.csv")
    spearman_corr.to_csv(OUTPUT_DIR / "spearman_correlation_coefficients.csv")
    spearman_p.to_csv(OUTPUT_DIR / "spearman_p_values.csv")

    # Also save Excel with multiple sheets
    with pd.ExcelWriter(OUTPUT_DIR / "correlation_matrices.xlsx") as writer:
        pearson_corr.to_excel(writer, sheet_name="pearson_r")
        pearson_p.to_excel(writer, sheet_name="pearson_p")
        spearman_corr.to_excel(writer, sheet_name="spearman_r")
        spearman_p.to_excel(writer, sheet_name="spearman_p")

    # Save LaTeX tables (useful for papers)
    try:
        with open(OUTPUT_DIR / "pearson_corr_table.tex", "w", encoding="utf-8") as f:
            f.write(pearson_corr.round(4).to_latex())
        with open(OUTPUT_DIR / "spearman_corr_table.tex", "w", encoding="utf-8") as f:
            f.write(spearman_corr.round(4).to_latex())
    except Exception as e:
        print("Warning: Could not write LaTeX files:", e)

    print("Saved numeric correlation CSV/Excel/LaTeX files to", OUTPUT_DIR)

    # Create and save HD heatmaps
    print("Creating heatmaps (HD, 1200 dpi)...")
    save_heatmap(pearson_corr, "Pearson Correlation Coefficients", OUTPUT_DIR / "pearson_correlation_heatmap.png", annot=True, annot_format=".2f")
    save_heatmap(pearson_p, "Pearson p-values", OUTPUT_DIR / "pearson_pvalues_heatmap.png", annot=True, cmap='viridis', vmin=0, vmax=1, annot_format=".3f")
    save_heatmap(spearman_corr, "Spearman Correlation Coefficients", OUTPUT_DIR / "spearman_correlation_heatmap.png", annot=True, annot_format=".2f")
    save_heatmap(spearman_p, "Spearman p-values", OUTPUT_DIR / "spearman_pvalues_heatmap.png", annot=True, cmap='viridis', vmin=0, vmax=1, annot_format=".3f")

    # Save correlation tables images (for inclusion in paper) using Times New Roman
    print("Creating table images (Times New Roman)...")
    # Round to 4 decimals for visuals
    save_table_image(pearson_corr.round(4), "Pearson Correlation Coefficients (r)", OUTPUT_DIR / "pearson_corr_table.png", max_cols=8)
    save_table_image(pearson_p.round(4), "Pearson p-values", OUTPUT_DIR / "pearson_pvalues_table.png", max_cols=8)
    save_table_image(spearman_corr.round(4), "Spearman Correlation Coefficients (rho)", OUTPUT_DIR / "spearman_corr_table.png", max_cols=8)
    save_table_image(spearman_p.round(4), "Spearman p-values", OUTPUT_DIR / "spearman_pvalues_table.png", max_cols=8)

    # Additionally, produce a CSV that pairs coefficient and p-value for each predictor vs target only (useful for paper)
    preds_plus_target = numeric_cols
    tgt = target_col if target_col in preds_plus_target else numeric_cols[-1]  # ensure target is in numeric_cols
    # Create a summary table of predictor vs target correlations (Pearson & Spearman)
    summary_rows = []
    for col in numeric_cols:
        if col == tgt:
            continue
        # Use pairwise complete
        valid = df_num[col].notna() & df_num[tgt].notna()
        if valid.sum() < 3:
            pr, pp = (np.nan, np.nan)
            sr, sp = (np.nan, np.nan)
        else:
            pr, pp = stats.pearsonr(df_num.loc[valid, col], df_num.loc[valid, tgt])
            sr, sp = stats.spearmanr(df_num.loc[valid, col], df_num.loc[valid, tgt])
        summary_rows.append({
            "Predictor": col,
            "Target": tgt,
            "Pearson_r": pr,
            "Pearson_p": pp,
            "Spearman_rho": sr,
            "Spearman_p": sp,
            "N_pairs": int(valid.sum())
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Predictor")
    summary_df.to_csv(OUTPUT_DIR / "predictor_vs_target_summary.csv")
    # also save as image
    save_table_image(summary_df.round(4), f"Predictor vs Target correlations (target={tgt})", OUTPUT_DIR / "predictor_vs_target_summary.png", max_cols=6)

    print("All done. Results saved in:", OUTPUT_DIR.resolve())
    print("Files of interest:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print("  -", p.name)

if __name__ == "__main__":
    main()
