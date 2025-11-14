import pandas as pd
import glob
import re
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

GROUND_TRUTH_PATH = "interview_scores.csv"
ABLATION_PATTERN = "ablation_*.csv"

def analyze_pair(gt, ab, name="ablation"):
    rows = []

    for col in gt.columns:
        gt_vals = gt[col]
        ab_vals = ab[col]

        pear_corr, pear_p = pearsonr(gt_vals, ab_vals)
        spear_corr, spear_p = spearmanr(gt_vals, ab_vals)
        r2 = r2_score(gt_vals, ab_vals)

        rows.append({
            "criterion": col,
            "pearson_corr": pear_corr,
            "pearson_pvalue": pear_p,
            "spearman_corr": spear_corr,
            "spearman_pvalue": spear_p,
            "r2_score": r2
        })

    df = pd.DataFrame(rows)
    print(f"\n========== Ablation: {name} ==========")
    print(df)
    print("Average R²:", df['r2_score'].mean())
    print("======================================\n")

    return df


def main():
    # Load ground truth CSV, first column is candidate_id
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH, index_col=0)

    # Find all ablation CSVs
    ablation_files = glob.glob(ABLATION_PATTERN)

    if not ablation_files:
        print("No ablation files found.")
        return

    for file_path in ablation_files:
        match = re.match(r"ablation_(.*)\.csv", file_path)
        ab_name = match.group(1) if match else file_path

        # Load ablation CSV, first column is candidate_id
        ab_df = pd.read_csv(file_path, index_col=0)

        # Align rows by candidate ID
        gt_aligned, ab_aligned = ground_truth.align(ab_df, join='inner', axis=0)

        # Only compare shared criteria columns
        common_cols = gt_aligned.columns.intersection(ab_aligned.columns)

        analyze_pair(
            gt_aligned[common_cols],
            ab_aligned[common_cols],
            name=ab_name
        )


if __name__ == "__main__":
    main()
