"""Compare final-head predictions on the same admissions and known labels."""

import pandas as pd
import torch


def save_comparison(results, label_counts, output_dir):
    """Write macro scores, paired score differences and all per-finding scores."""
    reference = results["all"]
    rows = []
    per_finding = label_counts.copy()
    for condition, result in results.items():
        for key in ("hadm_ids", "y_gt", "label_mask"):
            if not torch.equal(result[key], reference[key]):
                raise ValueError(f"Test conditions have different {key}.")
        if list(result["label_names"]) != list(reference["label_names"]):
            raise ValueError("Test conditions have different target orders.")
        rows.append({
            "condition": condition,
            "modalities": "+".join(result["modalities"]),
            "admissions": len(result["hadm_ids"]),
            "macro_AP": result["ap"],
            "macro_AUROC": result["auroc"],
            "AP_findings": int(result["mlaps"].isfinite().sum()),
            "AUROC_findings": int(result["mlroc"].isfinite().sum()),
            "delta_AP_vs_all": result["ap"] - reference["ap"],
            "delta_AUROC_vs_all": result["auroc"] - reference["auroc"],
        })
        per_finding[f"AP_{condition}"] = result["mlaps"].numpy()
        per_finding[f"AUROC_{condition}"] = result["mlroc"].numpy()

    summary = pd.DataFrame(rows)
    differences = pd.DataFrame([
        {
            "comparison": f"{left} minus {right}",
            "delta_AP": results[left]["ap"] - results[right]["ap"],
            "delta_AUROC": results[left]["auroc"] - results[right]["auroc"],
        }
        for left, right in [("cxr_only", "all"), ("ecg_labs", "all"),
                            ("ecg_labs", "cxr_only")]
    ])
    summary.to_csv(output_dir / "test_summary.csv", index=False)
    per_finding.to_csv(output_dir / "test_per_finding.csv", index=False)
    differences.to_csv(output_dir / "test_differences.csv", index=False)
    return summary
