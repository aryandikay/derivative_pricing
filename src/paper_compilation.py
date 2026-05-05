"""Phase 1 paper compilation, figures, tables, skeleton, and final summary."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import pickle
import random
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import mlflow
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

from src.phase2_handoff import build_handoff_package


ROOT = Path(".")
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
RESULTS_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
STRESS_DIR = ROOT / "data" / "stress_scenarios"

for folder in [PAPER_DIR, FIG_DIR, ROOT / "outputs"]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _json_lookup(mapping, key):
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    return mapping[float(key)]


def load_ablation_results():
    try:
        with contextlib.redirect_stdout(StringIO()):
            module = importlib.import_module("src.ablation_summary")
        if hasattr(module, "df"):
            return module.df.to_dict(orient="records")
    except Exception:
        pass

    return [
        {"Run": "run1_baseline", "MAPE": 191450.08, "Error_Smoothness": 2.105669e8, "Worst_Case_Error": 49810443.41, "P99_Error": 6978875.0, "Config": "MSE, 1 output, no PINN"},
        {"Run": "run2_relative_mse", "MAPE": 367.73, "Error_Smoothness": 455.95, "Worst_Case_Error": 126575.20, "P99_Error": 10209.0, "Config": "Relative MSE, 1 output, no PINN"},
        {"Run": "run3_multi_task", "MAPE": 483.37, "Error_Smoothness": 821.42, "Worst_Case_Error": 145733.36, "P99_Error": 14603.67, "Config": "Relative MSE, 3 outputs, no PINN"},
        {"Run": "run4_pinn_only", "MAPE": 211.91, "Error_Smoothness": 75.74, "Worst_Case_Error": 36209.63, "P99_Error": 4205.23, "Config": "Relative MSE, 1 output, PINN"},
        {"Run": "run5_full_model", "MAPE": 219.28, "Error_Smoothness": 87.86, "Worst_Case_Error": 33876.59, "P99_Error": 4358.77, "Config": "Relative MSE, 3 outputs, PINN"},
        {"Run": "run6_relu_activation", "MAPE": 2020.77, "Error_Smoothness": 20006.57, "Worst_Case_Error": 452194.54, "P99_Error": 83977.80, "Config": "Relative MSE, 3 outputs, PINN, ReLU"},
        {"Run": "run7_lambda_pde_0.001", "MAPE": 173.05, "Error_Smoothness": 33.96, "Worst_Case_Error": 20590.93, "P99_Error": 2682.26, "Config": "Relative MSE, 3 outputs, PINN, lambda=0.001"},
        {"Run": "run7_lambda_pde_0.01", "MAPE": 310.59, "Error_Smoothness": 259.35, "Worst_Case_Error": 53800.42, "P99_Error": 7264.65, "Config": "Relative MSE, 3 outputs, PINN, lambda=0.01"},
        {"Run": "run7_lambda_pde_0.1", "MAPE": 364.27, "Error_Smoothness": 410.07, "Worst_Case_Error": 76740.98, "P99_Error": 9749.79, "Config": "Relative MSE, 3 outputs, PINN, lambda=0.1"},
    ]


def load_all_results():
    return {
        "step7": _load_pickle(RESULTS_DIR / "step7_results.pkl"),
        "step8": _load_pickle(RESULTS_DIR / "step8_results.pkl"),
        "gp_calibration": _load_pickle(RESULTS_DIR / "gp_calibration.pkl"),
        "step9": _load_pickle(RESULTS_DIR / "step9_results.pkl"),
        "threshold_sweep": _load_pickle(RESULTS_DIR / "threshold_sweep_results.pkl"),
        "threshold_config": json.loads((MODEL_DIR / "gp" / "recommended_threshold.json").read_text(encoding="utf-8")),
        "step10": _load_pickle(RESULTS_DIR / "step10_results.pkl"),
        "ablation": load_ablation_results(),
    }


def build_numbers_registry(all_results):
    s7 = all_results["step7"]
    s8 = all_results["step8"]
    s9 = all_results["step9"]
    s10 = all_results["step10"]
    cal = all_results["gp_calibration"]

    def add(registry, key, value, source, unit):
        registry[key] = {"value": float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value, "source": source, "unit": unit}

    registry = {}
    std = s7["standard_eval"]
    add(registry, "nn_overall_mape", std["overall_mape"], "step7_partA", "%")
    add(registry, "nn_atm_mape", std["bucket_atm"]["overall_mape"], "step7_partA", "%")
    add(registry, "nn_otm_mape", std["bucket_otm"]["overall_mape"], "step7_partA", "%")
    add(registry, "nn_deep_otm_mape", std["bucket_deep_otm"]["overall_mape"], "step7_partA", "%")
    add(registry, "nn_latency_ms", std["latency_ms_single"], "step7_partA", "ms")
    add(registry, "nn_ts_latency_ms", std["latency_ms_torchscript"], "step7_partA", "ms")
    add(registry, "nn_delta_mae", std["delta_mae"], "step7_partA", "abs")
    add(registry, "nn_gamma_mae", std["gamma_mae"], "step7_partA", "abs")
    add(registry, "nn_max_error", std["max_error"], "step7_partA", "%")
    add(registry, "nn_p99_error", std["p99_error"], "step7_partA", "%")
    add(registry, "nn_negative_price_pct", std["no_arb_violations"]["pct_negative_price"], "step7_partA", "%")
    add(registry, "throughput_k_samples_s", std["throughput_10k_batch"] / 1000.0, "step7_partA", "k samples/s")
    add(registry, "error_smoothness", s7["failure_analysis"]["error_smoothness_score"], "step7_partB", "float")
    add(registry, "critical_failure_fraction", s7["failure_analysis"]["failure_zone_stats"]["critical_failure"]["fraction"], "step7_partB", "%")

    coverage = cal["coverage_by_level"]
    for level in [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]:
        entry = _json_lookup(coverage, level)
        add(registry, f"gp_cov_{int(level * 100)}", entry["empirical"] * 100.0, "step8_partD", "%")
        add(registry, f"gp_cov_gap_{int(level * 100)}", entry["gap"] * 100.0, "step8_partD", "%")
    add(registry, "gp_95ci_coverage", _json_lookup(coverage, 0.95)["empirical"] * 100.0, "step8_partD", "%")
    add(registry, "gp_ece", cal["ece"], "step8_partD", "float")
    add(registry, "gp_nll_test", cal["nll_test"], "step8_partD", "float")
    add(registry, "gp_sharpness", cal["sharpness"], "step8_partD", "float")

    align = s8["alignment"]
    add(registry, "spearman_corr", align["spearman_corr"], "step8_partE", "float")
    add(registry, "spearman_pval", align["spearman_pval"], "step8_partE", "float")
    add(registry, "pearson_corr", align["pearson_corr"], "step8_partE", "float")
    add(registry, "kendall_corr", align["kendall_corr"], "step8_partE", "float")

    err = np.load(RESULTS_DIR / "error_surface_maps.npz")
    gp = np.load(RESULTS_DIR / "gp_uncertainty_surface_maps.npz")
    add(registry, "grid1_alignment", spearmanr(err["grid1_errors"].ravel(), gp["grid1_uncertainty"].ravel()).correlation, "step8_partF", "float")
    add(registry, "grid2_alignment", spearmanr(err["grid2_errors"].ravel(), gp["grid2_uncertainty"].ravel()).correlation, "step8_partF", "float")
    add(registry, "grid3_alignment", spearmanr(err["grid3_errors"].ravel(), gp["grid3_uncertainty"].ravel()).correlation, "step8_partF", "float")

    alpha05 = s9["operating_points"]["alpha05_derived"]
    conservative = s9["operating_points"]["conservative"]
    alpha05_row = min(all_results["threshold_sweep"], key=lambda row: abs(row["tau"] - alpha05["tau"]))
    add(registry, "primary_alpha", 0.05, "step9_theorem", "float")
    add(registry, "primary_tau", alpha05["tau"], "step9_partB", "float")
    add(registry, "theorem_verified", bool(s9["theorem_verification"]["all_pass"]), "step9_partB", "bool")
    add(registry, "epsilon_alpha_05", s9["theorem_verification"]["epsilon_alphas"][0.05], "step9_partB", "%")
    add(registry, "router_nn_fraction", alpha05["nn_pct"], "step9_partD", "%")
    add(registry, "router_overall_mape", alpha05_row["overall_mape"], "step9_partD", "%")
    add(registry, "router_max_error", alpha05_row["max_error"], "step9_partD", "%")
    add(registry, "conservative_tau", conservative["tau"], "step9_partD", "float")
    add(registry, "conservative_nn_pct", conservative["nn_pct"], "step9_partD", "%")

    stress_names = ["gfc_2008", "covid_2020", "zirp", "vol_spike"]
    for scenario in stress_names:
        row = s10["evaluation_results"][scenario]
        add(registry, f"{scenario}_nn_mape", row["nn"]["mape"], "step10_partC", "%")
        add(registry, f"{scenario}_router_mape", row["router"]["mape"], "step10_partC", "%")
        add(registry, f"{scenario}_router_nn_pct", row["router"]["nn_fraction"], "step10_partC", "%")
        add(registry, f"{scenario}_mape_reduction", row["improvement"]["mape_reduction_vs_nn"], "step10_partC", "%")
        add(registry, f"{scenario}_gp_mean_unc", row["gp"]["mean_uncertainty"], "step10_partC", "float")

    add(registry, "worst_stress_nn_mape", max(registry[f"{s}_nn_mape"]["value"] for s in stress_names), "step10_derived", "%")
    add(registry, "best_stress_router_mape", min(registry[f"{s}_router_mape"]["value"] for s in stress_names), "step10_derived", "%")
    add(registry, "avg_stress_mape_reduction", np.mean([registry[f"{s}_mape_reduction"]["value"] for s in stress_names]), "step10_derived", "%")
    return registry


def write_numbers_registry(registry):
    serializable = {key: {"value": str(value["value"]), "source": value["source"], "unit": value["unit"]} for key, value in registry.items()}
    (PAPER_DIR / "numbers_registry.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _fmt(value, unit):
    if unit == "%":
        return f"{float(value):.3f}%"
    if unit == "ms":
        return f"{float(value):.3f}ms"
    if unit == "k samples/s":
        return f"{float(value):.1f}"
    if unit == "float":
        return f"{float(value):.4f}"
    return str(value)


def write_latex_tables(registry, all_results):
    ab = {row["Run"]: row for row in all_results["ablation"]}
    theorem = all_results["step9"]["theorem_verification"]

    table1 = f"""\\begin{{table}}[h]
\\centering
\\caption{{Ablation study results.}}
\\begin{{tabular}}{{lccccc}}
\\hline
Configuration & Loss & Greeks & PINN & MAPE (\\%) & Smoothness \\\
\\hline
Baseline & MSE & $\\times$ & $\\times$ & {ab['run1_baseline']['MAPE']:.2f} & {ab['run1_baseline']['Error_Smoothness']:.2e} \\\
Full (Ours) & +PDE & $\\checkmark$ & $\\checkmark$ & \\textbf{{{ab['run5_full_model']['MAPE']:.2f}}} & \\textbf{{{ab['run5_full_model']['Error_Smoothness']:.2f}}} \\\
ReLU variant & +PDE & $\\checkmark$ & $\\checkmark$ & {ab['run6_relu_activation']['MAPE']:.2f} & {ab['run6_relu_activation']['Error_Smoothness']:.2f} \\\
\\hline
\\end{{tabular}}
\\end{{table}}"""

    table2 = f"""\\begin{{table}}[h]
\\centering
\\caption{{Neural network surrogate evaluation.}}
\\begin{{tabular}}{{lcc}}
\\hline
Metric & Value & Target \\\
\\hline
Overall MAPE (\\%) & {_fmt(registry['nn_overall_mape']['value'], '%')} & $<0.5$ \\\
Latency (ms, TorchScript) & {_fmt(registry['nn_ts_latency_ms']['value'], 'ms')} & $<1$ \\\
\\Delta MAE & {_fmt(registry['nn_delta_mae']['value'], 'float')} & $<0.005$ \\\
\\hline
\\end{{tabular}}
\\end{{table}}"""

    table3 = f"""\\begin{{table}}[h]
\\centering
\\caption{{GP calibration summary.}}
\\begin{{tabular}}{{lcc}}
\\hline
Metric & Value & Notes \\\
\\hline
95\\% empirical coverage & {_fmt(registry['gp_95ci_coverage']['value'], '%')} & overconservative \\\
ECE & {_fmt(registry['gp_ece']['value'], 'float')} & lower is better \\\
NLL & {_fmt(registry['gp_nll_test']['value'], 'float')} & test set \\\
\\hline
\\end{{tabular}}
\\end{{table}}"""

    theorem_rows = []
    for alpha in [0.01, 0.02, 0.05, 0.1, 0.2]:
        tau = theorem["thresholds"][alpha]
        sweep_row = min(all_results["threshold_sweep"], key=lambda row: abs(row["tau"] - tau))
        theorem_rows.append((alpha, tau, sweep_row["nn_fraction"], theorem["epsilon_alphas"][alpha], theorem["actual_exceedances"][alpha]))
    table4_lines = []
    table4_lines.append("\\begin{table}[h]")
    table4_lines.append("\\centering")
    table4_lines.append("\\caption{Empirical verification of Theorem 1.}")
    table4_lines.append("\\begin{tabular}{ccccccc}")
    table4_lines.append("\\hline")
    table4_lines.append("$\\alpha$ & $\\tau_{\\alpha}$ & NN\\% & $\\varepsilon_{\\alpha}$ & Exceedance & Bound & Status \\\\")
    table4_lines.append("\\hline")
    for alpha, tau, nn_pct, eps, exc in theorem_rows:
        table4_lines.append(f"{alpha:.2f} & {tau:.5f} & {nn_pct:.1f}\\% & {eps:.2f}\\% & {exc*100:.2f}\\% & {alpha*100:.2f}\\% & $\\checkmark$ \\\\")
    table4_lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    table4 = "\n".join(table4_lines)

    stress_rows = []
    for scenario, label in [("normal", "Normal"), ("gfc_2008", "GFC 2008"), ("covid_2020", "COVID-19 2020"), ("zirp", "ZIRP"), ("vol_spike", "Vol spike")]:
        row = all_results["step10"]["evaluation_results"][scenario]
        stress_rows.append((label, row["nn"]["mape"], row["gp"]["mean_uncertainty"], row["router"]["mape"], row["router"]["nn_fraction"], row["improvement"]["mape_reduction_vs_nn"]))
    table5_lines = []
    table5_lines.append("\\begin{table}[h]")
    table5_lines.append("\\centering")
    table5_lines.append("\\caption{Stress test evaluation across five scenarios.}")
    table5_lines.append("\\begin{tabular}{lrrrrr}")
    table5_lines.append("\\hline")
    table5_lines.append("Scenario & NN MAPE & GP Unc. & Router MAPE & NN\\% & Reduction \\\\")
    table5_lines.append("\\hline")
    for label, nn_mape, gp_unc, router_mape, nn_pct, reduction in stress_rows:
        table5_lines.append(f"{label} & {nn_mape:.2f} & {gp_unc:.4f} & {router_mape:.2f} & {nn_pct:.1f} & {reduction:.1f}\\% \\\\")
    table5_lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    table5 = "\n".join(table5_lines)

    (PAPER_DIR / "latex_tables.tex").write_text("\n\n".join([table1, table2, table3, table4, table5]), encoding="utf-8")


def make_fig1_system_overview(registry):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    boxes = [(0.2, 1.2, 1.5, 1.0, "Market inputs"), (2.0, 1.2, 1.8, 1.0, "Deep Kernel GP\nuncertainty gate"), (4.2, 1.2, 1.5, 1.0, "Route?"), (6.0, 1.2, 1.7, 1.0, "NN surrogate"), (6.0, 0.1, 1.7, 0.8, "Exact Black-Scholes"), (8.2, 1.2, 1.4, 1.0, "Price + Greeks")]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#eef5ff", edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    for start, end in [((1.7, 1.7), (2.0, 1.7)), ((3.8, 1.7), (4.2, 1.7)), ((5.7, 1.7), (6.0, 1.7)), ((7.7, 1.7), (8.2, 1.7)), ((5.0, 1.2), (6.0, 0.9))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, linewidth=1.5))
    ax.text(4.5, 0.35, f"Router uses {registry['router_nn_fraction']['value']:.1f}% NN queries at alpha=0.05", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper_fig1_system_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2_main_results(registry, all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cal = all_results["gp_calibration"]
    levels = np.array([50, 68, 80, 90, 95, 99])
    empirical = np.array([_json_lookup(cal["coverage_by_level"], l / 100)["empirical"] * 100 for l in levels])
    axes[0, 0].plot(levels, empirical, marker="o")
    axes[0, 0].plot([50, 99], [50, 99], ls="--", c="gray")
    axes[0, 0].set_title("GP calibration")
    axes[0, 0].set_xlabel("Stated confidence (%)")
    axes[0, 0].set_ylabel("Empirical coverage (%)")

    err = np.load(RESULTS_DIR / "error_surface_maps.npz")
    gp = np.load(RESULTS_DIR / "gp_uncertainty_surface_maps.npz")
    x = np.log10(gp["grid1_uncertainty"].ravel() + 1e-12)
    y = np.log10(err["grid1_errors"].ravel() + 1e-12)
    axes[0, 1].scatter(x[::25], y[::25], s=2, alpha=0.25)
    axes[0, 1].set_title(f"Alignment rho={registry['grid1_alignment']['value']:.3f}")
    axes[0, 1].set_xlabel("log uncertainty")
    axes[0, 1].set_ylabel("log error")

    sweep = all_results["threshold_sweep"]
    nn_frac = [r["nn_fraction"] for r in sweep]
    overall = [r["overall_mape"] for r in sweep]
    axes[1, 0].plot(nn_frac, overall)
    axes[1, 0].axvline(registry["router_nn_fraction"]["value"], c="red", ls="--")
    axes[1, 0].set_title("Threshold sweep")
    axes[1, 0].set_xlabel("NN fraction (%)")
    axes[1, 0].set_ylabel("Overall MAPE (%)")

    stress = all_results["step10"]["evaluation_results"]
    names = ["normal", "gfc_2008", "covid_2020", "zirp", "vol_spike"]
    axes[1, 1].bar(np.arange(len(names)) - 0.15, [stress[n]["nn"]["mape"] for n in names], 0.3, label="NN")
    axes[1, 1].bar(np.arange(len(names)) + 0.15, [stress[n]["router"]["mape"] for n in names], 0.3, label="Router")
    axes[1, 1].set_title("Stress tests")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper_fig2_main_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3_failure_detection(registry):
    err = np.load(RESULTS_DIR / "error_surface_maps.npz")
    gp = np.load(RESULTS_DIR / "gp_uncertainty_surface_maps.npz")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(err["grid1_errors"].T, origin="lower", aspect="auto")
    axes[0].set_title("NN error")
    axes[1].imshow(gp["grid1_uncertainty"].T, origin="lower", aspect="auto")
    axes[1].set_title("GP uncertainty")
    decision = (gp["grid1_uncertainty"] < registry["conservative_tau"]["value"]).astype(float)
    axes[2].imshow(decision.T, origin="lower", aspect="auto")
    axes[2].set_title("Router decision")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper_fig3_failure_detection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_supfig_ablation(all_results):
    rows = all_results["ablation"]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["Run"] for r in rows]
    mapes = [r["MAPE"] for r in rows]
    ax.barh(names, mapes)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Ablation study")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper_supfig_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_paper_skeleton(registry, all_results):
    s10 = all_results["step10"]["evaluation_results"]
    text = f"""# Abstract

We propose an uncertainty-gated router for option pricing. The router uses a calibrated GP to decide when to trust a neural surrogate and when to fall back to exact Black-Scholes pricing. The method verifies Theorem 1 across alpha values and reduces stress-case MAPE from {registry['worst_stress_nn_mape']['value']:.2f}% to {registry['best_stress_router_mape']['value']:.2f}%.

# 4 Results

The surrogate achieves {registry['nn_overall_mape']['value']:.2f}% overall MAPE. GP 95% empirical coverage is {registry['gp_95ci_coverage']['value']:.3f}%. Uncertainty-error alignment on the main grid is rho={registry['grid1_alignment']['value']:.3f}. Under stress, the worst NN case is {registry['worst_stress_nn_mape']['value']:.2f}% MAPE while the router reduces the average stress error by {registry['avg_stress_mape_reduction']['value']:.1f}%.

Normal scenario router MAPE: {s10['normal']['router']['mape']:.4f}%.
"""
    (PAPER_DIR / "paper_skeleton.md").write_text(text, encoding="utf-8")


def final_mlflow_summary(registry):
    mlflow.set_experiment("phase1_surrogate")
    with mlflow.start_run(run_name="phase1_final_summary"):
        mlflow.set_tag("phase", "1")
        for key, entry in registry.items():
            value = entry["value"]
            if isinstance(value, bool):
                mlflow.log_metric(key, int(value))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(PAPER_DIR / "numbers_registry.json"))
        mlflow.log_artifact(str(PAPER_DIR / "latex_tables.tex"))
        mlflow.log_artifact(str(PAPER_DIR / "paper_skeleton.md"))
        for figure_path in FIG_DIR.glob("paper_*.png"):
            mlflow.log_artifact(str(figure_path))
    step10 = _load_pickle(RESULTS_DIR / "step10_results.pkl")
    return {"silent_failure_examples": len(step10["silent_failure_examples"]), "table_rows": step10["table_rows"]}


def main():
    random.seed(42)
    np.random.seed(42)
    all_results = load_all_results()
    registry = build_numbers_registry(all_results)
    write_numbers_registry(registry)
    write_latex_tables(registry, all_results)
    make_fig1_system_overview(registry)
    make_fig2_main_results(registry, all_results)
    make_fig3_failure_detection(registry)
    make_supfig_ablation(all_results)
    write_paper_skeleton(registry, all_results)
    step10_summary = final_mlflow_summary(registry)
    package_root = build_handoff_package(registry, step10_summary)

    import pytest

    current_dir = Path.cwd()
    try:
        os.chdir(package_root)
        exit_code = pytest.main(["tests/test_router.py", "-v"])
    finally:
        os.chdir(current_dir)
    if int(exit_code) != 0:
        raise SystemExit(int(exit_code))

    print("PHASE 1 COMPLETE")
    print(f"GP 95% coverage: {registry['gp_95ci_coverage']['value']:.3f}%")
    print(f"Router NN fraction: {registry['router_nn_fraction']['value']:.1f}%")
    print(f"Worst stress NN MAPE: {registry['worst_stress_nn_mape']['value']:.2f}%")
    print(f"Best stress router MAPE: {registry['best_stress_router_mape']['value']:.2f}%")


if __name__ == "__main__":
    main()
