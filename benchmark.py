import numpy as np
import xarray as xr
import yaml
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.site import XRSite
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.wind_farm_models import All2AllIterative

# --- PR-Like Documentation ---
BENCHMARK_DOCUMENTATION = """
### Benchmarking `All2AllIterative` with Anders' Acceleration (`use_aa`)

**Motivation**  
I'm benchmarking the `All2AllIterative` model to see how much Anders' Acceleration (`use_aa=True`) improves performance. The goal is to show how enabling this option affects runtime across different wind farm sizes and layouts.

**Methodology**  
I tested the algorithm with different parameters:
1. **Farm Size**:  
   - Small: 10 turbines
   - Moderate: 50 turbines
   - Medium: 100 turbines  
   - Large: 250 turbines

2. **Turbine Layout & Spacing**:  
   - Minimum distances between turbines: 1, 2, 3, and 5 rotor diameters for sampling random positions of a layout. 
   - 10 layouts per setup (random seeds to ensure robustness)

3. **`use_aa` Flag**:  
   - `use_aa=True` vs `use_aa=False`

4. **Parallelization (`n_cpu`) - simply to speed up larger simulations**:  
   - 10 turbines: `n_cpu=1`  
   - 50 turbines: `n_cpu=2`
   - 100 turbines: `n_cpu=4`  
   - 250 turbines: `n_cpu=8`

**Execution**  
I ran simulations for all combinations, recording the runtime for each. The data is then used to compare the performance of `use_aa=True` vs `use_aa=False`.

**Expected Results**  
I expect to see faster runtimes with `use_aa=True`, especially for larger farms and tighter layouts where wake interactions are stronger. The results will help show whether Anders' Acceleration is a worthwhile default.
"""

# --- Benchmark Configuration ---
N_TURBINES_LIST = [10, 50, 100, 250]
N_CPU_MAP = {
    N_TURBINES_LIST[0]: 1,
    N_TURBINES_LIST[1]: 2,
    N_TURBINES_LIST[2]: 4,
    N_TURBINES_LIST[3]: 8,
}
MIN_DIST_FACTORS = [1, 2, 3, 5]
N_SEEDS = 10
OUTPUT_DIR = "benchmark_results"

# --- PyWake Setup ---
# Load wind resource and site data
wind_resource = yaml.load(
    open("wr.yaml"),
    Loader=yaml.FullLoader,
)["wind_resource"]
A = wind_resource["weibull_a"]
k = wind_resource["weibull_k"]
freq = wind_resource["sector_probability"]
wd = np.array(wind_resource["wind_direction"])
ws = np.array(wind_resource["wind_speed"])
TI = wind_resource["turbulence_intensity"]["data"]
SITE = XRSite(
    ds=xr.Dataset(
        data_vars={
            "Sector_frequency": ("wd", freq["data"]),
            "Weibull_A": ("wd", A["data"]),
            "Weibull_k": ("wd", k["data"]),
            "TI": ("ws", TI),
        },
        coords={"wd": wd, "ws": ws},
    )
)
WTURBINE = DTU10MW()
WT_DIAMETER = WTURBINE.diameter()

# Configure models
params = {
    "A": 0.28169340293751577,
    "ceps": 0.1420674010683264,
    "cti1": 1.2015732556452348,
    "cti2": 3.148364906110177,
    "ctlim": 0.7750556898490065,
}
turbulenceModel = CrespoHernandez()
def_args = {
    "A": params["A"],
    "cTI": [params["cti1"], params["cti2"]],
    "ceps": params["ceps"],
    "ctlim": params["ctlim"],
    "ct2a": ct2a_mom1d,
    "rotorAvgModel": GaussianOverlapAvgModel(),
}
wake_deficitModel = TurboGaussianDeficit(**def_args)
wake_deficitModel.WS_key = "WS_jlk"


# --- Helper Functions ---
def generate_layout(n_turbines, min_distance, seed):
    """Generates a random turbine layout with a minimum distance constraint."""
    np.random.seed(seed)
    x, y = [], []
    # Define a boundary for placement to avoid infinite loops
    # Scale boundary with number of turbines
    bound_factor = int(np.ceil(np.sqrt(n_turbines)))
    x_bound = WT_DIAMETER * 3 * bound_factor
    y_bound = WT_DIAMETER * 3 * bound_factor

    while len(x) < n_turbines:
        candidate_x = np.random.uniform(-x_bound, x_bound)
        candidate_y = np.random.uniform(-y_bound, y_bound)

        if not x:  # First turbine
            x.append(candidate_x)
            y.append(candidate_y)
            continue

        distances = np.sqrt(
            (candidate_x - np.array(x)) ** 2 + (candidate_y - np.array(y)) ** 2
        )
        if np.all(distances >= min_distance):
            x.append(candidate_x)
            y.append(candidate_y)
    return np.array(x), np.array(y)


# --- Main Benchmark Execution ---
def run_benchmark():
    """Runs the full benchmark suite."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = []
    total_runs = len(N_TURBINES_LIST) * len(MIN_DIST_FACTORS) * N_SEEDS * 2
    run_count = 0

    print("Starting benchmark...")
    for n_turbines in N_TURBINES_LIST:
        n_cpu = N_CPU_MAP[n_turbines]
        for min_dist_factor in MIN_DIST_FACTORS:
            min_distance = WT_DIAMETER * min_dist_factor
            for seed in range(N_SEEDS):
                x, y = generate_layout(n_turbines, min_distance, seed)

                for use_aa in [True, False]:
                    run_count += 1
                    print(
                        f"Running ({run_count}/{total_runs}): "
                        f"Turbines={n_turbines}, DistFactor={min_dist_factor}, "
                        f"Seed={seed}, use_aa={use_aa}, n_cpu={n_cpu}"
                    )

                    wfm = All2AllIterative(
                        SITE,
                        WTURBINE,
                        wake_deficitModel=wake_deficitModel,
                        blockage_deficitModel=SelfSimilarityDeficit2020(),
                        turbulenceModel=turbulenceModel,
                        superpositionModel=LinearSum(),
                        use_aa=use_aa,
                    )

                    s = time.time()
                    # Run simulation without verbose to not pollute output
                    sim_res = wfm(x, y, ws=ws, wd=wd, n_cpu=n_cpu)
                    runtime = time.time() - s
                    aep = sim_res.aep().sum()

                    results.append(
                        {
                            "n_turbines": n_turbines,
                            "min_dist_factor": min_dist_factor,
                            "seed": seed,
                            "use_aa": use_aa,
                            "n_cpu": n_cpu,
                            "runtime": runtime,
                            "aep": aep.item(),
                        }
                    )

    print("Benchmark finished.")
    return pd.DataFrame(results)


def analyze_and_plot(df):
    """Saves results to CSV and generates plots."""
    if df.empty:
        print("No results to analyze.")
        return

    # Save raw data
    results_path = os.path.join(OUTPUT_DIR, "benchmark_data.csv")
    df.to_csv(results_path, index=False)
    print(f"Benchmark data saved to {results_path}")

    # Generate plots
    sns.set_theme(style="whitegrid")

    for n_turbines in df["n_turbines"].unique():
        plt.figure(figsize=(12, 7))
        subset_df = df[df["n_turbines"] == n_turbines]

        ax = sns.barplot(
            data=subset_df,
            x="min_dist_factor",
            y="runtime",
            hue="use_aa",
            errorbar="sd",  # Show standard deviation across seeds
        )

        ax.set_title(
            f"Runtime Comparison for {n_turbines} Turbines (n_cpu={subset_df.n_cpu.iloc[0]})"
        )
        ax.set_xlabel("Minimum Distance Factor (x Rotor Diameter)")
        ax.set_ylabel("Runtime (seconds)")
        ax.legend(title="use_aa (Accelerated)")

        plot_path = os.path.join(
            OUTPUT_DIR, f"runtime_comparison_{n_turbines}_turbines.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    # Plot speedup factor
    for n_turbines in df["n_turbines"].unique():
        plt.figure(figsize=(12, 7))
        subset_df = df[df["n_turbines"] == n_turbines]

        # Pivot to get runtimes side-by-side
        pivot_df = subset_df.pivot_table(
            index=["min_dist_factor", "seed"], columns="use_aa", values="runtime"
        ).reset_index()
        pivot_df.rename(
            columns={True: "runtime_aa", False: "runtime_no_aa"}, inplace=True
        )

        # Handle cases where one of the runs might have failed (NaN)
        pivot_df.dropna(inplace=True)
        if pivot_df.empty:
            continue

        pivot_df["speedup"] = pivot_df["runtime_no_aa"] / pivot_df["runtime_aa"]

        ax = sns.boxplot(data=pivot_df, x="min_dist_factor", y="speedup")
        ax.axhline(1, ls="--", color="red")  # Line for no speedup
        ax.set_title(f"Speedup Factor (use_aa=True) for {n_turbines} Turbines")
        ax.set_xlabel("Minimum Distance Factor (x Rotor Diameter)")
        ax.set_ylabel("Speedup (Runtime_No_AA / Runtime_AA)")

        plot_path = os.path.join(
            OUTPUT_DIR, f"speedup_factor_{n_turbines}_turbines.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    # Run the benchmark
    results_df = run_benchmark()

    # Analyze and plot results
    analyze_and_plot(results_df)

    # Print the documentation
    print("\n" + "=" * 80)
    print("Benchmark Methodology and Documentation")
    print("=" * 80)
    print(BENCHMARK_DOCUMENTATION)
