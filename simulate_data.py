import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm.auto import tqdm
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# --- GENERAL SETUP ---
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
sns.set_theme(style="whitegrid")
print("Pipeline configured with MULTI-STAGE generator and CORRECTED solver/sampler.")

# --- 1. ODE SYSTEM DEFINITION (Unchanged) ---
species_names_simple = ['S1', 'S2', 'S3', 'S4', 'S5']
ks_names_simple = ['k_prod', 'k1', 'k_1', 'k2', 'k_2', 'k3', 'k4', 'k_4', 'k_deg']
base_ks_simple = np.array([1.0, 0.1, 0.01, 0.05, 0.005, 0.02, 0.03, 0.003, 0.01])
num_species = len(species_names_simple)

# --- 2. CORE FUNCTIONS: SAMPLING, SOLVING, AND DISTANCE ---

def sample_ks_sobol(base_ks, n_samples, spread=2):
    """
    ### FIXED ###
    Generates uncorrelated kinetic parameter vectors. Each parameter is sampled
    independently from a log-uniform distribution centered on its base value.
    """
    n_params = len(base_ks)
    sampler = qmc.Sobol(d=n_params, scramble=True)
    sobol_samples = sampler.random(n=n_samples)

    log_base = np.log10(base_ks)
    log_min = log_base - spread
    log_max = log_base + spread

    log_values = log_min + sobol_samples * (log_max - log_min)
    return 10 ** log_values

def get_steady_state_algebraic(ks):
    """
    ### FIXED ###
    Finds the steady state using a corrected matrix A. Rejects non-physical solutions.
    """
    k_prod, k1, k_1, k2, k_2, k3, k4, k_4, k_deg = ks

    A = np.zeros((num_species, num_species))
    b = np.zeros(num_species)

    # Build the matrix and vector from the system's equations at steady state (d/dt = 0)
    # S1: k1*S1 - k_1*S2 = k_prod
    A[0, 0] = -k1; A[0, 1] = k_1
    b[0] = -k_prod
    # S2: k1*S1 - (k_1+k2)*S2 + k_2*S3 = 0
    A[1, 0] = k1;  A[1, 1] = -k_1 - k2; A[1, 2] = k_2
    # S3: k2*S2 - (k_2+k3)*S3 = 0
    A[2, 1] = k2;  A[2, 2] = -k_2 - k3
    # S4: k3*S3 - k4*S4 + k_4*S5 = 0
    A[3, 2] = k3;  A[3, 3] = -k4; A[3, 4] = k_4
    # S5: k4*S4 - (k_4+k_deg)*S5 = 0
    A[4, 3] = k4;  A[4, 4] = -k_4 - k_deg

    try:
        v_ss = np.linalg.solve(A, b)
        if np.all(v_ss >= 0) and np.any(v_ss > 1e-9):
            return v_ss
    except np.linalg.LinAlgError:
        return None
    return None

def relative_distance(v1, v2):
    """Calculates normalized relative distance."""
    epsilon = 1e-9
    norm_diff = np.linalg.norm((v1 - v2) / (v1 + v2 + epsilon))
    return norm_diff / np.sqrt(len(v1))

# --- 3. RESTORED MULTI-STAGE DATA GENERATION PIPELINE ---

def generate_seed_data(n_seeds, param_spread):
    """Stage 1: Generate seed data using the corrected sampler and solver."""
    print(f"--- Stage 1: Generating {n_seeds} seed data points ---")
    seed_results = []
    # Use the FIXED sampling function
    ks_list = sample_ks_sobol(base_ks_simple, n_seeds, param_spread)

    for ks in tqdm(ks_list, desc="Generating Seeds"):
        # Use the FIXED algebraic solver
        vss = get_steady_state_algebraic(ks)
        if vss is not None:
            seed_results.append({'k': ks, 'vss': vss})

    print(f"✅ Found {len(seed_results)} valid seed pairs.")
    return seed_results

def refine_initial_conditions(seed_data, perturbation_std, n_perturbations):
    """Stage 2: Create pools of unique k and v0 vectors from seed data."""
    print(f"\n--- Stage 2: Refining initial conditions from seed data ---")
    k_pool_list = []
    v0_pool_list = []

    for seed in tqdm(seed_data, desc="Refining v0"):
        k, vss_original = seed['k'], seed['vss']

        for _ in range(n_perturbations):
            # Perturb on a log scale for better behavior across magnitudes
            log_vss = np.log10(vss_original + 1e-12)
            log_noise = np.random.normal(0, perturbation_std, size=num_species)
            v0_new = 10**(log_vss + log_noise)
            v0_new[v0_new < 1e-12] = 1e-12

            k_pool_list.append(k)
            v0_pool_list.append(v0_new)

    # Use np.unique to create the final pools
    valid_k_pool = np.unique(k_pool_list, axis=0)
    valid_v0_pool = np.unique(v0_pool_list, axis=0)

    print(f"✅ Created pools with {len(valid_k_pool)} unique k vectors and {len(valid_v0_pool)} unique v0 vectors.")
    return valid_k_pool, valid_v0_pool

def run_combinatorial_simulations(k_pool, v0_pool, distance_threshold, sample_fraction):
    """
    Stage 3: Randomly samples a fraction of all possible (k, v0) combinations,
    simulates them, and filters by closeness.
    """
    total_combinations = len(k_pool) * len(v0_pool)
    n_samples_to_run = int(total_combinations * sample_fraction)

    print(f"\n--- Stage 3: Randomly sampling {n_samples_to_run} ({sample_fraction:.1%}) of {total_combinations} possible combinations ---")
    final_results = []

    # Generate random indices for k and v0 to create random pairs
    k_indices = np.random.randint(0, len(k_pool), size=n_samples_to_run)
    v0_indices = np.random.randint(0, len(v0_pool), size=n_samples_to_run)

    for i in tqdm(range(n_samples_to_run), desc="Sampled Combinatorial Run"):
        k = k_pool[k_indices[i]]
        v0 = v0_pool[v0_indices[i]]

        # Use the FIXED algebraic solver
        vss = get_steady_state_algebraic(k)

        if vss is not None and relative_distance(v0, vss) < distance_threshold:
            row = {name: k_val for name, k_val in zip(ks_names_simple, k)}
            row.update({f'v0_{s}': v for s, v in zip(species_names_simple, v0)})
            row.update({f'vss_{s}': v for s, v in zip(species_names_simple, vss)})
            final_results.append(row)

    print(f"✅ Found {len(final_results)} final training samples after filtering.")
    return pd.DataFrame(final_results)

def plot_histograms(df):
    """Stage 4: Generates and saves histograms of the final distributions."""
    print("\n--- Stage 4: Generating histograms of final distributions ---")
    if df.empty:
        print("❌ Cannot plot histograms, the final DataFrame is empty.")
        return

    fig, axes = plt.subplots(num_species, 2, figsize=(12, 3 * num_species))
    fig.suptitle('Log-Scale Distributions of Initial (v0) and Steady-State (vss) Concentrations', fontsize=16, y=0.99)

    for i, species in enumerate(species_names_simple):
        v0_col = f'v0_{species}'
        vss_col = f'vss_{species}'
        sns.histplot(df[v0_col], ax=axes[i, 0], log_scale=True, kde=True, color='skyblue')
        axes[i, 0].set_title(f'{species} - Initial Conditions (v0)')
        axes[i, 0].set_ylabel('Count')
        sns.histplot(df[vss_col], ax=axes[i, 1], log_scale=True, kde=True, color='salmon')
        axes[i, 1].set_title(f'{species} - Steady States (vss)')
        axes[i, 1].set_ylabel('')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = 'concentration_histograms_multistage_corrected.png'
    plt.savefig(plot_path)
    print(f"✅ Histograms saved to '{plot_path}'")
    plt.show()

# --- 4. EXECUTE AND SAVE ---
# Parameters for the pipeline
N_SEED_SAMPLES = 10000  # Increased seeds for a richer initial pool
PARAM_SPREAD = 2.5 # Slightly wider spread
V0_PERTURBATION_STD = 0.5 # Std dev for v0 perturbation (log scale)
DISTANCE_THRESHOLD = 0.6 # Loosened for higher yield
N_PERTURBATIONS_PER_SEED = 10
COMBINATORIAL_SAMPLE_FRACTION = 0.05 # Sample 5% of the massive combinatorial space

dataset_path = 'toy_steady_states_multistage_corrected.csv'

if not os.path.exists(dataset_path):
    # Stage 1
    seed_data = generate_seed_data(n_seeds=N_SEED_SAMPLES, param_spread=PARAM_SPREAD)

    if seed_data:
        # Stage 2
        k_pool, v0_pool = refine_initial_conditions(
            seed_data,
            V0_PERTURBATION_STD,
            n_perturbations=N_PERTURBATIONS_PER_SEED
        )

        if len(k_pool) > 0 and len(v0_pool) > 0:
            # Stage 3
            df_final = run_combinatorial_simulations(
                k_pool,
                v0_pool,
                DISTANCE_THRESHOLD,
                sample_fraction=COMBINATORIAL_SAMPLE_FRACTION
            )

            if not df_final.empty:
                df_final.to_csv(dataset_path, index=False)
                print(f"\n✅ Successfully generated and saved dataset!")
                print(f"   - Final Samples: {len(df_final)}")
                print(f"   - Location: {dataset_path}")

                # Stage 4
                plot_histograms(df_final)
            else:
                print("\n❌ No final samples found after combinatorial filtering. Try increasing COMBINATORIAL_SAMPLE_FRACTION or DISTANCE_THRESHOLD.")
        else:
            print("\n❌ Could not create valid pools of k or v0. Halting process.")
    else:
        print("\n❌ Could not generate any seed data. Halting process.")
else:
    print(f"Dataset '{dataset_path}' already exists. Loading data for histogram plot.")
    df_check = pd.read_csv(dataset_path)
    print(f"   - Samples: {len(df_check)}")
    plot_histograms(df_check)