
import numpy as np
import pandas as pd




# Generate a rough forecast of future prices based on historical data. 

def generate_synthetic_prices(
    y,
    years_out=20,
    noise_std=0.05,
    shuffle=True,
    weekly_bootstrap=False,
    seed=42
):
    """
    Generate a long-horizon synthetic electricity price series by repeating,
    shuffling, and lightly perturbing a historical dataset.

    Parameters
    ----------
    y : pandas.Series
        Hourly historical price series with a proper datetime index.
        Should cover at least ~3 years for good seasonal structure.
    years_out : int, default 20
        Number of years to generate.
    noise_std : float, default 0.05
        Standard deviation of multiplicative noise applied to each hour.
        Example: 0.05 = ±5% noise.
    shuffle : bool, default True
        If True, randomly shuffle the order of historical years.
    weekly_bootstrap : bool, default False
        If True, randomly sample weeks instead of whole years.
        Produces more diversity but slightly less structural fidelity.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    synthetic : pandas.Series
        A synthetic hourly price series of length `years_out` years,
        with realistic seasonality and mild stochastic variation.
    """

    np.random.seed(seed)

    # Ensure hourly frequency
    y = y.sort_index()
    y = y.asfreq("H")

    # Extract year boundaries
    years = sorted(y.index.year.unique())
    n_hist_years = len(years)

    if n_hist_years < 2:
        raise ValueError("Need at least 2 years of historical data.")

    # --- STEP 1: Break into chunks ---
    if weekly_bootstrap:
        # Break into weekly chunks
        weekly_groups = list(y.groupby([y.index.year, y.index.isocalendar().week]))
        chunks = [g[1] for g in weekly_groups]
    else:
        # Break into yearly chunks
        chunks = [y[y.index.year == yr] for yr in years]

    # --- STEP 2: Shuffle chunks ---
    if shuffle:
        np.random.shuffle(chunks)

    # --- STEP 3: Repeat chunks until we reach desired length ---
    needed_chunks = int(np.ceil(years_out / n_hist_years)) * len(chunks)
    repeated_chunks = chunks * int(np.ceil(needed_chunks / len(chunks)))

    # Concatenate and trim
    synthetic = pd.concat(repeated_chunks)
    synthetic = synthetic.iloc[: years_out * 365 * 24]

    # --- STEP 4: Add mild multiplicative noise ---
    noise = np.random.normal(loc=1.0, scale=noise_std, size=len(synthetic))
    synthetic = synthetic * noise

    # --- STEP 5: Fix index to be continuous hourly timestamps ---
    start = y.index[0]
    synthetic.index = pd.date_range(start=start, periods=len(synthetic), freq="H")

    return synthetic
