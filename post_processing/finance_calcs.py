
import pyomo.environ as pyo
import pandas as pd
import numpy as np

# Extract results

def extract_results(m, input_ts): 
    """
    Extracts results for a dispatch grid model from a pyomo model
    
    :param m: pyomo.environ.ConcreteModel that has been solved
    :param input_ts: DataFrame containing time series data with following columns: 
    * INTERVAL_START: Time column
    * LMP: Locational Marginal Price of electricity in $/MWh
    * load: Load data in MW 
    """
    model_vars = m.component_map(ctype=pyo.Var)
    df_out = pd.DataFrame({var.name: [var[t].value for t in m.t] for var in model_vars.values()})

    # Populate results df
    df_out['time (hours)'] = input_ts['INTERVAL_START']
    df_out['LMP'] = input_ts['LMP']  # Merge on time next time?
    df_out['load'] = input_ts['load']

    return df_out

def calculate_profit(m, input_ts, finance_inputs):
    """
    Extract model outputs and compute financial performance metrics for a 
    battery storage dispatch simulation.

    This function pulls all Pyomo decision variables from a solved model,
    aligns them with the input time series, computes operating profit with 
    and without the battery, and evaluates discounted cash flows to produce 
    an NPV of operating profit over the simulation horizon.

    Parameters
    ----------
    m : pyomo.ConcreteModel
        A solved Pyomo model containing time-indexed decision variables 
        (e.g., charge/discharge power, SOC, net grid power).
    input_ts : pandas.DataFrame
        Time series inputs used in the model. Must contain:
        - 'INTERVAL_START' : datetime-like index or column
        - 'LMP'            : locational marginal price ($/MWh)
        - 'load'           : baseline load (MW)
    finance_inputs : pandas.DataFrame
        Financial parameters used for discounting. Must contain:
        - 'Real Discount Rate (%)'
        - 'Inflation Rate (%)'

    Returns
    -------
    df_results : pandas.DataFrame
        Hourly results including model variables, LMP, load, and profit 
        components. Columns include:
        - 'time (hours)'
        - 'LMP'
        - 'load'
        - 'Grid Power (MW)'
        - 'Charge Power (MW)'
        - 'Discharge Power (MW)'
        - 'SOC (%)'
        - 'profit'
        - 'no_battery_profit'
        - 'battery_profit'
    npv_op_profit : float
        Net present value of operating profit over the simulation horizon,
        discounted back to the first day of the simulation.
    daily_finances : pandas.DataFrame
        Daily aggregated battery profit and discounted profit, including:
        - 'battery_profit'
        - 'day_of_sim'
        - 'discounted_profit'

    Notes
    -----
    - Profit is computed as: -LMP * power * timestep.
    - Discounting is performed using a nominal discount rate derived from 
      real discount rate and inflation.
    - All discounting is relative to the first timestamp in `input_ts`.
    """



    df_results = extract_results(m, input_ts)


    # Calculate profit
    df_results['profit'] = -1 * df_results['LMP'] * df_results['P_m'] * finance_inputs['Time Step (hours)'].iloc[0]
    df_results['no_battery_profit'] = -1 * df_results['LMP'] * df_results['load'] * finance_inputs['Time Step (hours)'].iloc[0]
    df_results['battery_profit'] = df_results['profit'] - df_results['no_battery_profit']
    df_results = df_results.rename(columns={
        'P_d': 'Discharge Power (MW)', 
        'P_c': 'Charge Power (MW)', 
        'S': 'SOC (%)', 
        'P_m': 'Grid Power (MW)'})


    # Add in discount rate
    real_discount_rate = finance_inputs['Real Discount Rate (%)'].iloc[0]  # Yearly opportunity cost of investing money into this project
    inflation_rate = finance_inputs['Inflation Rate (%)'].iloc[0]  # Yearly 
    nominal_discount_rate = (1 + real_discount_rate) * (1 + inflation_rate) - 1
    daily_discount_rate = (1 + nominal_discount_rate) ** (1/365) - 1

    # Financial Calculations
    # Note that this discounts everything back to the first day of the simulation. 
    start_date = df_results['time (hours)'].min()
    daily_finances = df_results.set_index('time (hours)')[['battery_profit']].resample('D').sum()
    daily_finances['day_of_sim'] = [x.days for x in daily_finances.index - start_date]
    daily_finances['discounted_profit'] = daily_finances['battery_profit'] / (1 + daily_discount_rate) ** daily_finances['day_of_sim']
    npv_op_profit = daily_finances['discounted_profit'].sum()

    return df_results, npv_op_profit, daily_finances



def calculate_system_cost_npv(finance_inputs):
    """
    Calculate the net present value (NPV) of system cost from financing inputs.

    Required columns in ``finance_inputs``:
    - 'Battery System Cost ($)'
    - 'Loan APR (%)'
    - 'Loan Term (yrs)'
    - 'Real Discount Rate (%)'
    - 'Inflation Rate (%)'

    Parameters
    ----------
    finance_inputs : pandas.DataFrame
        Financial input table with at least one row containing the required
        columns.

    Returns
    -------
    float
        NPV of system cost based on a monthly cash-flow series where monthly
        loan payments are discounted using an internal monthly discount rate
        derived from real discount rate and inflation.
    """

    # Validate inputs 
    required_cols = [
        'Battery System Cost ($)',
        'Loan APR (%)',
        'Loan Term (yrs)',
        'Real Discount Rate (%)',
        'Inflation Rate (%)'
    ]

    missing_cols = [col for col in required_cols if col not in finance_inputs.columns]
    if missing_cols:
        raise ValueError(f"Missing required finance_inputs columns: {missing_cols}")

    # Extract and validate scalar inputs
    system_cost = float(finance_inputs['Battery System Cost ($)'].iloc[0])
    apr_pct = float(finance_inputs['Loan APR (%)'].iloc[0])
    loan_term_years = float(finance_inputs['Loan Term (yrs)'].iloc[0])
    real_discount_rate = float(finance_inputs['Real Discount Rate (%)'].iloc[0])
    inflation_rate = float(finance_inputs['Inflation Rate (%)'].iloc[0])

    if system_cost < 0:
        raise ValueError("'Battery System Cost ($)' must be non-negative.")
    if apr_pct < 0:
        raise ValueError("'Loan APR (%)' must be non-negative.")
    if loan_term_years <= 0:
        raise ValueError("'Loan Term (yrs)' must be greater than 0.")

    # Loan terms for payment calculation.
    monthly_rate = apr_pct / 12.0
    n_payments = int(round(loan_term_years * 12))

    # Internal monthly discount rate derived from real + inflation.
    annual_nominal_discount = (1 + real_discount_rate) * (1 + inflation_rate) - 1
    monthly_discount_rate = (1 + annual_nominal_discount) ** (1 / 12) - 1

    if monthly_rate == 0:
        monthly_payment = system_cost / n_payments
    else:
        growth = (1 + monthly_rate) ** n_payments
        monthly_payment = system_cost * (monthly_rate * growth) / (growth - 1)

    # Cash-flow sign convention: initial principal inflow then loan-payment outflows.
    cash_flows = np.concatenate(([system_cost], np.full(n_payments, -monthly_payment)))

    np_npv = getattr(np, 'npv', None)
    if np_npv is None:
        try:
            import numpy_financial as npf
            np_npv = npf.npv
        except ImportError as exc:
            raise ImportError(
                "np.npv is unavailable in this NumPy version. Install numpy-financial "
                "or provide an np.npv-compatible implementation."
            ) from exc

    # Calculate NPV from the monthly cash-flow array.
    npv_system_cost = float(np_npv(monthly_discount_rate, cash_flows))

    return npv_system_cost
