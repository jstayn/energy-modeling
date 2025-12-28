
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
    # monthly_discount_rate = (1 + nominal_discount_rate) ** (1/12) - 1

    # Financial Calculations
    # Note that this discounts everything back to the first day of the simulation. 
    start_date = df_results['time (hours)'].min()
    daily_finances = df_results.set_index('time (hours)')[['battery_profit']].resample('D').sum()
    daily_finances['day_of_sim'] = [x.days for x in daily_finances.index - start_date]
    daily_finances['discounted_profit'] = daily_finances['battery_profit'] / (1 + daily_discount_rate) ** daily_finances['day_of_sim']
    npv_op_profit = daily_finances['discounted_profit'].sum()

    return df_results, npv_op_profit, daily_finances

