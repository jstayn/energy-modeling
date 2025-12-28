import pandas as pd 
import pyomo.environ as pyo


def battery_model(input_ts, input_scalar):
    """
    Create and solve a battery storage dispatch optimization model using Pyomo. Minimizes cost of 
    electricity procurement over the time horizon given LMP and load data with a linear program.
    This model will result in simultaneous charging and discharging if one-way conversion efficiency
    is 100%. 
    
    Parameters
    ----------
    input_ts : pandas.DataFrame
        Time-series input data containing 'LMP' and 'load' columns.
    input_scalar : pandas.DataFrame
        Scalar input parameters for the battery model.
    Returns
    -------
    m : pyomo.ConcreteModel
        The Pyomo model object after solving.
    results : Solver results object
        The results from the solver after solving the model.
    """


    # Create Model object
    m = pyo.ConcreteModel()

    # Set time steps
    m.t = pyo.Set(doc='Hour', initialize=range(len(input_ts)), ordered=True)

    # Define scalar parameters -- why do this? Mostly to keep the code functional if you pass the model to another namespace
    m.P_d_max = pyo.Param(initialize=input_scalar['Maximum Discharge (MW)'].iloc[0], doc='Max discharge power (MW)')
    m.P_c_max = pyo.Param(initialize=input_scalar['Maximum Charge (MW)'].iloc[0], doc='Max charge power (MW)')
    m.Emax = pyo.Param(initialize=input_scalar['Capacity (MWh)'].iloc[0], doc='Max energy (MWh)')
    m.SOC_min = pyo.Param(initialize=input_scalar['Minimum SOC'].iloc[0], doc='Min state of charge')
    m.SOC_max = pyo.Param(initialize=input_scalar['Maximum SOC'].iloc[0], doc='Max state of charge')
    m.delta_t = pyo.Param(initialize=input_scalar['Time Step (hours)'].iloc[0], doc='Time step (hours)')
    m.eta_conv = pyo.Param(initialize=input_scalar['One-Way Conversion Efficiency (%)'].iloc[0], doc='Battery conversion efficiency')
    m.eta_self = pyo.Param(initialize=input_scalar['Self-Discharge (%/hr)'].iloc[0], doc='Self-discharge rate per hour')

    # Time-dependent parameters
    m.LMP = pyo.Param(m.t, initialize=lambda m, t: input_ts['LMP'][t], doc='LMP (MW)')
    m.P_l = pyo.Param(m.t, initialize=lambda m, t: input_ts['load'][t], doc='Load (MW)')

    # Define variables

    m.P_d = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='Discharge power (MW)')
    m.P_c = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='Charge power (MW)')
    m.S = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='State of charge')
    m.P_m = pyo.Var(m.t, within=pyo.Reals, doc='Net Power (MW)')
    # m.u = pyo.Var(m.t, within=pyo.Binary, doc="Charging vs Discharging")

    # Set constraints

    def power_balance_rule(m, t):
        return m.P_m[t] == m.P_c[t] - m.P_d[t] + m.P_l[t]

    m.power_balance = pyo.Constraint(m.t, rule=power_balance_rule, doc='Power balance constraint')


    def storage_state_rule(m, t):
        if t == 0:
            return m.S[t] == input_scalar['Starting/Ending SOC'].iloc[0] 
        else:
            return m.S[t] == m.S[t-1] * (1 - m.eta_self * m.delta_t) + m.delta_t / m.Emax * (m.P_c[t-1] * m.eta_conv - m.P_d[t-1] / m.eta_conv) 
        
    m.storage_state = pyo.Constraint(m.t, rule=storage_state_rule, doc='State of charge constraint')


    def storage_power_rule(m, t):
        return m.P_d[t] * m.delta_t <= m.Emax * m.S[t]
    m.storage_power = pyo.Constraint(m.t, rule=storage_power_rule, doc='Storage power constraint')


    def boundary_conditions_rule(m, t):
        if t == len(m.t) - 1:
            return m.S[t] == input_scalar['Starting/Ending SOC'].iloc[0] 
        else:
            return pyo.Constraint.Skip
        
    m.boundary_conditions = pyo.Constraint(m.t, rule=boundary_conditions_rule, doc='SOC boundary conditions constraint')

    def max_charge_power_rule(m, t):   
        return m.P_c[t] <= m.P_c_max 
    m.max_charge_power = pyo.Constraint(m.t, rule=max_charge_power_rule, doc='Max charge power constraint')

    def max_discharge_power_rule(m, t):
        return m.P_d[t] <= m.P_d_max 
    m.max_discharge_power = pyo.Constraint(m.t, rule=max_discharge_power_rule, doc='Max discharge power constraint')

    def soc_min_rule(m, t):
        return m.S[t] >= m.SOC_min
    m.soc_min = pyo.Constraint(m.t, rule=soc_min_rule, doc='Min SOC constraint')

    def soc_max_rule(m, t):
        return m.S[t] <= m.SOC_max
    m.soc_max = pyo.Constraint(m.t, rule=soc_max_rule, doc='Max SOC constraint')




    # Define objective function 
    m.objective = pyo.Objective(rule=lambda m: sum(m.LMP[t] * m.P_m[t] * m.delta_t for t in m.t), sense=pyo.minimize, doc='Cost Minimization')

    # Solve the model 
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(m, tee=True)
    return m, results




def battery_model_no_eff(input_ts, input_scalar):
    """
    Create and solve a battery storage dispatch optimization model using Pyomo. Minimizes cost of 
    electricity procurement over the time horizon given LMP and load data with a mixed-integer linear program.
    This model is robust to an assumption of 100% conversion efficiency

    Parameters
    ----------
    input_ts : pandas.DataFrame
        Time-series input data containing 'LMP' and 'load' columns.
    input_scalar : pandas.DataFrame
        Scalar input parameters for the battery model.
    Returns
    -------
    m : pyomo.ConcreteModel
        The Pyomo model object after solving.
    results : Solver results object
        The results from the solver after solving the model.
    """


    # Create Model object
    m = pyo.ConcreteModel()

    # Set time steps
    m.t = pyo.Set(doc='Hour', initialize=range(len(input_ts)), ordered=True)

    # Define scalar parameters -- why do this? Mostly to keep the code functional if you pass the model to another namespace
    m.P_d_max = pyo.Param(initialize=input_scalar['Maximum Discharge (MW)'].iloc[0], doc='Max discharge power (MW)')
    m.P_c_max = pyo.Param(initialize=input_scalar['Maximum Charge (MW)'].iloc[0], doc='Max charge power (MW)')
    m.Emax = pyo.Param(initialize=input_scalar['Capacity (MWh)'].iloc[0], doc='Max energy (MWh)')
    m.SOC_min = pyo.Param(initialize=input_scalar['Minimum SOC'].iloc[0], doc='Min state of charge')
    m.SOC_max = pyo.Param(initialize=input_scalar['Maximum SOC'].iloc[0], doc='Max state of charge')
    m.delta_t = pyo.Param(initialize=input_scalar['Time Step (hours)'].iloc[0], doc='Time step (hours)')
    m.eta_conv = pyo.Param(initialize=input_scalar['One-Way Conversion Efficiency (%)'].iloc[0], doc='Battery conversion efficiency')
    m.eta_self = pyo.Param(initialize=input_scalar['Self-Discharge (%/hr)'].iloc[0], doc='Self-discharge rate per hour')

    # Time-dependent parameters
    m.LMP = pyo.Param(m.t, initialize=lambda m, t: input_ts['LMP'][t], doc='LMP (MW)')
    m.P_l = pyo.Param(m.t, initialize=lambda m, t: input_ts['load'][t], doc='Load (MW)')

    # Define variables

    m.P_d = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='Discharge power (MW)')
    m.P_c = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='Charge power (MW)')
    m.S = pyo.Var(m.t, within=pyo.NonNegativeReals, doc='State of charge')
    m.P_m = pyo.Var(m.t, within=pyo.Reals, doc='Net Power (MW)')
    m.u = pyo.Var(m.t, within=pyo.Binary, doc="Charging vs Discharging")

    # Set constraints

    def power_balance_rule(m, t):
        return m.P_m[t] == m.P_c[t] - m.P_d[t] + m.P_l[t]

    m.power_balance = pyo.Constraint(m.t, rule=power_balance_rule, doc='Power balance constraint')


    def storage_state_rule(m, t):
        if t == 0:
            return m.S[t] == input_scalar['Starting/Ending SOC'].iloc[0] 
        else:
            return m.S[t] == m.S[t-1] * (1 - m.eta_self * m.delta_t) + m.delta_t / m.Emax * (m.P_c[t-1] * m.eta_conv - m.P_d[t-1] / m.eta_conv) 
        
    m.storage_state = pyo.Constraint(m.t, rule=storage_state_rule, doc='State of charge constraint')


    def storage_power_rule(m, t):
        return m.P_d[t] * m.delta_t <= m.Emax * m.S[t]
    m.storage_power = pyo.Constraint(m.t, rule=storage_power_rule, doc='Storage power constraint')


    def boundary_conditions_rule(m, t):
        if t == len(m.t) - 1:
            return m.S[t] == input_scalar['Starting/Ending SOC'].iloc[0] 
        else:
            return pyo.Constraint.Skip
        
    m.boundary_conditions = pyo.Constraint(m.t, rule=boundary_conditions_rule, doc='SOC boundary conditions constraint')

    def max_charge_power_rule(m, t):   
        return m.P_c[t] <= m.P_c_max * m.u[t]  # Charge when u=1
    m.max_charge_power = pyo.Constraint(m.t, rule=max_charge_power_rule, doc='Max charge power constraint')

    def max_discharge_power_rule(m, t):
        return m.P_d[t] <= m.P_d_max * (1 - m.u[t])  # Discharge when u = 0
    m.max_discharge_power = pyo.Constraint(m.t, rule=max_discharge_power_rule, doc='Max discharge power constraint')

    def soc_min_rule(m, t):
        return m.S[t] >= m.SOC_min
    m.soc_min = pyo.Constraint(m.t, rule=soc_min_rule, doc='Min SOC constraint')

    def soc_max_rule(m, t):
        return m.S[t] <= m.SOC_max
    m.soc_max = pyo.Constraint(m.t, rule=soc_max_rule, doc='Max SOC constraint')




    # Define objective function 
    m.objective = pyo.Objective(rule=lambda m: sum(m.LMP[t] * m.P_m[t] * m.delta_t for t in m.t), sense=pyo.minimize, doc='Cost Minimization')

    # Solve the model 
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(m, tee=True)
    return m, results



