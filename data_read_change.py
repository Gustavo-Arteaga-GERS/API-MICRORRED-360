from flask import Blueprint, render_template, request, flash, redirect, url_for, Flask,jsonify,redirect, make_response
import json
import numpy as np
import pandas as pd
import random
import datetime
from pulp import *

app = Flask(__name__)

@app.route("/calculator" , methods = ['POST'])
def microgrid():

    # input_data:
    jsonInput = request.get_json()

    # input_radiation_level:

        # 1 - 1.5 - 2.0 kWh/m2 day
        # 2 - 2.0 - 2.5 kWh/m2 day
        # 3 - 2.5 - 3.0 kWh/m2 day
        # 4 - 3.0 - 3.5 kWh/m2 day
        # 5 - 3.5 - 4.0 kWh/m2 day
        # 6 - 4.0 - 4.5 kWh/m2 day
        # 7 - 4.5 - 5.0 kWh/m2 day
        # 8 - 5.0 - 5.5 kWh/m2 day
        # 9 - 5.5 - 6.0 kWh/m2 day
    
    radiation_level = jsonInput['radiation_level']
    
    # input_economic_level_(1, 2...6):

    economic_level = jsonInput['economic_level']

    # input_energy_company:

        # 1 - CELSIA
        # 2 - ENEL
        # 3 - AIR-E
        # 4 - EMCALI
        # 5 - EPM
        # 6 - Average

    energy_company = jsonInput['energy_company']

    # annualized_investment_costs_at_12%_discount_rate_20_years:
    Cpv = 361370  # [$/kW]
    Cpvh = 402760  # [$/kW]
    Ckwh = 75941  # [$/kWh]
    Cinvc = 206173  # [$/kW]

    # energy_tariffs_(subsistence_consumption_$/kWh)_by_economic_level_and_energy_company:

    # CELSIA:
    if economic_level == 1 and energy_company == 1:
        Cgrid = 319.6
    elif economic_level == 2 and energy_company == 1:
        Cgrid = 399.5
    elif economic_level == 3 and energy_company == 1:
        Cgrid = 679.2
    elif economic_level == 4 and energy_company == 1:
        Cgrid = 799.0
    elif economic_level == 5 and energy_company == 1:
        Cgrid = 958.8
    elif economic_level == 6 and energy_company == 1:
        Cgrid = 958.8

    # ENEL:
    if economic_level == 1 and energy_company == 2:
        Cgrid = 297.1
    elif economic_level == 2 and energy_company == 2:
        Cgrid = 371.4
    elif economic_level == 3 and energy_company == 2:
        Cgrid = 631.3
    elif economic_level == 4 and energy_company == 2:
        Cgrid = 742.7
    elif economic_level == 5 and energy_company == 2:
        Cgrid = 891.3
    elif economic_level == 6 and energy_company == 2:
        Cgrid = 891.3

    # AIR-E:
    if economic_level == 1 and energy_company == 3:
        Cgrid = 343.4
    elif economic_level == 2 and energy_company == 3:
        Cgrid = 429.2
    elif economic_level == 3 and energy_company == 3:
        Cgrid = 729.6
    elif economic_level == 4 and energy_company == 3:
        Cgrid = 858.4
    elif economic_level == 5 and energy_company == 3:
        Cgrid = 1030.0
    elif economic_level == 6 and energy_company == 3:
        Cgrid = 1030.0

    # EMCALI:
    if economic_level == 1 and energy_company == 4:
        Cgrid = 334.8
    elif economic_level == 2 and energy_company == 4:
        Cgrid = 418.5
    elif economic_level == 3 and energy_company == 4:
        Cgrid = 677.5
    elif economic_level == 4 and energy_company == 4:
        Cgrid = 797.1
    elif economic_level == 5 and energy_company == 4:
        Cgrid = 956.5
    elif economic_level == 6 and energy_company == 4:
        Cgrid = 956.5

    # EPM:
    if economic_level == 1 and energy_company == 5:
        Cgrid = 307.6
    elif economic_level == 2 and energy_company == 5:
        Cgrid = 384.5
    elif economic_level == 3 and energy_company == 5:
        Cgrid = 651.0
    elif economic_level == 4 and energy_company == 5:
        Cgrid = 765.9
    elif economic_level == 5 and energy_company == 5:
        Cgrid = 919.0
    elif economic_level == 6 and energy_company == 5:
        Cgrid = 919.0

    # average_energy_company:
    if economic_level == 1 and energy_company == 6:
        Cgrid = 320.5
    elif economic_level == 2 and energy_company == 6:
        Cgrid = 400.6
    elif economic_level == 3 and energy_company == 6:
        Cgrid = 673.7
    elif economic_level == 4 and energy_company == 6:
        Cgrid = 729.6
    elif economic_level == 5 and energy_company == 6:
        Cgrid = 951.1
    elif economic_level == 6 and energy_company == 6:
        Cgrid = 951.1

    # energy_sales_tariff_[$/kWh]
    Cout = Cgrid * 0.37  

    # others_input_data:
    etaC = 0.9
    etaD = 1/0.9
    rMax = 10  
    EbatMin = 0.06
    hoursFailure = 2
    deltaT = 1
    horizon = list(range(8760))
    M = 1e6
    FE = 0.126

    # load_and_solar_profile_reading_(local_database):
    # pv_profile_according_to_the_radiation_level_entered:
    if radiation_level == 1:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_1']).dropna().to_numpy()
    elif radiation_level == 2:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_2']).dropna().to_numpy()
    elif radiation_level == 3:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_3']).dropna().to_numpy()
    elif radiation_level == 4:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_4']).dropna().to_numpy()
    elif radiation_level == 5:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_5']).dropna().to_numpy()
    elif radiation_level == 6:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_6']).dropna().to_numpy()
    elif radiation_level == 7:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_7']).dropna().to_numpy()
    elif radiation_level == 8:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_8']).dropna().to_numpy()
    elif radiation_level == 9:
        Pv1kw = pd.read_csv('profiles.csv', sep=",", usecols = ['Solar_9']).dropna().to_numpy()

    # load_profile_according_to_the_entered_economic_level:
    if economic_level == 1:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_1']).dropna().to_numpy()
    elif economic_level == 2:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_2']).dropna().to_numpy()
    elif economic_level == 3:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_3']).dropna().to_numpy()
    elif economic_level == 4:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_4']).dropna().to_numpy()
    elif economic_level == 5:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_5']).dropna().to_numpy()
    elif economic_level == 6:
        Dem = pd.read_csv('profiles.csv', sep=",", usecols = ['Demand_6']).dropna().to_numpy()

    # line_of_code_with_preset_order_of-data:
    myOrder = pd.read_csv('profiles.csv', sep=",", usecols = ['Order']).dropna().to_numpy().tolist()
    myOrder = sum(myOrder, [])

    #  Function_create_failure_in_power_grid:
    def createFailure(h, n, dt):
        h = int(h // dt)
        vFailure = np.ones(n)
        start_ = random.randint(0, 8759)
        vFailure[start_:start_ + h] = np.zeros(h)
        dateFailure = datetime.timedelta(hours=start_ * dt)

        return vFailure, dateFailure, h, start_
    
    vFailure, dateFailure, h, start_ = createFailure(hoursFailure, len(Dem), deltaT)

    # logic_for_indexing_the_failure_date_according_to_the_annual_start_time:
    dailyProfile = pd.read_csv('profiles.csv', sep=",", usecols = ['Day']).dropna().to_numpy()
    monthProfile = pd.read_csv('profiles.csv', sep=",", usecols = ['Month']).dropna().to_numpy()
    hourlyProfile = pd.read_csv('profiles.csv', sep=",", usecols = ['Hour']).dropna().to_numpy()
    day = int(dailyProfile[start_])
    month = int(monthProfile[start_])
    hour = int(hourlyProfile[start_])

    # optimization_model_written_with_Pulp_syntax
    prob = LpProblem("model_a", LpMinimize)

    # variables_definition
    Pgrid = LpVariable.dicts('Pgrid', horizon, lowBound=0, cat=LpContinuous)
    Pout = LpVariable.dicts("Pout", horizon, lowBound=0, cat=LpContinuous)
    Pinst = LpVariable("Pinst", 0, 3, LpContinuous)
    Pinsth = LpVariable("Pinsth", 0, 3, LpContinuous)
    Ebat = LpVariable("Ebat", 0, 1e4, LpContinuous)
    Pinvc = LpVariable("Pinvc", 0, 1e4, LpContinuous)
    xc = LpVariable.dicts("xc", horizon, lowBound=0, cat=LpBinary)
    xd = LpVariable.dicts("xd", horizon, lowBound=0, cat=LpBinary)
    x_pinst = LpVariable("x_pinst", 0, 1e4, LpBinary)
    x_pinsth = LpVariable("x_pinsth", 0, 1e4, LpBinary)
    Pch = LpVariable.dicts("Pch", horizon, lowBound=0, cat=LpContinuous)
    Pdch = LpVariable.dicts("Pdch", horizon, lowBound=0, cat=LpContinuous)
    E = LpVariable.dicts("E", horizon, lowBound=0, cat=LpContinuous)
    EbatRes = LpVariable("EbatRes", 0, 1e4, LpContinuous)
    x_bat = LpVariable("x_bat", 0, 1e4, LpBinary)
    x_tmp = LpVariable("x_tmp", 0, 1e4, LpBinary)
    tmp = LpVariable("tmp")
    tmp2 = LpVariable("tmp2")
    tmp3 = LpVariable("tmp3")

    # definition_of_constraints:

    # constrainst_set_1_(battery_capacity):
    prob += Ebat >= min(0, EbatMin), "cEbatRes_1"
    prob += Ebat <= M, "cEbatRes_2"
    prob += Ebat >= EbatMin * x_bat, "cEbatRes_3"
    prob += Ebat <= M * x_bat, "cEbatRes_4"
    prob += Ebat >= EbatRes - (1 - x_bat) * M, "cEbatRes_5"
    prob += Ebat <= EbatRes - (1 - x_bat) * EbatMin, "cEbatRes_6"
    prob += Ebat <= EbatRes + (1 - x_bat) * M, "cEbatRes_7"
    
    # constrainst_set_2_(pv_capacity):
    prob += Pinst <= M * x_pinst, "cPinst"
    prob += Pinsth <= M * x_pinsth, "cPinsth"
    prob += x_pinst + x_pinsth <= 1, "cBin_2"

    # constrainst_set_3_(failure_hours):
    if hoursFailure > 0:
        prob += Ebat >= Pinst * (1 / (4 * rMax / 100 * 60)), "EbatMin"
        prob += E[0] == 0.9 * Ebat, "cargaInicial"
        prob += tmp == 1 / 2 * Pinst, "tmp"
        prob += tmp2 == max(Dem[start_:start_ + h]), "tmp2"
        prob += tmp3 <= tmp, "tmp3_1"
        prob += tmp3 <= tmp2, "tmp3_2"
        prob += tmp3 >= tmp - M * x_tmp, "tmp3_3"
        prob += tmp3 >= tmp2 - M * (1 - x_tmp), "tmp3_4"
        prob += Pinvc >= tmp3, "PinvcMin"

    # constrainst_set_4_(power_balance_injected_power_charge_discharge_and_battery_energy):
    for i in range(len(Dem)):
        if vFailure[i] == 0:
            prob += Pinst * Pv1kw[i] + Pinsth * Pv1kw[i] - Pch[i] + Pdch[i] >= Dem[i], "cPotInq" + str(i)
            prob += Pgrid[i] == 0, "cPgridFalla" + str(i)
            prob += Pout[i] == 0, "cPoutFalla" + str(i)
        else:
            prob += Pinst * Pv1kw[i] + Pinsth * Pv1kw[i] - Pch[i] + Pdch[i] - Pout[i] + Pgrid[i] == Dem[i], "cPotEq" + str(i)
        prob += Pch[i] <= M * xc[i], "cBatChM" + str(i)
        prob += Pch[i] <= Pinvc + Pinsth, "cBatCh" + str(i)
        prob += Pdch[i] <= M * xd[i], "cBatDchM" + str(i)
        prob += Pdch[i] <= Pinvc + Pinsth, "cBatDch" + str(i)
        prob += xc[i] + xd[i] <= 1, "cBin_1" + str(i)
        prob += Pout[i] <= Pinst * Pv1kw[i] + Pinsth * Pv1kw[i], "cPout" + str(i)
        prob += E[i] <= 0.9 * Ebat, "cEmax" + str(i)
        prob += E[i] >= 0.2 * Ebat, "cEmin" + str(i)

    # constrainst_set_5_(initial_SOC):
    for i in range(len(Dem)):
        if i == 0:
            if hoursFailure > 0:
                prob += E[i] == 0.9 * Ebat + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)
            else:
                prob += E[i] == 0.2 * Ebat + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)
        else:
            prob += E[i] == E[i - 1] + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)

    # objective_function:
    prob += lpSum([Cgrid * Pgrid[i] * deltaT for i in horizon]) + Cpv * Pinst + Cpvh * Pinsth + Ckwh * Ebat + Cinvc * Pinvc - lpSum([Cout * Pout[i] * deltaT for i in horizon])

    # optimization_problem_solution:
    prob.solve(PULP_CBC_CMD(fracGap = 0.00001, maxSeconds = 120, threads = None))

    # re-defining_time-dependent_variables_(empty_initial_state):
    Pgrid_ = []
    Pout_ = []
    Pch_ = []
    Pdch_ = []
    E_ = []

    # loop_to_extract_variables_from_the_Pulp_dictionary:
    for v in prob.variables():
        if v.name == "Pinst":
            x1 = v.varValue
        elif v.name == "Pinsth":
            x3 = v.varValue
        elif v.name == "Ebat":
            x2 = v.varValue
        elif v.name == "Pinvc":
            x4 = v.varValue
        elif v.name.startswith("Pgrid_"):
            Pgrid_.append(v.varValue)
        elif v.name.startswith("Pout_"):
            Pout_.append(v.varValue)
        elif v.name.startswith("Pch_"):
            Pch_.append(v.varValue)
        elif v.name.startswith("Pdch_"):
            Pdch_.append(v.varValue)
        elif v.name.startswith("E_"):
            E_.append(v.varValue)

    # logic_that_reorganizes_the_list_of_values_according_to_the_variable's_subindex:
    Pgrid_ = [Pgrid_[i] for i in myOrder]
    Pout_ = [Pout_[i] for i in myOrder]
    Pch_ = [Pch_[i] for i in myOrder]
    Pdch_ = [Pdch_[i] for i in myOrder]
    E_ = [E_[i] for i in myOrder]

    # recalculated_objective_function:
    obj = Cgrid * np.sum(Pgrid_) * deltaT + Cpv * x1 + Cpvh * x3 + Ckwh * x2 + Cinvc * x4 - Cout * np.sum(Pout_) * deltaT

    # set_of_results:
    failureE = np.sum(Dem[start_:start_ + h]) * deltaT
    importedE = np.sum(Pgrid_) * deltaT
    exportedE = np.sum(Pout_) * deltaT
    investmentC = Cpv * x1 + Cpvh * x3 + Cinvc * x4
    bankC = Ckwh * x2
    investmentT = investmentC + bankC
    purchaseC = np.sum(Pgrid_) * Cgrid * deltaT
    saleC = np.sum(Pout_) * Cout * deltaT
    energySav = np.sum(Dem) * deltaT - importedE
    economicSav = np.sum(Dem) * deltaT * Cgrid - obj
    environSav = energySav * FE

    # Conditional_to_show_PV_results_with_inverter(s)_type:
    if x1 != 0:
        xinv = x1
        InvType = "ongrid_and_charger_inverter"
        xinvc = "{:.2f}".format(x4)
    else:
        xinv = x3
        InvType = "hybrid"
        xinvc = "charger_inverter_is_not_required"

    # Show_results:
    response = [
        {
        "energy_saving": "{:,.0f}".format(energySav),
        "economic_saving": "{:,.0f}".format(economicSav),
        "environmental_saving": "{:.0f}".format(environSav),
        },

        {
        "pv_power": "{:.2f}".format(xinv),
        "inverter_type": InvType,
        "charger_inverter_power": xinvc,
        "battery_bank_power": "{:.2f}".format(x2),
        },

        {"failure_day": day, 
        "failure_month": month, 
        "failure_hour": hour,
        "failure_duration": "{:.0f}".format(hoursFailure),
        "failure_energy": "{:.2f}".format(failureE),
        },

        {
        "imported_energy": "{:,.0f}".format(importedE),
        "exported_energy": "{:,.0f}".format(exportedE),
        "energy_purchases": "{:,.0f}".format(purchaseC),
        "energy_sales": "{:,.0f}".format(saleC),
        "pv_and_inverter_cost": "{:,.0f}".format(investmentC),
        "battery_bank_cost": "{:,.0f}".format(bankC),
        "investment_cost": "{:,.0f}".format(investmentT),       
        }
    ]

    res = make_response(jsonify(response), 200)
    return res

if __name__ == '__main__':
    # app.run(debug = True, port = 5000)
    app.run(debug = True, host='0.0.0.0', port = 5000)