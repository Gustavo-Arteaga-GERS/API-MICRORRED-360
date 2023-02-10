from flask import Blueprint, render_template, request, flash, redirect, url_for, Flask,jsonify,redirect, make_response
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import random
from pathlib import Path
import xlwt
from xlwt import Workbook
import xlsxwriter
import datetime
from pulp import *

app = Flask(__name__)

@app.route("/calculator" , methods = ['POST'])
def microgrid():

    # Parámetros de entrada
    jsonInput = request.get_json()

    # Ingresar nivel de radiación solar:

        # 1 - 1.5 - 2.0 kWh/m2 día
        # 2 - 2.0 - 2.5 kWh/m2 día
        # 3 - 2.5 - 3.0 kWh/m2 día
        # 4 - 3.0 - 3.5 kWh/m2 día
        # 5 - 3.5 - 4.0 kWh/m2 día
        # 6 - 4.0 - 4.5 kWh/m2 día
        # 7 - 4.5 - 5.0 kWh/m2 día
        # 8 - 5.0 - 5.5 kWh/m2 día
        # 9 - 5.5 - 6.0 kWh/m2 día
    
    radiation_level = jsonInput['radiation_level']
    
    # Ingresar economic_level socioeconómico (1, 2...6):

    economic_level = jsonInput['economic_level']

    # Ingresar empresa comercializadora de energía:

        # 1 - CELSIA
        # 2 - ENEL
        # 3 - AIR-E
        # 4 - EMCALI
        # 5 - EPM
        # 6 - Promedio

    energy_company = jsonInput['energy_company']

    # Costos de inversión anualizados con tasa de descuento de 12%, 20 años:
    Cpv = 361370  # [$/kW]
    Cpvh = 402760  # [$/kW]
    Ckwh = 75941  # [$/kWh]
    Cinvc = 206173  # [$/kW]

    # Tarifas de energía (para consumo de subsistencia $/kWh), por estrato y empresa comercializadora:

    # CELSIA:
    if economic_level == 1 and energy_company == 1:
        Cred = 319.6
    elif economic_level == 2 and energy_company == 1:
        Cred = 399.5
    elif economic_level == 3 and energy_company == 1:
        Cred = 679.2
    elif economic_level == 4 and energy_company == 1:
        Cred = 799.0
    elif economic_level == 5 and energy_company == 1:
        Cred = 958.8
    elif economic_level == 6 and energy_company == 1:
        Cred = 958.8

    # ENEL:
    if economic_level == 1 and energy_company == 2:
        Cred = 297.1
    elif economic_level == 2 and energy_company == 2:
        Cred = 371.4
    elif economic_level == 3 and energy_company == 2:
        Cred = 631.3
    elif economic_level == 4 and energy_company == 2:
        Cred = 742.7
    elif economic_level == 5 and energy_company == 2:
        Cred = 891.3
    elif economic_level == 6 and energy_company == 2:
        Cred = 891.3

    # AIR-E:
    if economic_level == 1 and energy_company == 3:
        Cred = 343.4
    elif economic_level == 2 and energy_company == 3:
        Cred = 429.2
    elif economic_level == 3 and energy_company == 3:
        Cred = 729.6
    elif economic_level == 4 and energy_company == 3:
        Cred = 858.4
    elif economic_level == 5 and energy_company == 3:
        Cred = 1030.0
    elif economic_level == 6 and energy_company == 3:
        Cred = 1030.0

    # EMCALI:
    if economic_level == 1 and energy_company == 4:
        Cred = 334.8
    elif economic_level == 2 and energy_company == 4:
        Cred = 418.5
    elif economic_level == 3 and energy_company == 4:
        Cred = 677.5
    elif economic_level == 4 and energy_company == 4:
        Cred = 797.1
    elif economic_level == 5 and energy_company == 4:
        Cred = 956.5
    elif economic_level == 6 and energy_company == 4:
        Cred = 956.5

    # EPM:
    if economic_level == 1 and energy_company == 5:
        Cred = 307.6
    elif economic_level == 2 and energy_company == 5:
        Cred = 384.5
    elif economic_level == 3 and energy_company == 5:
        Cred = 651.0
    elif economic_level == 4 and energy_company == 5:
        Cred = 765.9
    elif economic_level == 5 and energy_company == 5:
        Cred = 919.0
    elif economic_level == 6 and energy_company == 5:
        Cred = 919.0

    # Promedio comercializadoras:
    if economic_level == 1 and energy_company == 6:
        Cred = 320.5
    elif economic_level == 2 and energy_company == 6:
        Cred = 400.6
    elif economic_level == 3 and energy_company == 6:
        Cred = 673.7
    elif economic_level == 4 and energy_company == 6:
        Cred = 729.6
    elif economic_level == 5 and energy_company == 6:
        Cred = 951.1
    elif economic_level == 6 and energy_company == 6:
        Cred = 951.1

    # Tarifa de venta de energía [$/kWh]
    Cout = Cred * 0.37  

    # Otros parámetros:
    etaC = 0.9  # Eficiencia de carga de la batería
    etaD = 1/0.9  # Eficiencia de descarga de la batería
    rMax = 10  # Límite de rampa del generador PV [2%/min, 5%/min, 10%/min, 20%/min]    
    EbatMin = 0.06  # Energía mínima de la batería: Batería de GEL de 12V y 5Ah
    horasFalla = 2  # Duración de las fallas [horas]
    deltaT = 1  # Resolución de los perfiles de carga y generación
    horizonte = list(range(8760))  # Lista de 1 a 8760
    M = 1e6  # Parámetro requerido para el método Big-M
    FE = 0.126 # Factor de emisiones equivalente para Col. 2021 [tonCO2/MWh]

    # Lectura del perfil de carga y perfil solar (base de datos local):
    sheet_1 = pd.read_excel('Perfiles.xlsx', sheet_name=0)

    # Lógica para la selección de los perfiles de carga y generación:

    # Perfil PV de acuerdo al nivel de radiación ingresado:
    if radiation_level == 1:
        Pv1kw = sheet_1['Solar_1'].dropna().to_numpy()
    elif radiation_level == 2:
        Pv1kw = sheet_1['Solar_2'].dropna().to_numpy()
    elif radiation_level == 3:
        Pv1kw = sheet_1['Solar_3'].dropna().to_numpy()
    elif radiation_level == 4:
        Pv1kw = sheet_1['Solar_4'].dropna().to_numpy()
    elif radiation_level == 5:
        Pv1kw = sheet_1['Solar_5'].dropna().to_numpy()
    elif radiation_level == 6:
        Pv1kw = sheet_1['Solar_6'].dropna().to_numpy()
    elif radiation_level == 7:
        Pv1kw = sheet_1['Solar_7'].dropna().to_numpy()
    elif radiation_level == 8:
        Pv1kw = sheet_1['Solar_8'].dropna().to_numpy()
    elif radiation_level == 9:
        Pv1kw = sheet_1['Solar_9'].dropna().to_numpy()

    # Perfil de carga de acuerdo al economic_level ingresado:
    if economic_level == 1:
        Dem = sheet_1['Demanda_1'].dropna().to_numpy()
    elif economic_level == 2:
        Dem = sheet_1['Demanda_2'].dropna().to_numpy()
    elif economic_level == 3:
        Dem = sheet_1['Demanda_3'].dropna().to_numpy()
    elif economic_level == 4:
        Dem = sheet_1['Demanda_4'].dropna().to_numpy()
    elif economic_level == 5:
        Dem = sheet_1['Demanda_5'].dropna().to_numpy()
    elif economic_level == 6:
        Dem = sheet_1['Demanda_6'].dropna().to_numpy()

    # Línea de código con orden preestablecido de datos (requerida cuando se optimiza con CBC):
    MiOrden = sheet_1['Orden'].dropna().to_numpy()

    #  Función crear falla en la red eléctrica principal:
    def crearFalla(h, n, dt):
        h = int(h // dt)
        vFalla = np.ones(n)
        inicio = random.randint(0, 8759)
        vFalla[inicio:inicio + h] = np.zeros(h)
        fechaFalla = datetime.timedelta(hours=inicio * dt)

        return vFalla, fechaFalla, h, inicio
    
    vFalla, fechaFalla, h, inicio = crearFalla(horasFalla, len(Dem), deltaT)

    # Lógica para indexar la fecha de la falla de acuerdo a la hora de inicio anual:
    PerfilDia = sheet_1['Dia'].dropna().to_numpy()
    PerfilMes = sheet_1['Mes'].dropna().to_numpy()
    PerfilHora = sheet_1['Hora'].dropna().to_numpy()
    Dia = int(PerfilDia[inicio])
    Mes = int(PerfilMes[inicio])
    Hora = int(PerfilHora[inicio])

    # Modelo de optimización escrito con sintaxis Pulp
    prob = LpProblem("model_a", LpMinimize)

    # Definición de variables
    Pred = LpVariable.dicts('Pred', horizonte, lowBound=0, cat=LpContinuous)
    Pout = LpVariable.dicts("Pout", horizonte, lowBound=0, cat=LpContinuous)
    Pinst = LpVariable("Pinst", 0, 3, LpContinuous)
    Pinsth = LpVariable("Pinsth", 0, 3, LpContinuous)
    Ebat = LpVariable("Ebat", 0, 1e4, LpContinuous)
    Pinvc = LpVariable("Pinvc", 0, 1e4, LpContinuous)
    xc = LpVariable.dicts("xc", horizonte, lowBound=0, cat=LpBinary)
    xd = LpVariable.dicts("xd", horizonte, lowBound=0, cat=LpBinary)
    x_pinst = LpVariable("x_pinst", 0, 1e4, LpBinary)
    x_pinsth = LpVariable("x_pinsth", 0, 1e4, LpBinary)
    Pch = LpVariable.dicts("Pch", horizonte, lowBound=0, cat=LpContinuous)
    Pdch = LpVariable.dicts("Pdch", horizonte, lowBound=0, cat=LpContinuous)
    E = LpVariable.dicts("E", horizonte, lowBound=0, cat=LpContinuous)
    EbatRes = LpVariable("EbatRes", 0, 1e4, LpContinuous)
    x_bat = LpVariable("x_bat", 0, 1e4, LpBinary)
    x_tmp = LpVariable("x_tmp", 0, 1e4, LpBinary)
    tmp = LpVariable("tmp")
    tmp2 = LpVariable("tmp2")
    tmp3 = LpVariable("tmp3")

    # Definición de restricciones:

    # Conjunto 1 de restricciones (Capacidad de la batería):
    prob += Ebat >= min(0, EbatMin), "cEbatRes_1"
    prob += Ebat <= M, "cEbatRes_2"
    prob += Ebat >= EbatMin * x_bat, "cEbatRes_3"
    prob += Ebat <= M * x_bat, "cEbatRes_4"
    prob += Ebat >= EbatRes - (1 - x_bat) * M, "cEbatRes_5"
    prob += Ebat <= EbatRes - (1 - x_bat) * EbatMin, "cEbatRes_6"
    prob += Ebat <= EbatRes + (1 - x_bat) * M, "cEbatRes_7"
    
    # Conjunto 2 de restricciones (Capacidad del sistema PV):
    prob += Pinst <= M * x_pinst, "cPinst"
    prob += Pinsth <= M * x_pinsth, "cPinsth"
    prob += x_pinst + x_pinsth <= 1, "cBin_2"

    # Conjunto 3 de restricciones (Horas de falla):
    if horasFalla > 0:
        prob += Ebat >= Pinst * (1 / (4 * rMax / 100 * 60)), "EbatMin"
        prob += E[0] == 0.9 * Ebat, "cargaInicial"
        prob += tmp == 1 / 2 * Pinst, "tmp"
        prob += tmp2 == max(Dem[inicio:inicio + h]), "tmp2"
        prob += tmp3 <= tmp, "tmp3_1"
        prob += tmp3 <= tmp2, "tmp3_2"
        prob += tmp3 >= tmp - M * x_tmp, "tmp3_3"
        prob += tmp3 >= tmp2 - M * (1 - x_tmp), "tmp3_4"
        prob += Pinvc >= tmp3, "PinvcMin"

    # Conjunto 4 de restricciones (Balance de potencia, potencia inyectada, carga, descarga y energía de la batería):
    for i in range(len(Dem)):
        if vFalla[i] == 0:
            prob += Pinst * Pv1kw[i] + Pinsth * Pv1kw[i] - Pch[i] + Pdch[i] >= Dem[i], "cPotInq" + str(i)
            prob += Pred[i] == 0, "cPredFalla" + str(i)
            prob += Pout[i] == 0, "cPoutFalla" + str(i)
        else:
            prob += Pinst * Pv1kw[i] + Pinsth * Pv1kw[i] - Pch[i] + Pdch[i] - Pout[i] + Pred[i] == Dem[i], "cPotEq" + str(i)
        prob += Pch[i] <= M * xc[i], "cBatChM" + str(i)
        prob += Pch[i] <= Pinvc + Pinsth, "cBatCh" + str(i)
        prob += Pdch[i] <= M * xd[i], "cBatDchM" + str(i)
        prob += Pdch[i] <= Pinvc + Pinsth, "cBatDch" + str(i)
        prob += xc[i] + xd[i] <= 1, "cBin_1" + str(i)
        prob += Pout[i] <= Pinst * Pv1kw[i] + Pinsth * Pv1kw[i], "cPout" + str(i)
        prob += E[i] <= 0.9 * Ebat, "cEmax" + str(i)
        prob += E[i] >= 0.2 * Ebat, "cEmin" + str(i)

    # Conjunto 5 de restricciones (Carga inicial de la batería):
    for i in range(len(Dem)):
        if i == 0:
            if horasFalla > 0:
                prob += E[i] == 0.9 * Ebat + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)
            else:
                prob += E[i] == 0.2 * Ebat + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)
        else:
            prob += E[i] == E[i - 1] + Pch[i] * etaC * deltaT - (Pdch[i] * etaD) * deltaT, "cE" + str(i)

    # Función Objetivo:
    prob += lpSum([Cred * Pred[i] * deltaT for i in horizonte]) + Cpv * Pinst + Cpvh * Pinsth + Ckwh * Ebat + Cinvc * Pinvc - lpSum([Cout * Pout[i] * deltaT for i in horizonte])

    # Solución del problema de optimización:
    prob.writeLP("model_a.lp")
    prob.solve(GUROBI_CMD())  # Solución del problema con Gurobi
    #prob.solve(PULP_CBC_CMD(fracGap = 0.00001, maxSeconds = 60, threads = None)) # Solución del problema con CBC

    # Mostrar en pantalla el estado de la solución:
    # print("Estado:", LpStatus[prob.status])

    # Re-defición de variables que dependen del tiempo (estado inicial vacío):
    Pred_ = []
    Pout_ = []
    Pch_ = []
    Pdch_ = []
    E_ = []

    # Bucle para extraer las variables del diccionario Pulp
    for v in prob.variables():
        if v.name == "Pinst":
            x1 = v.varValue
        elif v.name == "Pinsth":
            x3 = v.varValue
        elif v.name == "Ebat":
            x2 = v.varValue
        elif v.name == "Pinvc":
            x4 = v.varValue
        elif v.name.startswith("Pred_"):
            Pred_.append(v.varValue)
        elif v.name.startswith("Pout_"):
            Pout_.append(v.varValue)
        elif v.name.startswith("Pch_"):
            Pch_.append(v.varValue)
        elif v.name.startswith("Pdch_"):
            Pdch_.append(v.varValue)
        elif v.name.startswith("E_"):
            E_.append(v.varValue)

    # Código que reorganiza la lista de valores de acuerdo al subindice de la variable:
    Pred_ = [Pred_[i] for i in MiOrden]
    Pout_ = [Pout_[i] for i in MiOrden]
    Pch_ = [Pch_[i] for i in MiOrden]
    Pdch_ = [Pdch_[i] for i in MiOrden]
    E_ = [E_[i] for i in MiOrden]

    # Función objetivo recalculada:
    obj = Cred * np.sum(Pred_) * deltaT + Cpv * x1 + Cpvh * x3 + Ckwh * x2 + Cinvc * x4 - Cout * np.sum(Pout_) * deltaT

    # Conjunto de resultados:
    scores = (x1, x3, x2, x4, np.sum(Dem[inicio:inicio + h]) * deltaT, np.sum(Pred_) * deltaT, np.sum(Pout_) * deltaT,
            np.sum(Pch_) * deltaT, np.sum(Pdch_) * deltaT, obj, Cpv * x1 + Cpvh * x3, Ckwh * x2, Cinvc * x4,
            Cpv * x1 + Cpvh * x3 + Ckwh * x2 + Cinvc * x4, (Cpv * x1 + Cpvh * x3 + Ckwh * x2 + Cinvc * x4),
            np.sum(Pred_) * Cred * deltaT, np.sum(Pout_) * Cout * deltaT)
    
    Efalla = np.sum(Dem[inicio:inicio + h]) * deltaT
    Eimpor = np.sum(Pred_) * deltaT
    Eexpor = np.sum(Pout_) * deltaT
    Cinversion = Cpv * x1 + Cpvh * x3 + Cinvc * x4
    Cbanco = Ckwh * x2
    CinversionT = Cinversion + Cbanco
    Ccompra = np.sum(Pred_) * Cred * deltaT
    Cventa = np.sum(Pout_) * Cout * deltaT
    AhorroEner = np.sum(Dem) * deltaT - Eimpor
    AhorroEco = np.sum(Dem) * deltaT * Cred - obj
    AhorroEmi = AhorroEner * FE

    # Condicional para mostrar resultados de PV con tipo de inversor(es)
    if x1 != 0:
        xinv = x1
        TipoInv = "ongrid_and_charger_inverter"
        xinvc = "{:.2f}".format(x4)
    else:
        xinv = x3
        TipoInv = "hybrid"
        xinvc = "charger_inverter_is_not_required"

    # Mostrar resultados
    response = [
        {
        "energy_saving": "{:,.0f}".format(AhorroEner),
        "economic_saving": "{:,.0f}".format(AhorroEco),
        "environmental_saving": "{:.0f}".format(AhorroEmi),
        },

        {
        "pv_power": "{:.2f}".format(xinv),
        "inverter_type": TipoInv,
        "charger_inverter_power": xinvc,
        "battery_bank_power": "{:.2f}".format(x2),
        },

        {"failure_day": Dia, 
        "failure_month": Mes, 
        "failure_hour": Hora,
        "failure_duration": "{:.0f}".format(horasFalla),
        "failure_energy": "{:.2f}".format(Efalla),
        },

        {
        "imported_energy": "{:,.0f}".format(Eimpor),
        "exported_energy": "{:,.0f}".format(Eexpor),
        "energy_purchases": "{:,.0f}".format(Ccompra),
        "energy_sales": "{:,.0f}".format(Cventa),
        "pv_and_inverter_cost": "{:,.0f}".format(Cinversion),
        "battery_bank_cost": "{:,.0f}".format(Cbanco),
        "investment_cost": "{:,.0f}".format(CinversionT),       
        #"total_cost": "{:,.0f}".format(obj),
        }
    ]

    res = make_response(jsonify(response), 200)
    return res

if __name__ == '__main__':
    # app.run(debug = True, port = 5000)
    app.run(debug = True, host='0.0.0.0', port = 5000)