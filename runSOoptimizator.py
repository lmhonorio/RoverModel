from geneticSOoptimizator import GeneticRoverParameterIdentifier
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
import numpy as np

xlsx_path = "./planilhas/sequencia_14s.xlsx"
# xlsx_path = "./planilhas/sequencia_2_1.xlsx"




# 472 - velocidade linear
#Melhores parâmetros encontrados: [0.023319205167229094, 0.8457562523069684, 0.1929447329755878, 10.0]
#                                 [0.05752922841924408, 0.33421583119060905, 0.754640975829705, 2.699310950438279]
#                                 [0.0468962474309844, 0.4348856573075798, 1.0, 2.353877205538026]
#                                 [0.09710433522754894, 0.2070698228434173, 1.1223568257201602, 1.9388901629568023]
#                                 [0.049034415282554866, 0.4031641555625117, 0.9577139068112256, 2.413128307801792]

param_bounds_linear = [
    (0.01, 0.1),    # scale
    (0.01, 2.0),     # gain_L
    (0.01, 2.0),     # tau_L
    (0.01, 5.0)    #  zeta_L
]


# 653 - velocidade angular
# [0.046784634670426525, 0.012994968946145026, 1.9682300864696605, 0.020515564595477175]
# [0.08441857526100693, 0.03891163460044581, 2.0, 2.0]
# [0.03315811690034011, 0.26720455706332963, 5.0, 5.0]

param_bounds_angular = [
    (0.001, 0.1),    # scale
    (-3.0, 3.0),     # gain_L
    (0.001, 15.0),     # tau_L
    (0.001, 15.0)    #  zeta_L
]




if __name__ == '__main__':

    islinear = False

    identificador = GeneticRoverParameterIdentifier(
        excel_file=xlsx_path,
        sheet_name="Sheet1",
        population_size=60,
        generations=30,
        isLinear= islinear,
        param_bounds= param_bounds_linear if islinear else param_bounds_angular
    )

    melhores_parametros = identificador.run_genetic_algorithm()
    print(melhores_parametros)


    identificador.plot_results()