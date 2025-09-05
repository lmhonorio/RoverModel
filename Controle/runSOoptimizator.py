from roverdynamics import EvaluateRoverParameters
import cma






# 472 - velocidade linear
#Melhores parâmetros encontrados: [0.023319205167229094, 0.8457562523069684, 0.1929447329755878, 10.0]
#                                 [0.05752922841924408, 0.33421583119060905, 0.754640975829705, 2.699310950438279]
#                                 [0.0468962474309844, 0.4348856573075798, 1.0, 2.353877205538026]
#                                 [0.09710433522754894, 0.2070698228434173, 1.1223568257201602, 1.9388901629568023]
#                                 [0.049034415282554866, 0.4031641555625117, 0.9577139068112256, 2.413128307801792]




param_bound = [
    (0.001, 0.1),  # vscale
    (0.01, 2.0),  # vgain_L
    (0.01, 2.0),  # vtau_L
    (0.01, 3.0),  # vzeta_L
    (0.001, 0.1),  # ascale
    (0.0, 1.0),    # again_L
    (0.001, 8.0),  # atau1
    (0.001, 8.0),  # azeta_L
    (0.0, 1.0),  # again_r
    (0.001, 8.0),  # ataur
    (0.001, 8.0)  # azeta_r
]




if __name__ == '__main__':



    xlsx_path = "../planilhas/sequencia_1_1.xlsx"
    # xlsx_path = "./planilhas/sequencia_2_1.xlsx"

    sheet_name = "Sheet1"

    # identificador = GeneticRoverParameterIdentifier(
    #     excel_file=xlsx_path,
    #     sheet_name=sheet_name,
    #     population_size=30,
    #     generations=50,
    #     param_bounds= param_bound
    # )
    #
    # melhores_parametros = identificador.run_genetic_algorithm()
    # print(melhores_parametros)
    #
    #
    # identificador.plot_results()



    lower_bounds = [b[0] for b in param_bound]
    upper_bounds = [b[1] for b in param_bound]
    bounds = [lower_bounds, upper_bounds]

    rover = EvaluateRoverParameters(xlsx_path,sheet_name,dt=0.05)


    def evaluate(individual):
        return rover.evaluate(individual)


    x0 = [0.0150199, 0.34243617, 1.56524389, 1.35486333, 0.0142325, 0.60528597, 6.01851242, 3.27074781, 0.65726919, 5.63780158, 3.47293682]
    sigma0 = 0.2


    res = cma.fmin(evaluate, x0, sigma0, {
        'bounds': bounds,
        'popsize': 100,
        'maxiter': 30
    })

    print(res[0])
    print(res[1])