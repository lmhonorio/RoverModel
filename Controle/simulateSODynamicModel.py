import numpy as np
# from sympy.printing.pretty.pretty_symbology import line_width

from Controle.roverdynamics import EvaluateRoverParameters

param_names = [
    "vscale", "vgain", "vtau", "vzeta",
    "ascale", "again", "atau", "azeta"
]

param_bound = [
    (0.01, 0.1),  # vscale
    (0.01, 2.0),  # vgain_L
    (0.01, 2.0),  # vtau_L
    (0.01, 5.0),  # vzeta_L
    (0.001, 0.1),  # ascale
    (0.0, 3.0),  # again_L
    (0.001, 15.0),  # atau_L
    (0.001, 15.0),  # azeta_L
    (0.0, 3.0),  # again_r
    (0.001, 15.0),  # atau_r
    (0.001, 15.0)  # azeta_r
]


# individual =  [0.021027225997049783, 0.4639901836021286, 1.3099957307951422, 1.529476294909712, 0.013038761684299915, 2.0352262266192467, 1.019714796310165, 12.010420843334602]
#
s = "[0.0150199  0.34243617 1.56524389 1.35486333 0.0142325  0.60528597 6.01851242 3.27074781 0.65726919 5.63780158 3.47293682]"



#
# # Converte direto:
res = np.fromstring(s.strip("[]"), sep=' ')
print(res.tolist())

individual =res

excel_file = "../planilhas/sequencia_2_1.xlsx"
sheet_name = "Sheet1"

rover = EvaluateRoverParameters(excel_file, sheet_name,dt=0.05)

erro_total = rover.evaluate(individual)

print(erro_total)
rover.imprimir_resultado(individual,param_names)

rover.plot_results()

















