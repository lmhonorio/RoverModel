import numpy as np
import matplotlib.pyplot as plt
from roverclass import MotorModel, SkidSteerRoverModel, EvaluateRoverParameters



param_names = [
    "I - momento de inércia", "m - massa", "r - raio da roda", "L - entre-eixos",
    "kt", "ktarget_velocity", "time_constant", "torque_scale",
    "linear_force_scale", "angular_force_scale", "rwheel", "lwheel",
    "C_r", "C_omega", "r_FR", "r_FL", "r_RL", "r_RR"
]

individual = [3.646465528646285, 9.65814489363485, 0.5, 0.8224688903921193, 0.8940170780417092, 0.2012976170620885, 0.04590521246354088, 0.028442619242259888, 2.0, 2.928117243253443, 2.5166122073282535, 3.05, 3.937398180335976, 3.5484531665805896, 0.24452204198298327, 0.3997932866438163, 0.3093710438492553, 0.40742299063002274]
individual = [7.38843438, 6.23076006, 0.49926106, 0.61531902, 1.48013604, 0.66269152, 0.0760918,  0.01031666, 1.56940289, 2.97840951, 2.83477282, 1.98279523, 6.81041878, 6.03190296, 0.42540815, 0.21179836, 0.21279214, 0.31565927]
individual = [6.61532699, 5.01771887, 0.56332537, 0.9938069,  1.6791308,  0.6985648,  0.20019771, 0.01197814, 1.58822935, 2.96753579, 2.21756364, 2.80395149,  7.99779026, 4.36294693, 0.49528919, 0.22287651, 0.46269301, 0.21101814]

s = "[5.93989374 5.05153555 0.32709552 0.90555982 1.0232813 0.68036464 0.352118 0.01000042 1.28572813 1.16333492 2.97910925 3.04047996 7.98131271 1.01999508 0.24093472 0.25974187 0.24125928 0.25962003]"

#
# # Converte direto:
res = np.fromstring(s.strip("[]"), sep=' ')
print(res.tolist())

individual = res

excel_file = "../planilhas/sequencia_1_1.xlsx"
sheet_name = "Sheet1"

rover = EvaluateRoverParameters(excel_file, sheet_name)

erro_total = rover.evaluate(individual)

print(erro_total)
rover.imprimir_resultado(individual,param_names)

rover.plot_results()


