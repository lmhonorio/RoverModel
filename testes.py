# Exemplo de uso:
from geneticoptimizator import GeneticRoverParameterIdentifier
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
import numpy as np


def filtrar_por_tempo(Time, *arrays, t_min=0.0, t_max=9999.0):
    mask = (Time >= t_min) & (Time <= t_max)
    return (Time[mask],) + tuple(arr[mask] for arr in arrays)




if __name__ == "__main__":
    file_bin = "./Arquivossuporte/sequencia1.bin"
    mavlog = MavLog(file_bin)
    mavlog.parse()

    xkf1 = mavlog.get("XKF1")
    rcou = mavlog.get("RCOU")

    mask = ~np.isnan(xkf1["VN"]) & ~np.isnan(xkf1["VE"]) & ~np.isnan(xkf1["Yaw"])
    VN = xkf1["VN"][mask]
    VE = xkf1["VE"][mask]
    Yaw = xkf1["Yaw"][mask]
    GZ =  xkf1["GZ"][mask]
    Time = xkf1["TimeUS"][mask] * 1e-6






    Time, VN, VE, Yaw = filtrar_por_tempo(Time, VN, VE, Yaw, t_min=75.8, t_max=94.5)
    Time = Time - Time[0]

    Yaw_rad = np.unwrap(np.deg2rad(Yaw)) if np.max(Yaw) > 2 * np.pi else Yaw

    Yaw_rad = pd.Series(Yaw_rad).rolling(window=60, center=True, min_periods=30).mean().to_numpy()

    # 1. Eliminar repetições para evitar erro
    Time = np.linspace(Time.min(), Time.max(), len(Time))


    plt.subplot(2, 1, 1)
    plt.plot(Time, Yaw_rad, label=" (Yaw rad)", linewidth=2)
    # plt.plot(Time, Yaw_unwrapped, label=" (Yaw unwrapped)", linewidth=2)
    # plt.plot(Time, angular_velocity, label="GZ", linewidth=2)
    plt.ylabel("Velocidade")
    plt.title("Velocidades do Veículo (Frame Local)")
    plt.grid(True)
    plt.legend()
    plt.show()

    cos_yaw = np.cos(Yaw_rad)
    sin_yaw = np.sin(Yaw_rad)
    v_forward = cos_yaw * VN + sin_yaw * VE

    v_forward = pd.Series(v_forward).rolling(window=10, center=True, min_periods=5).mean().to_numpy()

    Time_pwm = rcou["TimeUS"] * 1e-6
    pwm_scaled = {}
    for i in range(1, 5):
        raw = rcou[f"C{i}"]
        scaled = (raw - 1500) * 100 / 400
        pwm_scaled[i] = scaled

    Time_pwm, pwm1, pwm2, pwm3, pwm4 = filtrar_por_tempo(
        Time_pwm,
        pwm_scaled[1], pwm_scaled[2], pwm_scaled[3], pwm_scaled[4],
        t_min=75.8, t_max=94.5
    )
    Time_pwm = Time_pwm - Time_pwm[0]
    Time_pwm = np.linspace(Time_pwm.min(), Time_pwm.max(), len(Time_pwm))

    pwm1 = np.interp(Time, Time_pwm, pwm1)
    pwm2 = np.interp(Time, Time_pwm, pwm2)
    pwm3 = np.interp(Time, Time_pwm, pwm3)
    pwm4 = np.interp(Time, Time_pwm, pwm4)
    Time_pwm = Time.copy()

    pwm_scaled = {
        1: pwm1,
        2: pwm2,
        3: pwm3,
        4: pwm4
    }


    angular_velocity = np.gradient(Yaw_rad, Time)
    angular_velocity_smooth = pd.Series(angular_velocity).rolling(window=60, center=True, min_periods=30).mean().to_numpy()- 0.25

    plt.subplot(2, 1, 1)
    plt.plot(Time, v_forward, label="Velocidade Linear (Forward)", linewidth=2)
    plt.plot(Time, angular_velocity, label="Velocidade Angular (Yaw)", linewidth=2)
    plt.plot(Time, angular_velocity_smooth, label="Yaw Suavizado", linewidth=2)
    plt.ylabel("Velocidade")
    plt.title("Velocidades do Veículo (Frame Local)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(1, 5):
        plt.plot(Time_pwm, pwm_scaled[i], label=f'Motor {i}')
    plt.xlabel("Tempo [s]")
    plt.ylabel("PWM Escalonado [-100, 100]")
    plt.title("PWM dos Motores")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    df_sim = pd.DataFrame({
        "timestamp(ms)": (Time_pwm * 1000).astype(int),
        "RCOU.C1": ((pwm1 * 400 / 100) + 1500).astype(int),
        "RCOU.C2": ((pwm2 * 400 / 100) + 1500).astype(int),
        "RCOU.C3": ((pwm3 * 400 / 100) + 1500).astype(int),
        "RCOU.C4": ((pwm4 * 400 / 100) + 1500).astype(int),
        "GPS[0].Spd": v_forward,
        "IMU[0].GyrZ": angular_velocity_smooth
    })

    xlsx_path = "./planilhas/sequenciaa1.xlsx"
    df_sim.to_excel(xlsx_path, index=False)


