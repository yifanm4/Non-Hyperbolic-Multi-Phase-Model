import numpy as np
import matplotlib.pyplot as plt

from simple import solve_full_two_fluid_model


def run_case(label, C_E, C_max, n_cells, t_final):
    x, ag, ug, ul, P = solve_full_two_fluid_model(
        t_final=t_final,
        n_cells=n_cells,
        C_vm=0.0,
        C_ip=0.0,
        C_E=C_E,
        C_max=C_max,
        use_muscl=True,
        use_quick=False,
    )
    ug_c = 0.5 * (ug[1:] + ug[:-1])
    ul_c = 0.5 * (ul[1:] + ul[:-1])
    return {"label": label, "x": x, "ag": ag, "ug_c": ug_c, "ul_c": ul_c, "P": P}


def main():
    # Lightweight sweep to compare entropy-viscosity settings on MUSCL
    n_cells = 200
    t_final = 0.2
    cases = [
        ("EV 0/0", 0.0, 0.0),
        ("EV 0.5/0.5", 0.5, 0.5),
        ("EV 1/0.5", 1.0, 0.5),
        ("EV 1/1", 1.0, 1.0),
    ]

    results = [run_case(lbl, cE, cMax, n_cells, t_final) for lbl, cE, cMax in cases]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"MUSCL: Entropy Viscosity Sensitivity (t={t_final}s, n={n_cells})", fontsize=14)

    # Void fraction
    for res in results:
        axs[0, 0].plot(res["x"], res["ag"], label=res["label"])
    axs[0, 0].set_title("Void Fraction")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Pressure
    for res in results:
        axs[0, 1].plot(res["x"], res["P"], label=res["label"])
    axs[0, 1].set_title("Pressure (Pa)")
    axs[0, 1].grid(True)

    # Liquid velocity
    for res in results:
        axs[1, 0].plot(res["x"], res["ul_c"], label=res["label"])
    axs[1, 0].set_title("Liquid Velocity $u_l$")
    axs[1, 0].set_xlabel("Position (m)")
    axs[1, 0].grid(True)

    # Gas velocity
    for res in results:
        axs[1, 1].plot(res["x"], res["ug_c"], label=res["label"])
    axs[1, 1].set_title("Gas Velocity $u_g$")
    axs[1, 1].set_xlabel("Position (m)")
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
