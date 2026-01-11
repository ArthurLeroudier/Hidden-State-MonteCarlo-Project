import numpy as np
import matplotlib.pyplot as plt


def plot_fig3_from_npz(npz_path, title, cmap, show_colorbar=True, em_style="o", linewidth=2):
    data = np.load(npz_path)

    tau_axis = data["tau_axis"]
    sigma0_axis = data["sigma0_axis"]
    Z = data["Z"]

    x_path = data["x_path"]
    y_path = data["y_path"]

    # Grid coords in the same space as the paper:
    # X = log10(tau), Y = log10(sigma0^2)
    X = np.log10(tau_axis)[None, :].repeat(len(sigma0_axis), axis=0)
    Y = np.log10((sigma0_axis ** 2))[:, None].repeat(len(tau_axis), axis=1)

    plt.figure(figsize=(7, 5))
    pcm = plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)

    if show_colorbar:
        plt.colorbar(pcm, label="log-likelihood (SMC estimate)")

    plt.plot(x_path, y_path, em_style, linewidth=linewidth)

    plt.xlabel(r"$\log_{10}(\tau)$")
    plt.ylabel(r"$\log_{10}(\sigma_0^2)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_fig3_from_npz(
        npz_path="data/fig3_smc_data.npz",
        title="Log-likelihood of WTA under SMC",
        cmap="viridis",
    )
