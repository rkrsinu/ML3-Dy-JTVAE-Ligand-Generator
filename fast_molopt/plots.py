import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.labelsize": 18,
        "legend.fontsize": 22,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.frameon": False,
    }
)


def plot_latent_trajectory(results):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    # Unpack the 'similarity' variable from each dictionary in the results list
    similarities = [
        result["current_similarity"] for result in results["steps_along_gradient"]
    ]
    latent_vector_idx = [
        result["latent_vector_idx"] for result in results["steps_along_gradient"]
    ]

    predictions = results["predictions"]

    ax[0].set(ylabel="Tanimoto score", xlabel="While step")
    ax[0].plot(similarities, "go-", linewidth=2)

    ax[1].set(ylabel="Sampled z index", xlabel="While step")
    ax[1].plot(latent_vector_idx, "bo-", linewidth=2)

    ax[2].set(ylabel="Predicted homo-lumo (eV)", xlabel="Iteration")
    ax[2].plot(
        [x * 27.2114 for x in predictions["homo-lumo-gap"]],
        "ro-",
        linewidth=2,
    )
    ax[3].set(ylabel="Predicted Ir -CM5 charge", xlabel="Iteration")
    ax[3].plot(
        predictions["Ir-cm5"],
        color="darkorange",
        linestyle="-",
        marker="o",
        linewidth=2,
    )
    plt.tight_layout()
    # [ax[i].set_xticks([x for x in latent_vector_idx]) for i in range(4)]  # set_xticks([])
    # fig.suptitle('Sampling 1000 latent vectors in direction of gradient', fontsize=16)
    # plt.tight_layout()
    plt.show()
    fig.savefig("./image.png", format="png", dpi=300, bbox_inches="tight")
