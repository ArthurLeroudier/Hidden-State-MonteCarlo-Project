# Temporary file for tests
from models.particle_filter import run_wta_filter, PFParams

params = PFParams(
    sigma0=0.5,   # prior std
    tau=0.02,     # diffusion
    beta=1.0      # pente logistic
)

pf, loglik, means_hist, id2name = run_wta_filter(
    params=params,
    n_particles=2000,
    seed=0,
    store_means=False
)

print("Log-likelihood (estimate):", loglik)
