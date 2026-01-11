# Hidden-State-MonteCarlo-Project
Project conducted in the context of the lecture Hidden Markov models and sequential Monte-Marco method, directed by Nicolas Chopin. This project recreate Figure 3 of the article [A state-space perspective on modelling and inference for online skill rating](https://doi.org/10.1093/jrsssc/qlae035). 

To run the code, you need to install the libraries in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running SMC plot

To run the SMC plot, simply place yourself at the root of the folder and run the following command:

```bash
python -m experiments.smc
```

 This will create a `file fig3_smc_data.npz` in the folder `data/`. 
 If you want to change the parameters, or the file, run:

 ```bash
python -m experiments.smc --help
```

To plot the heat map run:

 ```bash
python -m experiments.plot_smc
```