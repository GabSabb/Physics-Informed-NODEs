# Physics-Informed-NODEs
This repository constitutes complementary material for the paper "Physics-Informed  Neural Ordinary Differential Equations for Multi-Zone Residential Thermal Modeling".  

## Requirements
The experiments were done using Python 3.9.18. In addition, the following Python packages are needed to run the Jupyter notebooks:

- diffrax: 0.4.1
- equinox: 0.11.0
- jax: 0.4.30
- jaxlib: 0.4.30
- matplotlib: 3.7.2
- optax: 0.1.7
- numpy: 1.26.0
- scikit-learn: 1.3.1
- pandas: 2.0.3

## Repository structure
* EHBE: Experiment on test residence (EHBE) data, the folder contains:
  - JAX_NODE_EHBE.ipynb: Jupyter notebook that imports and treats the data, generates the model, trains it and visualises the results.
  - house_data.csv: Includes the measured data of 9 thermostats at Hydro-Québec's Experimental House for Building Efficiency (EHBE) situated in Shawinigan, Quebec, Canada. It includes the room temperature measurements and the heating outputs of each electric baseboard associated with each thermostat. It also includes the global horizontal irradiation (GHI) and exterior drybulb temperature. Temperatures are measured in $°C$, heating energy in $Wh$ and GHI in $Wm^{-2}$. The original dataset can be found under the same name on [Zenodo](https://doi.org/10.5281/zenodo.10156745) or [GitHub](https://github.com/HarryVallianos/Automated-MultiZone-Model-Generation).
  - best_model_Masked_Conn_False.eqx: Equinox model of a NODE trained on the EHBE data. The NODE is using a fully connected neural network.
  - best_model_Masked_Conn_True.eqx: Equinox model of a NODE trained on the EHBE data. The NODE is using a masked neural network.
* Residence: Experiment on an occupied residence's data, the folder contains:
  - JAX_Residence.ipynb: Jupyter notebook that imports and treats the data, generates and imports the model, and visualises the results.
  - 46.45_-72.66_May17-April18.csv: Solar measurement data at (46.45° N; -72.66° W) from May 2017 to April 2018.
  - Weather_Shawinigan_0517_0418.csv: Weather data from the Shawinigan weather station from May 2017 to April 2018.
  - THER_00079 (2017-05-01_2018-04-30).csv: Thermostat measurement data from an occupied residence from February 1 to February 8 2018. Measurements include temperatures and setpoints in $°C$ and hourly heating energy in $Wh$ for each thermal zone in the residence.
  - best_model_79_Hour_Shift.eqx: Equinox model of a NODE trained on the occupied residence data.
* requirements.txt: A text file listing the different Python libraries required to run the Jupyter notebook.

## Other publications using the same test residence data as this repository

* Gabriel Sabbagh, Massimo Cimmino, Benoit Delcroix (2024). Neural Ordinary Differential Equations for Simulations of Residences Heated with Electric Baseboards, 13th Conference of IBPSA-Canada, Edmonton.
* Charalampos Vallianos, Matin Abtahi, Andreas Athienitis, Benoit Delcroix & Luis Rueda (2023) Online model-based predictive control with smart thermostats: application to an experimental house in Québec, Journal of Building Performance Simulation, https://doi.org/10.1080/19401493.2023.2243602
* Charalampos Vallianos, Andreas Athienitis, Benoit Delcroix, Automatic generation of multi-zone RC models using smart thermostat data from homes, Energy and Buildings, Volume 277, 2022, 112571, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2022.112571
* Vallianos, C. et al. (2023). Automated RC Model Generation for MPC Applications to Energy Flexibility Studies in Quebec Houses. In: Wang, L.L., et al. Proceedings of the 5th International Conference on Building Energy and Environment. COBEE 2022. Environmental Science and Engineering. Springer, Singapore. https://doi.org/10.1007/978-981-19-9822-5_73
