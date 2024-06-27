# Predicting critical transitions with surrogate data-based machine learning

This repository contains code to accompany the publication


## Requirements

Requires Python 3.7 or later to compute early warning signals (EWS). To install python package dependencies, use the command

```bash
conda create surr_ews python=3.9
conda activate surr_ews
pip install -r requirements_surr_ews.txt
```
within a new virtual environment.

Python 3.6 and MATLAB 2018b are required to generate surrogate data. To install python package dependencies, use the command

```bash
conda create -n surr_mat python=3.8
conda activate surr_mat
pip install -r requirements_surr_mat.txt
```

The Versions of Python and MATLAB have a one-to-one correspondence, otherwise the connection cannot be established, see below for the correspondence: [Versions of Python Compatible with MATLAB Products by Release - MATLAB & Simulink (mathworks.cn)](https://ww2.mathworks.cn/support/requirements/python-compatibility.html)


## Directories

**./anoxia:** Code and data for the geochemical data  

**./anoxia/01_data_preprocessing/01_organise\_\_transition\_\_ews:** Code to pre-process and compute generic early warning signals

**./anoxia/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset:** Code to generate training data

**./anoxia/02_train_predicate_model:** Code to compute early warning signal by surrogate data-based machine learning



**./chick_heart:** Code and data for the chick heart data  

**./chick_heart/01_data_preprocessing/01_organise\_\_transition\_\_ews:** Code to pre-process and compute generic early warning signals

**./chick_heart/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset:** Code to generate training data

**./chick_heart/02_train_predicate_model:** Code to compute early warning signal by surrogate data-based machine learning



**./paleoclimate:** Code and data for the paleoclimate data  

**./paleoclimate/01_data_preprocessing/01_organise\_\_transition\_\_interpolate\_\_ews:** Code to pre-process and compute generic early warning signals

**./paleoclimate/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset:** Code to generate training data

**./paleoclimate/02_train_predicate_model:** Code to compute early warning signal by surrogate data-based machine learning



**./tree_felling:** Code and data for the tree-ring data  

**./tree_felling/01_data_preprocessing/01_organise\_\_transition\_\_ews:** Code to pre-process and compute generic early warning signals

**./tree_felling/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset:** Code to generate training data

**./tree_felling/02_train_predicate_model:** Code to compute early warning signal by surrogate data-based machine learning



**./00_make_figures:** Code and data to generate figures used in manuscript.


## Workflow

The results in the paper are obtained from the following workflow:

1. **Pre-process data**. Codes to process the data are available in the directory `./01_data_preprocessing/01_organise__transition__ews/.` Residuals are obtained and variance and lag-1 autocorrelation are computed.

2. **Generate the training data**. We used five different surrogate methods to generate surrogate data with different sample sizes as training data. Note: we are running from the command line interface of the Git software installation. Run

   ```bash
   conda activate surr_mat
   bash ./01_data_preprocessing/02_extracte_and_generate_surrogate_dataset/code/run_all_script.sh | tee oe_run_all_script.log
   ```

   where generate the training data save: `./02_extracte_and_generate_surrogate_dataset/data`. Due to the large size of the generated surrogate datasets, the data was deleted before being pushed to `GitHub`. The final compressed output is 27.5GB for the whole project. This dataset is archived on `Zenodo` at [Predicting critical transitions with surrogate data-based machine learning: Project (zenodo.org)](https://zenodo.org/records/12562316).

3. **Train the surrogate data-based machine learning  (SDML) classifiers, Generate SDML predictions, and Compute ROC statistics.** Results reported in the paper take the average prediction from these 10 networks. Note: we are running from the command line interface of the Git software installation. Run 

   ```bash
   conda activate surr_ews
   bash ./02_train_predicate_model/xxx_ml/run_all_script_ml.sh | tee oe_run_all_script_ml.log
   bash ./02_train_predicate_model/xxx_dl/run_all_script_dl.sh | tee oe_run_all_script_dl.log
   ```

   The training models and result data, which are large in volume, were deleted before being pushed to `GitHub`. This dataset is archived on `Zenodo` at [https://zenodo.org/records/12562316](https://zenodo.org/records/12562316).

## Data sources

The empirical data used in this study are available from the following sources:
1. **chick heart** data have been deposited in [GitHub](https://github.com/ThomasMBury/dl_discrete_bifurcation/tree/main/data/df chick.csv). Data were preprocessed according to the study [Bury, Thomas M., et al. "Predicting discrete-time bifurcations with deep learning." Nature Communications 14.1 (2023): 6331.]([Predicting discrete-time bifurcations with deep learning | Nature Communications](https://www.nature.com/articles/s41467-023-42020-z))
2. **Sedimentary archive** data from the Mediterranean Sea are available at the [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.923197) data repository. Data were preprocessed according to the study [Hennekam, Rick, et al. "Early‐warning signals for marine anoxic events." Geophysical Research Letters 47.20 (2020): e2020GL089183.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL089183)
3. **Paleoclimate transition** data are available from the [World Data Center for Paleoclimatology](http://www.ncdc.noaa.gov/paleo/data.html), National Geophysical Data Center, Boulder, Colorado. Data were preprocessed according to the study [Dakos, Vasilis, et al. "Slowing down as an early warning signal for abrupt climate change." Proceedings of the National Academy of Sciences 105.38 (2008): 14308-14312.](https://www.pnas.org/content/105/38/14308.short)
4. **tree-ring** data from the Southwestern United States can be accessed at the [Digital Archaeological Record](https://doi.org/10.6067/XCV82J6D7B). Data were collected by Timothy A. Kohler and Marten Scheffer and were published in [Kohler, Timothy A., et al. "Compiled Tree-ring Dates from the Southwestern United States (Restricted)." (2016)](https://core.tdar.org/dataset/399314/compiled-tree-ring-dates-from-the-southwestern-united-states-restricted) and [Marten Scheffer, et al. "Loss of resilience preceded transformations of pre-Hispanic Pueblo societies" Proceedings of the National Academy of Sciences 118.18 (2021): e2024397118.](https://www.pnas.org/doi/abs/10.1073/pnas.2024397118)

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg



