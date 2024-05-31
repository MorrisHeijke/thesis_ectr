# Inflation Forecasting in the Era of Big Data: Machine Learning and Bayesian Econometrics

In this thesis project the FRED-MD monthly dataset containing macroeconomic time series are used to produce inflation rate forecasts of 12 months ahead. Several forecasting methods are explored which include machine learning, Bayesian and Frequentist approaches. The ML methods included are Random Forest and XGBoost. The Bayesian Econometrics methods include a Bayesian LASSO, Bayesian Ridge, Bayesian VAR and a Time-Varying Parameter VAR. For Frequentist approaches, auto-regressive methods, LASSO, Ridge and Elastic Net are implemented.

## Explanation Files

The project contains the following files with a short description of what can be found in them.

- eda.ipynb contains the Exploratory Data Analysis;
- main.ipynb runs the main loop for the final models to obtain their results;
- BVAR.ipynb contains the results and sampling of the BVAR model;
- tvpvar.ipynb contains the results and sampling of the TVP-VAR model;
- functions.py contains all helper functions and classes of models to be run in main.ipynb;
- 2023-10.csv contains the FRED-MD monthly dataset;
- requirements.txt contains the dependencies to be installed to run the code.
## Installation

Install my-project with pip

```bash
  pip install -r requirements.txt
```
    
## Author

- [@MorrisHeijke](https://github.com/MorrisHeijke)

