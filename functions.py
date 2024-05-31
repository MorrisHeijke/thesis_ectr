import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.stattools      import adfuller, lagmat, grangercausalitytests
from statsmodels.tools              import add_constant
from statsmodels.graphics.tsaplots  import plot_acf, plot_pacf
from statsmodels.tsa.statespace     import mlemodel
from statsmodels.tsa.arima.model    import ARIMA
from statsmodels.tsa.api            import VAR
from sklearn.feature_selection      import mutual_info_regression
from sklearn.metrics                import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection        import TimeSeriesSplit
from sklearn.linear_model           import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble               import RandomForestRegressor
from scipy.stats                    import invgamma, invwishart



#### Auxiliar functions

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') 
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

def adf_test_multiple(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    result = adfuller(series.dropna(),autolag='AIC')
    
    if result[1] > 0.05:
        return title


def plot_tsresiduals(Y, y, acf_lags=np.r_[1:26], name=""):
    """
    Plot timeseries residuals for ground truth Y and estimate y.
    """
    fig = plt.figure()
    gs = plt.GridSpec(3, 2, figure=fig)
    ts_ax = fig.add_subplot(gs[0,:])
    axs = np.array([ts_ax] + [fig.add_subplot(gs[i,j]) for j in (0,1) for i in (1,2)])
    ax, rax, hax, acfax, pacfax = axs
    dy = Y - y
    ax.plot(Y, color='k')
    ax.plot(y)
    ax.set(title=f'{name}')
    lim = 1.1 * max(-dy.min(), dy.max())
    lim = -lim, lim
    rax.plot(dy)
    rax.set(ylim=lim, title='Residuals')
    plt.setp(rax.get_xticklabels(), rotation=30, horizontalalignment='right')
    sns.histplot(dy, bins=np.linspace(lim[0], lim[1], 22),
                kde=True, ax=hax)
    hax.set(title='Residual Distribution')
    plot_acf(dy, lags=acf_lags, ax=acfax)
    plot_pacf(dy, lags=acf_lags, ax=pacfax)
    for a in axs.ravel():
        a.grid()
    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.0)
    return fig, axs

def grangers_causation_matrix(data, variables, maxlag, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df 

def create_lagged_vars(df, window):
    """
    Creates lagged variables for a given dataframe and until lags of t-window.
    """
    df_lagged = df.copy()
    for i in range(1, window+1):
        shifted = df.shift(i)
        shifted.columns = [x + " t-" + str(i) for x in df.columns]
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    return df_lagged

def make_lagged_vars(X, p):
    """
    Creates p lags of variables in X for BVAR & TVP-VAR.
    X is [T x M], Xlag is [T x (Mp)].
    """
    T, N = X.shape
    Xlag = np.zeros((T, N * p))

    for t in range(1, p + 1):
        Xlag[p:T, (N * (t - 1)):(N * t)] = X[(p - t):(T - t), :N]
        
    return Xlag

def get_non_stat_vars(df):
    non_stationary_vars = []
    for _, series in df.items():
        non_stationary_vars.append(adf_test_multiple(series, title=series.name))

    non_stationary_vars = [x for x in non_stationary_vars if x is not None]
    return non_stationary_vars

def make_mi_scores(X, Y):
    mi_scores = mutual_info_regression(X, Y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_correlations(df, number_of_vars, title=''):
    correlation = df.corr()['inflation'].sort_values(ascending=False)
    pearson_scores = correlation[1:number_of_vars+1].sort_values(ascending=True)

    Y = df.loc[:, 'inflation']
    X = df.drop(columns=['inflation'])
    mi_scores = make_mi_scores(X, Y)[:number_of_vars].sort_values(ascending=True)
 
    mi_width = np.arange(len(mi_scores))
    mi_ticks = list(mi_scores.index)    

    p_width = np.arange(len(pearson_scores))
    p_ticks = list(pearson_scores.index)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.barh(mi_width, mi_scores, color='blue')
    ax1.set_yticks(mi_width)
    ax1.set_yticklabels(mi_ticks)
    ax1.set_title("Mutual Information Scores")
    ax2.barh(p_width, pearson_scores, color='green')
    ax2.set_yticks(p_width)
    ax2.set_yticklabels(p_ticks)
    ax2.set_title("Pearson Scores")
    fig.suptitle(f'Feature Importance {title}')
    plt.tight_layout()

def train_test_split(df, split_date):
    # Start date for training since many missing entries before 1960 and some variables are not available before 1960
    start_date = "1960-01-01"

    train = df.loc[start_date:split_date,:]
    test = df.loc[split_date:,:]  

    X = train.drop(columns=["inflation"])
    y = train.loc[:,"inflation"]

    X_test = test.drop(columns=["inflation"])
    y_test  = test.loc[:,"inflation"]

    return X, y, X_test, y_test

def gibbs_sampler_lambda(X, y, B, lambda_):
    # Initialization
    beta = [np.random.uniform(size = X.shape[1])]
    sigma_sq = [np.random.uniform()]
    tau_sq = [np.random.uniform(size = X.shape[1])]
    
    for i in range(B):
        # Full conditional for beta
        D_tau = np.diag(tau_sq[i])
        A = X.transpose().dot(X) + np.linalg.inv(D_tau)
        multi_norm_mean = np.linalg.inv(A).dot(X.transpose()).dot(y)
        multi_norm_cov = sigma_sq[i] * np.linalg.inv(A)
        beta.append(np.random.multivariate_normal(multi_norm_mean, multi_norm_cov))

        # Full conditional for sigma_sq
        shape = (X.shape[0] - 1 + X.shape[1]) / 2
        scale = ((y - X.dot(beta[i+1])).dot((y - X.dot(beta[i+1]))) + beta[i+1].transpose().dot(np.linalg.inv(D_tau)).dot(beta[i+1])) / 2
        sigma_sq.append(invgamma.rvs(a = shape, scale = scale))

        # Full conditional for tau_1,...,tau_p
        mean = np.sqrt(lambda_**2 * sigma_sq[i+1] / beta[i+1]**2)
        scale = np.repeat(lambda_**2, X.shape[1])
        tau_sq.append(1/np.random.wald(mean, scale))
    return tau_sq[int(B/2):]

def gibbs_sampler_beta(X, y, B, lambda_):
    # Initialization
    beta = [np.random.uniform(size = X.shape[1])]
    sigma_sq = [np.random.uniform()]
    tau_sq = [np.random.uniform(size = X.shape[1])]
    
    for i in range(B):
        # Full conditional for beta
        D_tau = np.diag(tau_sq[i])
        A = X.transpose().dot(X) + np.linalg.inv(D_tau)
        multi_norm_mean = np.linalg.inv(A).dot(X.transpose()).dot(y)
        multi_norm_cov = sigma_sq[i] * np.linalg.inv(A)
        beta.append(np.random.multivariate_normal(multi_norm_mean, multi_norm_cov))

        # Full conditional for sigma_sq
        shape = (X.shape[0] - 1 + X.shape[1]) / 2
        scale = ((y - X.dot(beta[i+1])).dot((y - X.dot(beta[i+1]))) + beta[i+1].transpose().dot(np.linalg.inv(D_tau)).dot(beta[i+1])) / 2
        sigma_sq.append(invgamma.rvs(a = shape, scale = scale))

        # Full conditional for tau_1,...,tau_p
        mean = np.sqrt(lambda_**2 * sigma_sq[i+1] / beta[i+1]**2)
        scale = np.repeat(lambda_**2, X.shape[1])
        tau_sq.append(1/np.random.wald(mean, scale))
    return beta[int(B/2):]

def ts_cross_validation(models, X, y, k, test_size=12):
    # set up results storage
    result_dict = dict()
    for model in models:
        result_dict[model] = []

    # set up cross validation
    tscv = TimeSeriesSplit(n_splits=k, test_size=test_size)
    
    for k, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]  

        # fit, predict and score models
        for m in models.keys():
            model = models[m]
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            results_model = result_dict[m]
            results_model.append([mean_squared_error(y_train, y_pred_train), mean_squared_error(y_val, y_pred_val), 
                                mean_absolute_error(y_train, y_pred_train), mean_absolute_error(y_val, y_pred_val), 
                                r2_score(y_train, y_pred_train), r2_score(y_val, y_pred_val)])

    # average model scores
    for model in result_dict.keys():
        mse_train = np.mean([x[0] for x in result_dict[model]])
        mse_val = np.mean([x[1] for x in result_dict[model]])
        mae_train = np.mean([x[2] for x in result_dict[model]])
        mae_val = np.mean([x[3] for x in result_dict[model]])
        rsqd_train = np.mean([x[4] for x in result_dict[model]])
        rsqd_val = np.mean([x[5] for x in result_dict[model]])
        result_dict[model] = [mse_train, mse_val, mae_train, mae_val, rsqd_train, rsqd_val]
        
    return result_dict

def out_of_sample_forecast(models, df, split_date):
    # set up results storage
    result_dict = dict()
    y_pred_dict = dict()
    for model in models:
        result_dict[model] = []
        y_pred_dict[model] = []

    X, y, X_test, y_test = train_test_split(df, split_date)

    for m in models.keys():
        model = models[m]
        model.fit(X, y)
        y_pred = model.predict(X_test)
        y_pred_dict[m] = y_pred
        result_dict[m] = [mean_squared_error(y_test, y_pred), 
                        mean_absolute_error(y_test, y_pred), 
                        r2_score(y_test, y_pred)]

    return result_dict, y_pred_dict

#### Predictors
    
class mean_predictor():
    def __init__(self):
        self.mean = 0

    def predict(self, X):
        return pd.Series(np.full(X.shape[0], self.mean))
    
    def fit(self, X, y):
        self.mean = y.mean()

class naive_predictor():
    def __init__(self):
        self.naive = 0

    def predict(self, X):
        return pd.Series(np.full(X.shape[0], self.naive[-X.shape[0]:]))

    def fit(self, X, y):
        self.naive = y.tail(X.shape[0])

class mlr_predictor():
    def __init__(self, linear_significant=None):
        self.model = LinearRegression()

        if linear_significant == None:
            self.significant = None
        else:
            self.significant = linear_significant
            try:
                self.significant.remove('inflation')
                self.significant.remove('log_inflation')
            except:
                pass
        
    def predict(self, X):
        if not self.significant:
            return self.model.predict(X)

        X = X.loc[:, self.significant]
        return self.model.predict(X)

    def fit(self, X, y):
        if not self.significant:
            self.model.fit(X,y)
            return

        X = X.loc[:, self.significant]
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass
        self.model.fit(X,y)

class lasso_predictor():
    def __init__(self):
        self.model = Lasso(alpha=0.9)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        self.model.fit(X,y)

class bayes_lasso_predictor():
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.betas = []

    def predict(self, X):
        return X.dot(self.betas)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        gibbs_beta_samples = gibbs_sampler_beta(X, y, 500, self.lambda_)
        beta_bayes_est = np.median(gibbs_beta_samples, axis=0)
        self.betas = beta_bayes_est

class ridge_predictor():
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        self.model.fit(X,y)

class bayes_ridge_predictor():
    def __init__(self):
        self.model = BayesianRidge()

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass
        
        init = [1 / np.var(y), 1.0]
        self.model.set_params(alpha_init=init[0], lambda_init=init[1])
        self.model.fit(X,y)

class elasticnet_predictor():
    def __init__(self):
        self.model = ElasticNet(alpha=0.9)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        self.model.fit(X,y)

class xgboost_predictor():
    def __init__(self, best_params):
        self.model = xgb.XGBRegressor(learning_rate = best_params["learning_rate"],
                                    n_estimators    = best_params["n_estimators"],
                                    max_depth       = best_params["max_depth"],
                                    objective       = 'reg:squarederror')

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        self.model.fit(X,y)

class rf_predictor():
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators = 25,
                                           max_depth    = None,
                                           criterion    = 'squared_error')

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        try:
            X = X.drop(columns=['inflation','log_inflation'], axis=1)
        except:
            pass

        self.model.fit(X,y)

class arima_predictor():
    def __init__(self, y, p, d, q):
        self.model = ARIMA(y, order=(p,d,q))
        self.model_fit = 0

    def predict(self, X):
        return self.model_fit.forecast(X.shape[0]) 

    def fit(self, X, y):
        self.model_fit = self.model.fit()

class VAR_predictor():
    def __init__(self, vars):
        self.vars = vars

    def predict(self, X):
        return self.model_fit.forecast(self.model_fit.endog, steps=X.shape[0])[:,0]

    def fit(self, X, y):
        y_eqs = X[self.vars]
        y_eqs['inflation'] = y
        y_eqs = y_eqs[["inflation"]+self.vars]

        self.model = VAR(y_eqs)
        self.model_fit = self.model.fit(maxlags=3)

class TVPVAR(mlemodel.MLEModel):
    def __init__(self, y):
        # Create a matrix with [y_t' : y_{t-1}'] for t = 2, ..., T
        p = 1
        augmented = lagmat(y, p, trim='both', original='in', use_pandas=True)
        # Separate into y_t and z_t = [1 : y_{t-1}']
        T, M = y.shape
        y_t = augmented.iloc[:, :M]
        z_t = add_constant(augmented.iloc[:, M:])

        # Length of the state vector is M * (M + 1)
        k_states = M * (M + 1)
        super().__init__(y_t, exog=z_t, k_states=k_states)

        # Construct the design matrix Z_t
        # -> self.k_endog = M is the dimension of the observed vector
        # -> self.k_states = M * (M + 1) is the dimension of the observed vector
        # -> self.nobs = T is the number of observations in y_t
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = z_t.T

        # Construct the transition matrix T = I
        self['transition'] = np.eye(k_states)

        # Construct the selection matrix R = I
        self['selection'] = np.eye(k_states)

        # Initialize the state vector as alpha_1 ~ N(0, 5I)
        self.ssm.initialize('known', stationary_cov = 5 * np.eye(self.k_states))

    # Call to update observation cov H and state cov diag(\sigma^2)
    def update_variances(self, obs_cov, state_cov_diag):
        self['obs_cov'] = obs_cov
        self['state_cov'] = np.diag(state_cov_diag)

    @property
    def state_names(self):
        state_names = np.empty((self.k_endog, self.k_endog + 1), dtype=object)
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names[i] = (
                ['intercept.%s' % endog_name] +
                ['L1.%s->%s' % (other_name, endog_name) for other_name in self.endog_names])
        return state_names.ravel().tolist()
