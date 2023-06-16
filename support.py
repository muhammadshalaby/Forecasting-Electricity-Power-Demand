import math
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
plt.style.use('fivethirtyeight')
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)
from pmdarima.arima.stationarity import KPSSTest
from pmdarima.arima.utils import ndiffs
from scipy.stats.mstats import normaltest
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose


def removeOutLiers(data, remove=True):
    '''Input: DataFrame
       Remove outliers
       Output: DataFrame'''
    changed_rows = 0
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # Finding first quartile and third quartile
            q1, q3 = np.percentile(data[col],[25,75])
            # Find the IQR which is the difference between third and first quartile
            iqr = q3 - q1
            # Find lower and upper bound
            lower_bound = q1 - (1.5 * iqr) 
            upper_bound = q3 + (1.5 * iqr)
            # Create dataframe contain heavey players who played more than the upper_bound
            idx = data.query(f"{col} > {upper_bound} | {col} < {lower_bound}").index
            print(f'{col} Feature Has {len(idx)} Outliers')
            changed_rows += len(idx)
            if remove:
                # Remove index less than or equal the upper_bound
                data = data[~data.index.isin(idx)]
            else:
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    print('=' * 80)
    if remove:
        print(f'NO of Deleted Rows Is: {changed_rows}, New Data Shape Is: {data.shape}')
    else:
        print(f'NO of Changed Rows Is: {changed_rows}, New Data Shape Is: {data.shape}')    
    return data

def seasonal_plot(X, y, period, freq, ax=None):
    # days within a week
    X["day"] = X.index.dayofweek  # the x-axis (freq)
    X["week"] = X.index.week  # the seasonal period (period)
    # days within a year
    X["dayofyear"] = X.index.dayofyear
    X["year"] = X.index.year
    
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(x=freq, y=y, hue=period, data=X, ci=False, ax=ax, palette=palette, legend=False)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(name, xy=(1, y_), xytext=(6, 0), color=line.get_color(), xycoords=ax.get_yaxis_transform(), 
                    textcoords="offset points", size=14, va="center")
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(ts, fs=fs, detrend=detrend, window="boxcar", scaling='spectrum')
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(["Annual (1)", "Semiannual (2)", "Quarterly (4)", "Bimonthly (6)", "Monthly (12)", 
                        "Biweekly (26)", "Weekly (52)", "Semiweekly (104)"], rotation=30)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(alpha=0.75, s=3)
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_, y=y_, scatter_kws=scatter_kws, line_kws=line_kws, lowess=True, ax=ax, **kwargs)
    at = AnchoredText(f"{corr:.2f}", prop=dict(size="large"), frameon=True, loc="upper left")
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

def plot_forecasts(y_train, y_test, forecasts, title):
    x = np.arange(y_train.shape[0] + forecasts.shape[0])
    fig, axes = plt.subplots(1, 2, sharex=False, figsize=(20, 8))
    # Plot the forecasts
    axes[0].plot(x[:y_train.shape[0]], y_train, c='b')
    axes[0].plot(x[y_train.shape[0]:], forecasts, c='g')
    axes[0].set_xlabel(f'Sunspots (RMSE={np.sqrt(mse(y_test, forecasts)):.3f})')
    axes[0].set_title(title)
    # Plot the residuals
    resid = y_test - forecasts
    _, p = normaltest(resid)
    axes[1].hist(resid, bins=15)
    axes[1].axvline(0, linestyle='--', c='r')
    axes[1].set_title(f'Residuals (p={np.around(p, 3)[0]})')
    plt.tight_layout()
    plt.show()

def diffs_number(data, alpha=0.05):
    # Test whether we should difference at the alpha=0.05
    # significance level
    kpss_test = KPSSTest(alpha=alpha)
    p_val, should_diff = kpss_test.should_diff(data)
    print(f'Pvalue Is: {p_val}\nShould we difference at the alpha={alpha}? {should_diff}')
    if should_diff:
        n_kpss = ndiffs(data, alpha=alpha, test='kpss')
        print(f'Number of differences using kpss test Is: {n_kpss}')

    adf_diffs = ndiffs(data, alpha=alpha, test='adf')
    print(f'Number of differences using adf test Is: {adf_diffs}')


def ts_1step(y_label, model, y_test, exo=False):
    forecasts = []
    if exo:
        exog = y_test.drop([y_label], axis=1)
        for new_ob in range(y_test.shape[0]):
            x = exog.iloc[new_ob,:].to_frame().T
            forecasts.append(model.predict(n_periods=1, X=x).tolist()[0])
            # Updates the existing model with a small number of MLE steps
            model.update(y_test[y_label].iloc[new_ob], X=x)
    else:
        for new_ob in range(y_test.shape[0]):
            forecasts.append(model.predict(n_periods=1).tolist()[0])
            # Updates the existing model with a small number of MLE steps
            model.update(y_test[y_label].iloc[new_ob])
    
    return forecasts


def decompose(ts):
    ss_decomposition = seasonal_decompose(ts)#, model='additive')  # multiplicative, additive
    estimated_obs = ss_decomposition.observed
    estimated_trend = ss_decomposition.trend
    estimated_seasonal = ss_decomposition.seasonal
    estimated_residual = ss_decomposition.resid

    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(20, 8))
    axes[0].plot(estimated_obs, label='observation')
    axes[0].legend(loc='upper left')
    axes[1].plot(estimated_trend, label='Trend')
    axes[1].legend(loc='upper left')
    axes[2].plot(estimated_seasonal, label='Seasonality')
    axes[2].legend(loc='upper left')
    axes[3].plot(estimated_residual, label='Residuals')
    axes[3].legend(loc='upper left')
    plt.tight_layout()
    plt.show()


