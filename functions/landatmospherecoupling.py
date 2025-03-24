"""Create land-atmosphere coupling class"""

import sys
import os
import numpy as np
import pandas as pd

from scipy import optimize
from scipy.stats import linregress


### CST ###


### FUNC ###

# See https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python

def piecewise_linear_dt(x, x0, y0, k1):
    """Make piecewise linear model for dry to transitional regimes transition"""
    out = np.piecewise(x, [x < x0, x >= x0], [lambda x:y0, lambda x:k1*x + y0-k1*x0])
    return out

def piecewise_linear_tw(x, x0, y0, k1):
    """Make piecewise linear model for transitional to wet regimes transition"""
    out = np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:y0])
    return out

def piecewise_linear_dtw(x, x0, x1, y0, k1):
    """Make piecewise linear model for dry to transitional to wet regimes transitions"""
    condlist = [x < x0, np.logical_and(x >= x0, x < x1), x >= x1]
    funclist = [lambda x: y0, lambda x: k1*(x-x0) + y0, lambda x:k1*(x1-x0) + y0]
    out = np.piecewise(x, condlist, funclist)
    return out


### CLASS

class LACR:
    """Land-Atmosphere Coupling Regime"""
    def __init__(self, sm, ef):
        self.x = sm
        self.y = ef

    def fit_linear_model(self):
        """Fit linear model"""
        lr = linregress(self.x, self.y)
        out = (lr.intercept, lr.slope)
        return out

    def fit_piecewise_linear_dt(self):
        """Fit dry-to-transitional model"""
        p, e = optimize.curve_fit(piecewise_linear_dt, self.x, self.y, p0=[self.x.mean(), self.y.mean(), 0.01])
        return p

    def fit_piecewise_linear_tw(self):
        """Fit transitional-to-wet model"""
        p, e = optimize.curve_fit(piecewise_linear_tw, self.x, self.y, p0=[self.x.mean(), self.y.mean(), 0.01])
        return p

    def fit_piecewise_linear_dtw(self):
        """Fit dry-to-transitional-to-wet model"""
        wilt = self.fit_piecewise_linear_dt()[0]
        crit = self.fit_piecewise_linear_tw()[0]
        p, e = optimize.curve_fit(piecewise_linear_dtw, self.x, self.y, p0=[wilt, crit, 0.1, 0.01]) # self.y.mean()/3
        return p

    def predicted_lr(self):
        """Compute predicted values from linear model"""
        lr = self.fit_linear_model()
        xd = np.linspace(self.x.min(), self.x.max(), len(self.x))
        out = xd * lr[1] + lr[0]
        return out

    def predicted_dt(self):
        """Compute predicted values from dry-to-transitional model"""
        fit_dt = self.fit_piecewise_linear_dt()
        xd = np.linspace(self.x.min(), self.x.max(), len(self.x))
        out = piecewise_linear_dt(xd, *fit_dt)
        return out

    def predicted_tw(self):
        """Compute predicted values from transitional-to-wet model"""
        fit_tw = self.fit_piecewise_linear_tw()
        xd = np.linspace(self.x.min(), self.x.max(), len(self.x))
        out = piecewise_linear_tw(xd, *fit_tw)
        return out

    def predicted_dtw(self):
        """Compute predicted values from dry-to-transitional-to-wet model"""
        fit_dtw = self.fit_piecewise_linear_dtw()
        xd = np.linspace(self.x.min(), self.x.max(), len(self.x))
        out = piecewise_linear_dtw(xd, *fit_dtw)
        return out

    def compute_rss_lr(self):
        """Compute residual sum of squares for linear model"""
        y_ = self.predicted_lr()
        out = np.sum(np.square(self.y - y_))
        return out

    def compute_rss_dt(self):
        """Compute residual sum of squares for dry-to-transitional piecewise linear model"""
        df = pd.DataFrame(data={'x': self.x, 'y': self.y})
        fit_dt = self.fit_piecewise_linear_dt()
        x0 = fit_dt[0]
        y0 = fit_dt[1]
        yd = self.predicted_dt()
        rss0 = np.sum(np.square(df.y[df.x < x0] - y0))
        rss1 = np.sum(np.square(df.y[df.x >= x0] - yd[len(df.x[df.x < x0]):]))
        out = rss0 + rss1
        return out

    def compute_rss_tw(self):
        """Compute residual sum of squares for transitional-to-wet piecewise linear model"""
        df = pd.DataFrame(data={'x': self.x, 'y': self.y})
        fit_tw = self.fit_piecewise_linear_tw()
        x0 = fit_tw[0]
        y0 = fit_tw[1]
        yd = self.predicted_tw()
        rss0 = np.sum(np.square(df.y[df.x < x0] - yd[:len(df.x) - len(df.x[df.x >= x0])]))
        rss1 = np.sum(np.square(df.y[df.x >= x0] - y0))
        out = rss0 + rss1
        return out

    def compute_rss_dtw(self):
        """Compute residual sum of squares for dry-to-transitional-to-wet piecewise linear model"""
        df = pd.DataFrame(data={'x': self.x, 'y': self.y})
        fit_dtw = self.fit_piecewise_linear_dtw()
        x0 = fit_dtw[0]
        x1 = fit_dtw[1]
        y0 = fit_dtw[2]
        k1 = fit_dtw[3]
        y1 = k1*(x1 - x0) + y0  # y1 = k1*(x1 - x0) + y0
        yd = self.predicted_dtw()
        rss0 = np.sum(np.square(df.y[df.x < x0] - y0))
        rss1 = np.sum(np.square(df.y[(df.x >= x0) & (df.x < x1)] - yd[len(df.x[df.x < x0]):len(df.x) - len(df.x[df.x >= x1])]))
        rss2 = np.sum(np.square(df.y[df.x >= x1] - y1))
        out = rss0 + rss1 + rss2
        return out

    def get_models(self):
        """Get fitted models"""
        out = {'linear': self.fit_linear_model(),
               'dry-to-transitional': self.fit_piecewise_linear_dt(),
               'transitional-to-wet': self.fit_piecewise_linear_tw(),
               'dry-to-transitional-to-wet': self.fit_piecewise_linear_dtw()}
        return out

    def get_models_aic(self):
        """Compute model AIC: 2xDoF + Nxln(RSS)"""
        aic_lr = 2*2 + len(self.x)*np.log(self.compute_rss_lr())
        aic_dt = 2*3 + len(self.x)*np.log(self.compute_rss_dt())
        aic_tw = 2*3 + len(self.x)*np.log(self.compute_rss_tw())
        aic_dtw = 2*4 + len(self.x)*np.log(self.compute_rss_dtw())
        aics =  {'linear': aic_lr, 'dry-to-transitional': aic_dt, 'transitional-to-wet': aic_tw, 'dry-to-transitional-to-wet': aic_dtw}
        return aics

    def get_best_model(self):
        """Return model with both lowest RSS and smallest number of degrees of freedom"""
        aics = self.get_models_aic()
        imod = np.argmin([aic for aic in aics.values()])
        best_model = list(aics.keys())[imod]
        return best_model

    def get_best_model_number(self):
        """Return the best model number: linear=0; dry-to-trans.=1; trans.-to-wet=2; dry-to-trans.-to-wet=3"""
        numbers =  {'linear': 0, 'dry-to-transitional': 1, 'transitional-to-wet': 2, 'dry-to-transitional-to-wet': 3}
        best_model = self.get_best_model()
        out = numbers[best_model]
        return out

    def get_best_model_params(self):
        """Get model parameters"""
        mod = self.get_best_model()
        if mod == 'linear':
            pmod = self.fit_linear_model()
            wilt = np.nan
            crit = np.nan
            k = pmod[1]
        elif mod == 'dry-to-transitional':
            pmod = self.fit_piecewise_linear_dt()
            wilt = pmod[0]
            crit = np.nan
            k = pmod[2]
        elif mod == 'transitional-to-wet':
            pmod = self.fit_piecewise_linear_tw()
            wilt = np.nan
            crit = pmod[0]
            k = pmod[2]
        elif mod == 'dry-to-transitional-to-wet':
            pmod = self.fit_piecewise_linear_dtw()
            wilt = pmod[0]
            crit = pmod[1]
            k = pmod[3]
        return {'wilt': wilt, 'crit': crit, 'slope': k}

    def get_wilting_point(self):
        """Get wilting point estimated with dry-to-transitional or dry-to-transitional-to-wet model"""
        params = self.get_best_model_params()
        wilt = params['wilt']
        if (wilt > 0) and (wilt < self.x.max()):
            out = wilt
        else:
            out = np.nan
        return out

    def get_critical_point(self):
        """Get critical point estimated with dry-to-transitional-to-wet or transitional-to-wet model"""
        params = self.get_best_model_params()
        crit = params['crit']
        if (crit > 0) and (crit < self.x.max()):
            out = crit
        else:
            out = np.nan
        return out

    def get_slope(self):
        """Get dEF/dSM if a wilting and/or a critical point was founds"""
        params = self.get_best_model_params()
        wilt = self.get_wilting_point()
        crit = self.get_critical_point()
        if not (np.isnan(wilt)) or not (np.isnan(crit)):
            out = params['slope']
        else:
            out = np.nan
        return out

    def get_transitional_time_frac(self):
        """Compute the fraction of time spent in the transitional regime [%]"""
        mod = self.get_best_model()
        params = self.get_best_model_params()
        if mod == 'linear':
            x_trans = np.nan
            t_trans = np.nan
        elif mod == 'dry-to-transitional':
            x_trans = self.x[self.x > params['wilt']]
            t_trans = len(x_trans) / len(self.x)
        elif mod == 'transitional-to-wet':
            x_trans = self.x[self.x < params['crit']]
            t_trans = len(x_trans) / len(self.x)
        elif mod == 'dry-to-transitional-to-wet':
            x_trans = self.x[(self.x > params['wilt']) & (self.x < params['crit'])]
            t_trans = len(x_trans) / len(self.x)
        return t_trans*100
