################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

from morphomatics.manifold import SE3
from morphomatics.stats import RiemannianRegression
import jax
import jax.numpy as jnp
from experiments.synthetic_data.dataloader import load_syndata
from morphomatics.stats import ExponentialBarycenter
import pandas as pd
import os

dir = os.path.join(os.getcwd(), "experiments/results")

if not os.path.exists(dir):
       os.makedirs(dir)


def sumOfSquared(gam, Y: jnp.array, param: jnp.array) -> float:
    return jnp.sum(jax.vmap(lambda y, t: Rie.metric.squared_dist(gam(t), y))(Y, param))

def unexplained_variance(gam, Y, param) -> float:
    """Variance in the data set that is not explained by the regressed Bézier spline.
    """
    cost = sumOfSquared(gam, Y, param)
    return cost / len(Y)

def R2statistic(gam, Y, param) -> float:
    """ Computes Fletcher's generalized R2 statistic for Bézier spline regression. For the definition see
                    Fletcher, Geodesic Regression on Riemannian Manifolds (2011), Eq. 7.

    :return: generalized R^2 statistic (in [0, 1])
    """

    # total variance
    total_var = ExponentialBarycenter.total_variance(Rie, Y)
    unexp_var = unexplained_variance(gam, Y, param)

    return 1 - unexp_var / total_var

Rie = SE3(structure='CanonicalRiemannian')

# data along group gedesic
grid, f, data = load_syndata('group')

gamgrid =  gamgrid = jnp.linspace(0., 1., num=100)

RGR = RiemannianRegression(Rie, data['se3'][0], grid)
RGR_start = RGR.trend.eval(.0)
RGR_end = RGR.trend.eval(1.)
gam_endpts = lambda t: Rie.connec.geopoint(RGR_start, RGR_end, t)
sumOfSquared(gam_endpts, data['se3'][0], grid)
unexplained_variance(gam_endpts, data['se3'][0], grid)
R2statistic(gam_endpts, data['se3'][0], grid)

R2=[]
R2_Rf = []
Rf_R2 = []
R2_endpts = []
for i in range(f.shape[0]):
    # RGR on data
    RGR = RiemannianRegression(Rie, data['se3'][i], grid)
    R2.append(RGR.R2statistic)

    # Rf o RGR(data) on Rf(data)
    c = lambda t: Rie.group.righttrans(RGR.trend.eval(t), f[i])
    Rf_R2.append(R2statistic(c, data['Rf_se3'][i], grid))

    # geodesic between
    start, end = c(.0), c(1.),
    gam = lambda t: Rie.connec.geopoint(start, end, t)
    R2_endpts.append(R2statistic(gam, data['Rf_se3'][i], grid))

    # RGR o Rf(data) on Rf(data)
    RGR_Rf = RiemannianRegression(Rie, data['Rf_se3'][i], grid)
    R2_Rf.append(RGR_Rf.R2statistic)

df = pd.DataFrame(data = {'R2(RGR, data)': R2, 'R2(Rf o RGR, Rf o Data)': Rf_R2
    , 'R2(RGR o Rf, Rf o data)': R2_Rf, 'R2(enpoint-geodesic of Rf o RGR, Rf o data)': R2_endpts})

df.to_pickle(os.path.join(dir, 'R2_RGR_synth_group.pkl'))
df.to_csv(os.path.join(dir, 'R2_RGR_synth_group.csv'))


# data along metric gedesic
grid, f, data = load_syndata('metric')

gamgrid =  gamgrid = jnp.linspace(0., 1., num=100)

RGR = RiemannianRegression(Rie, data['se3'][0], grid)
RGR_start = RGR.trend.eval(.0)
RGR_end = RGR.trend.eval(1.)
gam_endpts = lambda t: Rie.connec.geopoint(RGR_start, RGR_end, t)
sumOfSquared(gam_endpts, data['se3'][0], grid)
unexplained_variance(gam_endpts, data['se3'][0], grid)
R2statistic(gam_endpts, data['se3'][0], grid)

R2=[]
R2_Rf = []
Rf_R2 = []
R2_endpts = []
for i in range(f.shape[0]):
    # RGR on data
    RGR = RiemannianRegression(Rie, data['se3'][i], grid)
    R2.append(RGR.R2statistic)

    # Rf o RGR(data) on Rf(data)
    c = lambda t: Rie.group.righttrans(RGR.trend.eval(t), f[i])
    Rf_R2.append(R2statistic(c, data['Rf_se3'][i], grid))

    # geodesic between
    start, end = c(.0), c(1.),
    gam = lambda t: Rie.connec.geopoint(start, end, t)
    R2_endpts.append(R2statistic(gam, data['Rf_se3'][i], grid))

    # RGR o Rf(data) on Rf(data)
    RGR_Rf = RiemannianRegression(Rie, data['Rf_se3'][i], grid)
    R2_Rf.append(RGR_Rf.R2statistic)

df = pd.DataFrame(data = {'R2(RGR, data)': R2, 'R2(Rf o RGR, Rf o Data)': Rf_R2
    , 'R2(RGR o Rf, Rf o data)': R2_Rf, 'R2(enpoint-geodesic of Rf o RGR, Rf o data)': R2_endpts})

df.to_pickle(os.path.join(dir, 'R2_RGR_synth_metric.pkl'))
df.to_csv(os.path.join(dir, 'R2_RGR_synth_metric.csv'))