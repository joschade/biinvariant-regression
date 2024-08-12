import numpy as np
import pyvista as pv
import csv
import scipy.integrate as integrate

from sklearn import svm

from morphomatics.manifold import FundamentalCoords, DifferentialCoords, PointDistributionModel
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.geom import Surface, BezierSpline
from morphomatics.stats import StatisticalShapeModel

space = 'DCM'
degree = 3
type = '_full'
n_knees = 2

if degree == 1:
    curveT = 'geodesic'
elif degree == 2:
    curveT = 'quadratic'
elif degree == 3:
    curveT = 'cubic'
else:
    curveT = 'exotic'

# create SSM (for mean computation)
if space == 'PDM':
    SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
elif space == 'FCM':
    SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
elif space == 'DCM':
    SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))

# directory = '/data/visual/online/projects/shape_trj/OAI/data/femur'
directory = '/Users/martinhanik/Documents/Arbeit/ZIB/femur'

T = []
surf = []
with open('meshes' + type + '.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        # for row in reversed(list(csv.reader(csvfile))):
        s = row[0][1:]
        if s[0] == '.':
            s = s[1:]

        print(directory + s)
        pyT = pv.read(directory + s)
        v = np.array(pyT.points)
        f = pyT.faces.reshape(-1, 4)[:, 1:]
        T.append(pyT)
        surf.append(Surface(v, f))

n_samples = int(len(surf) / 7)

SSM.construct(surf)

# use intrinsic mean as reference
ref = SSM.mean

if space == 'PDM':
    M = PointDistributionModel(ref)
elif space == 'FCM':
    M = FundamentalCoords(ref)
elif space == 'DCM':
    M = DifferentialCoords(ref)

mesh = pv.PolyData(ref.v, pyT.faces)
mesh.save('mean__full_DCM.ply')

B = Bezierfold(M, degree)

filename = curveT + '_' + type + '_' + space + '_' + str(n_knees) + 'knees.npy'

mean = np.load('P_' + filename, allow_pickle=True)
print('I have loaded P_' + filename + '...')

F_controlPoints = np.load('legs_' + filename, allow_pickle=True)
print('...and legs_' + filename)

mean = BezierSpline(M, mean)

F = []

for ff in F_controlPoints:
    leg = []
    for f in ff:
        leg.append(BezierSpline(M, f))
    F.append(leg)

disgeo = B.connec.discgeodesic(F[0][0], F[22][2], n=2)


