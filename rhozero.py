from skfem import *
from skfem.helpers import *
import numpy as np
import matplotlib.pyplot as plt


mesh = MeshQuad.init_tensor(
    np.linspace(-1, 1, 32),
    np.linspace(-1, 1, 32),
).to_meshtri(style='x')

mesh.draw()
e = (ElementVector(ElementTriP1())
     * ElementTriP1()
     * ElementVector(ElementTriSkeletonP0()))

basis = Basis(mesh, e)
fbasis = basis.boundary()

from uzawa import solver


x = solver(
    basis,
    fbasis,
    rho=0,
    alpha1=1e-2,
    alpha2=1e-2,
    eps=1e-1,
    dirichlet=lambda x: np.isclose(x[1], 1),
    kappat=lambda x, y: 10 - np.isclose(y, -1) * (10 - 0.2),
)

(u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(x)
pbasis.plot(p, colorbar=True, shading='gouraud')
(lamx, lamxbasis), (lamy, lamybasis) = lambasis.split(lam)
lamxbasis.plot(lamx, colorbar=True, shading='gouraud')
lamybasis.plot(lamy, colorbar=True, shading='gouraud')
(ux, uxbasis), (uy, uybasis) = ubasis.split(u)
uxbasis.plot(np.sqrt(ux ** 2 + uy ** 2), colorbar=True, nrefs=3, shading='gouraud')
plt.show()
