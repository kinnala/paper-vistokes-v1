from skfem import *
from skfem.helpers import *
import numpy as np
import matplotlib.pyplot as plt


mesh = MeshTri.init_circle(nrefs=4)
#mesh = MeshTri().refined(4).scaled((2, 2)).translated((-1, -1))
# mesh = MeshQuad.init_tensor(
#     np.linspace(-1, 1, 10pp),
#     np.linspace(-1, 1, 10),
# ).to_meshtri(style='x')

mesh.draw()
e = (ElementVector(ElementTriP1())
     * ElementTriP1()
     * ElementVector(ElementTriSkeletonP0()))

basis = Basis(mesh, e)
fbasis = basis.boundary()

from uzawa import solver


x = solver(basis, fbasis, rho=1)

(u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(x)
ubasis.plot(u, colorbar=True).show()
pbasis.plot(p, colorbar=True, shading='gouraud').show()
lambasis.plot(lam)
