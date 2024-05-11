from skfem import *
from skfem.helpers import *
import numpy as np
import matplotlib.pyplot as plt


mesh = MeshTri.init_circle(nrefs=5).smoothed()
mesh = mesh.remove_elements(mesh.elements_satisfying(lambda x: x[1] > 0))
mesh = mesh.translated((0, 0.5))
# mesh = MeshQuad.init_tensor(
#     np.linspace(-1, 1, 32),
#     np.linspace(-1, 1, 32),
# ).to_meshtri(style='x')

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
    rho=1,
    eps=1e-1,
    kappat=lambda x, y: x * 0 + 0.1,
)

(u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(x)
pbasis.plot(p, colorbar={'orientation': 'horizontal'}, shading='gouraud')
(lamx, lamxbasis), (lamy, lamybasis) = lambasis.split(lam)
lamxbasis.plot(lamx, colorbar={'orientation': 'horizontal'}, shading='gouraud')
lamybasis.plot(lamy, colorbar=True, shading='gouraud')
(ux, uxbasis), (uy, uybasis) = ubasis.split(u)
uxbasis.plot(np.sqrt(ux ** 2 + uy ** 2), colorbar={'orientation': 'horizontal'},
             nrefs=3, shading='gouraud')

lamfbasis = lambasis.boundary()
out = (Functional(lambda w: np.sqrt(dot(w['lam'] - dot(w['lam'], w.n) * w.n,
                                       w['lam'] - dot(w['lam'], w.n) * w.n)))
       .elemental(lamfbasis, lam=lam))
lengths = np.sqrt(np.diff(mesh.p[0, mesh.facets[:, mesh.boundary_facets()]], axis=0) ** 2
                  + np.diff(mesh.p[1, mesh.facets[:, mesh.boundary_facets()]], axis=0) ** 2)

plt.show()

plt.figure()
xs = mesh.p[0, mesh.facets[:, mesh.boundary_facets()]].mean(axis=0)
ys = mesh.p[1, mesh.facets[:, mesh.boundary_facets()]].mean(axis=0)
vals = out / lengths[0]
ixtop = np.nonzero(ys >= 0.5-1e-8)[0]
ixbottom = np.nonzero(ys < 0.5-1e-8)[0]
ixs = np.argsort(xs[ixtop])
plt.plot(xs[ixtop][ixs], vals[ixtop][ixs], 'b.')
plt.xlabel('$x$')
plt.ylabel('$\|{\\bf \lambda}_t\|$')
plt.tight_layout()
plt.savefig('figs/curved_lambdat_top.pdf', dpi=50)
ixs = np.argsort(xs[ixbottom])
plt.figure()
plt.plot(xs[ixbottom][ixs], vals[ixbottom][ixs], 'r.')
plt.xlabel('$x$')
plt.ylabel('$\|{\\bf \lambda}_t\|$')
plt.tight_layout()
plt.savefig('figs/curved_lambdat_bottom.pdf', dpi=50)
