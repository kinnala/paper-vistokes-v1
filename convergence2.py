from skfem import *
from skfem.helpers import *
import numpy as np
import matplotlib.pyplot as plt


hs = []
ul2 = []
uh1 = []
pl2 = []
lamhm1 = []

uxfun = None

for nnodes in [4, 8, 16, 32, 64]:
    mesh = MeshQuad.init_tensor(
        np.linspace(-1, 1, nnodes),
        np.linspace(-1, 1, nnodes),
    ).to_meshtri(style='x')
    # mesh = MeshTri.init_circle(nrefs=int(nnodes / 4))

    mesh.draw()
    plt.savefig('figs/uniform_{}_mesh.pdf'.format(nnodes))

    e = ElementVector(ElementTriP1()) * ElementTriP1() * ElementVector(ElementTriSkeletonP0())

    # while not presented in the paper,
    # it is possible to try different finite element combinations:

    #e = ElementVector(ElementTriP2G()) * ElementDG(ElementTriP1()) * ElementVector(ElementTriSkeletonP1())
    #e = ElementVector(ElementTriP2G()) * ElementTriP1() * ElementVector(ElementTriP1())
    #e = ElementVector(ElementTriP1()) * ElementTriP1() * ElementVector(ElementTriSkeletonP1())
    #e = ElementVector(ElementTriP1()) * ElementTriP1() * ElementVector(ElementTriP1())
    #e = ElementVector(ElementTriP2G()) * ElementDG(ElementTriP1()) * ElementVector(ElementTriSkeletonP0())
    #e = ElementVector(ElementTriP1()) * ElementTriP1() * ElementVector(ElementTriSkeletonP0())
    #e = ElementVector(ElementTriP1()) * ElementTriP1() * ElementVector(ElementTriSkeletonP1())
    #e = ElementVector(ElementTriP2()) * ElementTriP1() * ElementVector(ElementTriSkeletonP1())
    #e = ElementVector(ElementTriP3()) * ElementTriP2() * ElementVector(ElementTriSkeletonP1())
    #e = ElementVector(ElementTriP2()) * ElementTriP2() * ElementVector(ElementTriSkeletonP1())

    basis = Basis(mesh, e)
    fbasis = basis.boundary()

    from uzawa import solver

    # plotting
    # uncomment some of the following lines to get different plots
    x = solver(basis, fbasis)

    (u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(x)
    #ubasis.plot(u, colorbar=True).show()
    pbasis.plot(p, colorbar=True, shading='gouraud').show()
    #lambasis.plot(lam)
    (lamx, lamxbasis), (lamy, lamybasis) = lambasis.split(lam)
    #lamxbasis.plot(lamx, colorbar=True, shading='gouraud')
    #lamybasis.plot(lamy, colorbar=True, shading='gouraud')
    (ux, uxbasis), (uy, uybasis) = ubasis.split(u)
    #uxbasis.plot(np.sqrt(ux ** 2 + uy ** 2), colorbar=True, nrefs=3, shading='gouraud').show()

    if 1:
        dlamxfun = lamxbasis.interpolator(lamx)
        xs = np.linspace(-0.999, 0.999, 200)

        dlamyfun = lamybasis.interpolator(lamy)
        xs = np.linspace(-0.999, 0.999, 200)

        fig = plt.figure()
        fig.set_size_inches(5/1.2, 4/1.2)
        plt.plot(xs, dlamxfun(np.array([0*xs + 1, xs])), 'k-')
        plt.xlabel('$y$')
        plt.ylabel('$\lambda_n$')
        plt.tight_layout()
        plt.savefig('figs/uniform_{}_lambdan.pdf'.format(nnodes), dpi=50)

        fig = plt.figure()
        fig.set_size_inches(5/1.2, 4/1.2)
        plt.plot(xs, np.abs(dlamyfun(np.array([0*xs + 1, xs]))), 'k-')
        plt.xlabel('$y$')
        plt.ylabel('$\|{\\bf \lambda}_t\|$')
        plt.tight_layout()
        plt.savefig('figs/uniform_{}_lambdat.pdf'.format(nnodes), dpi=50)

    # calculate errors

    if uxfun is not None:
        @Functional
        def ul2err(w):
            return ((w['u'][0] - uxfun(w.x)) ** 2
                    + (w['u'][1] - uyfun(w.x)) ** 2)

        @Functional
        def uh1err(w):
            return ((w['u'].grad[0][0] - duxxfun(w.x)) ** 2
                    + (w['u'].grad[0][1] - duxyfun(w.x)) ** 2
                    + (w['u'].grad[1][0] - duyxfun(w.x)) ** 2
                    + (w['u'].grad[1][1] - duyyfun(w.x)) ** 2)

        @Functional
        def pl2err(w):
            return ((w['p'] - pfun(w.x)) ** 2)

        @Functional
        def lamerr(w):
            return w.h * ((w['lam'][0] - lamxfun(w.x)) ** 2
                          + (w['lam'][1] - lamyfun(w.x)) ** 2)

        ul2.append(np.sqrt(ul2err.assemble(ubasis, u=u)))
        uh1.append(np.sqrt(uh1err.assemble(ubasis, u=u)))
        pl2.append(np.sqrt(pl2err.assemble(pbasis, p=p)))
        lamhm1.append(np.sqrt(lamerr.assemble(lambasis.boundary(), lam=lam)))
        hs.append(mesh.param())

    uxfun = uxbasis.interpolator(ux)
    uyfun = uybasis.interpolator(uy)

    duxbasis = uxbasis.with_element(ElementDG(ElementTriP1()))
    duxx = duxbasis.project(grad(uxbasis.interpolate(ux))[0])
    duxy = duxbasis.project(grad(uxbasis.interpolate(ux))[1])
    duyx = duxbasis.project(grad(uxbasis.interpolate(uy))[0])
    duyy = duxbasis.project(grad(uxbasis.interpolate(uy))[1])

    duxxfun = duxbasis.interpolator(duxx)
    duxyfun = duxbasis.interpolator(duxy)
    duyxfun = duxbasis.interpolator(duyx)
    duyyfun = duxbasis.interpolator(duyy)

    pfun = pbasis.interpolator(p)

    lamxfun = lamxbasis.interpolator(lamx)
    lamyfun = lamybasis.interpolator(lamy)


hs = np.array(hs)
uh1 = np.sqrt(np.array(uh1) ** 2 + np.array(ul2) ** 2)

width = 5 * 1.2
height = 4 * 1.2
fig = plt.figure()
fig.set_size_inches(width, height)
plt.loglog(hs, uh1, 'ko-')
plt.loglog(hs, hs / hs[0] * uh1[0], 'k:')
plt.xlabel('mesh parameter')
plt.ylabel('error')
plt.legend(['$\|{\\bf u}_{2h} - {\\bf u}_h\|_{1,\Omega}$', '$O(h)$'])
plt.tight_layout()
plt.grid('on')
plt.savefig('figs/uniform_{}_uh1.pdf'.format(nnodes), dpi=50)

fig = plt.figure()
fig.set_size_inches(width, height)
plt.loglog(hs, pl2, 'ko-')
plt.loglog(hs, hs / hs[0] * pl2[0], 'k:')
plt.xlabel('mesh parameter')
plt.ylabel('error')
plt.legend(['$\|p_{2h} - p_{h}\|_{0,\Omega}$', '$O(h)$'])
plt.tight_layout()
plt.grid('on')
plt.savefig('figs/uniform_{}_pl2.pdf'.format(nnodes), dpi=50)

fig = plt.figure()
fig.set_size_inches(width, height)
plt.loglog(hs, lamhm1, 'ko-')
plt.loglog(hs, hs / hs[0] * lamhm1[0], 'k:')
plt.xlabel('mesh parameter')
plt.ylabel('error')
plt.legend(['$\|h^{1/2}({\\bf \lambda}_{2h} - {\\bf \lambda}_h)\|_{0,\partial \Omega}$', '$O(h)$'])
plt.tight_layout()
plt.grid('on')
plt.savefig('figs/uniform_{}_lamhm1.pdf'.format(nnodes), dpi=50)

total = np.sqrt(uh1 ** 2 + np.array(pl2) ** 2 + np.array(lamhm1) ** 2)
fig = plt.figure()
fig.set_size_inches(width, height)
plt.loglog(hs, total, 'ko-')
plt.loglog(hs, hs / hs[0] * total[0], 'k:')
plt.xlabel('mesh parameter')
plt.ylabel('error')
plt.legend(['error', '$O(h)$'])
plt.tight_layout()
plt.grid('on')
plt.savefig('figs/uniform_{}_total.pdf'.format(nnodes), dpi=50)
