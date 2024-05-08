from skfem import *
from skfem.helpers import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu


def solver(basis, fbasis, y=None):
    alpha1 = 1e-2
    alpha2 = 1e-2

    rho = 1

    # test different kappa's, e.g., space-dependent here
    kappat_const = 0.3
    #kappat = lambda x, y: 10 - np.isclose(x, 1) * (10 - kappat_const) * (y < 1) * (y > -1)
    kappat = lambda x, y: x * 0 + kappat_const

    N = 1e4
    nu = 1.


    @BilinearForm
    def bilinf(u, p, lam, v, q, mu, w):
        return rho * dot(u, v) + 2 * nu * ddot(sym_grad(u), sym_grad(v)) - div(v) * p + div(u) * q

    @LinearForm
    def mean(v, q, mu, w):
        return q


    # pressure mean value is set to zero using a Lagrange multiplier
    m = mean.assemble(basis)

    @BilinearForm
    def stabbilinf(u, p, lam, v, q, mu, w):
        if u.hess is not None:
            ddu = np.array([
                u.hess[0, 1, 1] + u.hess[1, 0, 1] + 2 * u.hess[0, 0, 0],
                2 * u.hess[1, 1, 1] + u.hess[0, 1, 0] + u.hess[1, 0, 0],
            ])
        else:
            ddu = 0.
        if v.hess is not None:
            ddv = np.array([
                v.hess[0, 1, 1] + v.hess[1, 0, 1] + 2 * v.hess[0, 0, 0],
                2 * v.hess[1, 1, 1] + v.hess[0, 0, 1] + v.hess[1, 0, 0],
            ])
        else:
            ddv = 0.
        return - alpha1 * w.h ** 2 * dot(rho * u - nu * ddu + grad(p), rho * v - nu * ddv - grad(q))

    def loadf(x):
        return np.array([-x[1], x[0]])

    # plot load
    # plotbasis = Basis(m, ElementVector(ElementTriP1()))
    # plotbasis.plot(plotbasis.project(loadf)).show()

    @LinearForm
    def load(v, q, mu, w):
        return dot(loadf(w.x), v)

    @LinearForm
    def stabload(v, q, mu, w):
        f = np.array([-w.x[1], w.x[0]])
        if v.hess is not None:
            ddv = np.array([
                v.hess[0, 1, 1] + v.hess[1, 0, 1] + 2 * v.hess[0, 0, 0],
                2 * v.hess[1, 1, 1] + v.hess[0, 0, 1] + v.hess[1, 0, 0],
            ])
        else:
            ddv = 0.
        return - alpha1 * w.h ** 2 * dot(f, rho * v - nu * ddv - grad(q))

    @BilinearForm
    def mixed(u, p, lam, v, q, mu, w):
        return - dot(lam, v) - 0 * dot(mu, u)

    @BilinearForm
    def stabmixed(u, p, lam, v, q, mu, w):
        sigu = - p * w.n + 2 * nu * mul(sym_grad(u), w.n)
        sigv = - q * w.n + 2 * nu * mul(sym_grad(v), w.n)
        return - alpha2 * w.h * dot(lam - sigu, 0 * mu - sigv)


    A = bilinf.assemble(basis)
    f = load.assemble(basis)
    B = mixed.assemble(fbasis)

    As = stabbilinf.assemble(basis)
    fs = stabload.assemble(basis)
    Bs = stabmixed.assemble(fbasis)

    y = basis.zeros() if y is None else y
    (u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(y)
    (ux, uxbasis), (uy, uybasis) = ubasis.split(u)
    (lamx, lamxbasis), (lamy, lamybasis) = lambasis.split(lam)

    # projections for iteration
    @LinearForm
    def proj1(mux, w):
        return w.n[0] * mux

    @LinearForm
    def proj2(mux, w):
        return w.n[1] * mux

    @BilinearForm
    def mass(lamx, mux, w):
        return lamx * mux

    lamfbasis = lambasis.boundary(quadrature=fbasis.quadrature)
    lamxfbasis = lamxbasis.boundary(quadrature=fbasis.quadrature)
    ufbasis = ubasis.boundary(quadrature=fbasis.quadrature)
    pfbasis = pbasis.boundary(quadrature=fbasis.quadrature)
    P = proj1.assemble(lamxfbasis)
    M = mass.assemble(lamxfbasis)

    I = lamxbasis.get_dofs().all()
    normal1 = lamxbasis.zeros()
    normal1[I] = solve(M[I].T[I].T, P[I])


    P = proj2.assemble(lamxfbasis)
    M = mass.assemble(lamxfbasis)

    I = lamybasis.get_dofs().all()
    normal2 = lamybasis.zeros()
    normal2[I] = solve(M[I].T[I].T, P[I])

    residual = []

    i1 = basis.get_dofs(elements=True).all(['u^1^1', 'u^2^1', 'u^2'])
    K = A + B + As + Bs
    F = f + fs
    K = bmat([[K, m[:, None]],
              [np.array([m]), None]], 'csr')
    F = np.concatenate((F, [0.]))
    AA, bb, xx, II = condense(K, F,
                              I=np.concatenate((i1, [len(F)-1])),
                              x=y)
    Alu = splu(AA)



    for itr in range(int(N)):

        # perform return mapping
        lamxdofs = basis.get_dofs(elements=True).all(['u^1^3'])
        lamydofs = basis.get_dofs(elements=True).all(['u^2^3'])
        bnd = lamxbasis.get_dofs().all()
        yprev = y.copy()
        # normal tangential split
        lam1 = y[lamxdofs][bnd]
        lam2 = y[lamydofs][bnd]
        lamn1 = (y[lamxdofs][bnd] * normal1[bnd]
                 + y[lamydofs][bnd] * normal2[bnd]) * normal1[bnd]
        lamn2 = (y[lamxdofs][bnd] * normal1[bnd]
                 + y[lamydofs][bnd] * normal2[bnd]) * normal2[bnd]
        lamt1 = y[lamxdofs][bnd] - lamn1
        lamt2 = y[lamydofs][bnd] - lamn2

        lami = lamfbasis.interpolate(lam)
        pi = pfbasis.interpolate(p)
        Pi = lami * 0
        hf = ufbasis.default_parameters()['h']
        nf = ufbasis.default_parameters()['n']
        Pi[0] = pi * nf[0]
        Pi[1] = pi * nf[1]
        ui = ufbasis.interpolate(u)
        Pu = lamfbasis.project(ui + alpha2 * hf * (lami - (-Pi + nu * mul(grad(ui) + transpose(grad(ui)), nf))))
        bndx = lambasis.get_dofs().all(['u^1'])
        bndy = lambasis.get_dofs().all(['u^2'])

        # Pun1 = (Pu[bndx] * normal1[bnd] + Pu[bndy] * normal2[bnd]) * normal1[bnd]
        # Pun2 = (Pu[bndx] * normal1[bnd] + Pu[bndy] * normal2[bnd]) * normal2[bnd]
        # Put1 = Pu[bndx] - Pun1
        # Put2 = Pu[bndy] - Pun2

        eps = 4e-1
        sx = lam1 - eps * Pu[bndx]
        sy = lam2 - eps * Pu[bndy]

        sxn = (sx * normal1[bnd] + sy * normal2[bnd]) * normal1[bnd]
        syn = (sx * normal1[bnd] + sy * normal2[bnd]) * normal2[bnd]
        sxt = sx - sxn
        syt = sy - syn

        X = lamxfbasis.project(lambda x: x[0])[bnd]
        Y = lamxfbasis.project(lambda x: x[1])[bnd]

        length = np.sqrt(sxt ** 2 + syt ** 2)
        y = basis.zeros()
        resx = sxn + kappat(X, Y) * sxt / np.maximum(kappat(X, Y), length)
        resy = syn + kappat(X, Y) * syt / np.maximum(kappat(X, Y), length)
        lamxdofs = basis.get_dofs().all(['u^1^3'])
        lamydofs = basis.get_dofs().all(['u^2^3'])
        y[lamxdofs] = resx
        y[lamydofs] = resy

        # solve
        y = np.concatenate((y, [0]))
        AA, bb, xx, II = condense(K,
                                  F,
                                  I=np.concatenate((i1, [len(F)-1])),
                                  x=y)
        y[II] = Alu.solve(bb)
        y = y[:-1]

        if itr > 2:
            diff = np.linalg.norm(yprev - y) / np.linalg.norm(yprev)
            residual.append(diff)
            print("{}: {}".format(itr, diff))
            if diff < 1e-5:
                break

        (u, ubasis), (p, pbasis), (lam, lambasis) = basis.split(y)
        (ux, uxbasis), (uy, uybasis) = ubasis.split(u)
        (lamx, lamxbasis), (lamy, lamybasis) = lambasis.split(lam)

    #plt.figure()
    #plt.loglog(uzawa)
    #plt.show()

    return y
