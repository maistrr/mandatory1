import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")

# ==============================================================================

class Wave2D:
    """Class for solving the 2D wave equation"""

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = np.linspace(0, 1, N+1)
        return np.meshgrid(xi, xi, indexing="ij", sparse=sparse)


    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D


    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx * np.pi
        ky = self.my * np.pi
        return np.sqrt(self.c**2 * (kx**2 + ky**2))


    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)


    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        xij, yij = self.create_mesh(N)
        self.xij, self.yij = xij, yij

        D = self.D2(N) / self.h**2
        self.D = D

        Unm1 = sp.lambdify((x, y), self.ue(mx, my).subs(t, 0), "numpy")(xij, yij)
        Un = Unm1 + 0.5*(self.c * self.dt)**2 * (D @ Unm1 + Unm1 @ D.T)
        return Unm1, Un


    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.h / self.c


    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        uej = sp.lambdify((x, y), self.ue(self.mx, self.my).subs(t, t0), "numpy")(self.xij, self.yij)
        return np.sqrt(self.h**2 * np.sum((u - uej)**2))


    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0] = 0
        u[-1] = 0
        u[:, 0] = 0
        u[:, -1] = 0


    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.h = 1.0 / N

        Unm1, Un = self.initialize(N, mx, my)
        D = self.D

        errors = [self.l2_error(Unm1, 0), self.l2_error(Un, self.dt)]
        data = {0: Unm1.copy()}
        if store_data > 0:
            data[1] = Un.copy()

        for n in range(1, Nt):
            Unp1 = 2*Un - Unm1 + (self.c * self.dt)**2 * (D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)
            Unm1, Un = Un, Unp1
            tn = (n + 1) * self.dt
            errors.append(self.l2_error(Un, tn))
            if store_data > 0 and (n + 1) % store_data == 0:
                data[n + 1] = Un.copy()

        if store_data == -1:
            return self.h, np.array(errors)
        return data


    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


# ==============================================================================


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        raise NotImplementedError("The D2 method is not implemented yet.")

    def ue(self, mx: int, my: int) -> sp.Expr:
        raise NotImplementedError("The ue method is not implemented yet.")

    def apply_bcs(self, u: np.ndarray):
        raise NotImplementedError("The apply_bcs method is not implemented yet.")


# ==============================================================================


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")


if __name__ == "__main__":
    test_convergence_wave2d()
    print('test_convergence_wave2d passed!')

    # test_convergence_wave2d_neumann()
    # print('test_convergence_wave2d_neumann passed!')

    # test_exact_wave2d()
    # print('All tests passed!')