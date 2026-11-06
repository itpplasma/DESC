"""Main command line interface to DESC for solving fixed boundary equilibria."""

import sys

from desc.input_reader import InputReader


def main(cl_args=sys.argv[1:]):
    """Run the main DESC code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.

    """
    ir = InputReader(cl_args=cl_args, save_converted_vmec_input=True)

    if ir.args.version:
        return

    import desc

    if ir.args.verbose:
        print(desc.BANNER)

    import matplotlib.pyplot as plt

    from desc.backend import print_backend_info
    from desc.equilibrium import EquilibriaFamily, Equilibrium
    from desc.plotting import plot_section, plot_surfaces

    if ir.args.verbose:
        print_backend_info()
        print("Reading input from {}".format(ir.input_path))
        print("Outputs will be written to {}".format(ir.output_path))

    inputs = ir.inputs
    if (
        len(inputs) == 1
        and (inputs[-1]["pres_ratio"] is None)
        and (inputs[-1]["bdry_ratio"] is None)
    ):
        eq = Equilibrium(**inputs[-1], check_kwargs=False, ensure_nested=False)
        equil_fam = EquilibriaFamily.solve_continuation_automatic(
            eq,
            objective=inputs[-1]["objective"],
            optimizer=inputs[-1]["optimizer"],
            pert_order=inputs[-1]["pert_order"],
            ftol=inputs[-1]["ftol"],
            xtol=inputs[-1]["xtol"],
            gtol=inputs[-1]["gtol"],
            maxiter=inputs[-1]["maxiter"],
            verbose=ir.args.verbose,
            checkpoint_path=ir.output_path,
        )
    else:
        # initialize
        equil_fam = EquilibriaFamily(inputs)
        # check vmec path input
        if ir.args.guess is not None:
            if ir.args.verbose:
                print("Initial guess from {}".format(ir.args.guess))
            equil_fam[0].set_initial_guess(ir.args.guess)
        # solve equilibrium
        equil_fam.solve_continuation(
            objective=inputs[0]["objective"],
            optimizer=inputs[0]["optimizer"],
            pert_order=[inp["pert_order"] for inp in inputs],
            ftol=[inp["ftol"] for inp in inputs],
            xtol=[inp["xtol"] for inp in inputs],
            gtol=[inp["gtol"] for inp in inputs],
            maxiter=[inp["maxiter"] for inp in inputs],
            verbose=ir.args.verbose,
            checkpoint_path=ir.output_path,
        )

    if ir.free_boundary:
        equil_fam.append(
            _solve_free_boundary(
                equil_fam[-1], ir.free_boundary, ir.input_path, ir.args.verbose
            )
        )
        equil_fam.save(ir.output_path)

    if ir.args.plot > 1:
        for i, eq in enumerate(equil_fam[:-1]):
            print("Plotting solution at step {}".format(i + 1))
            _ = plot_surfaces(eq)
            plt.show()
            _ = plot_section(eq, "|F|_normalized", log=True)
            plt.show()
    if ir.args.plot > 0:
        print("Plotting final solution")
        _ = plot_surfaces(equil_fam[-1])
        plt.show()
        _ = plot_section(equil_fam[-1], "|F|_normalized", log=True)
        plt.show()


def _solve_free_boundary(eq, free_boundary, input_path, verbose):
    """Solve the free-boundary problem of a VMEC input in its mgrid field.

    Starts from the fixed-boundary solution and moves the boundary until the
    plasma and vacuum pressures balance on it, with profiles and flux fixed.
    """
    import os

    from desc.magnetic_fields import SplineMagneticField
    from desc.objectives import (
        BoundaryError,
        FixBoundaryR,
        FixBoundaryZ,
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )

    mgrid = free_boundary["mgrid_file"]
    if not os.path.isabs(mgrid):
        mgrid = os.path.join(os.path.dirname(os.path.abspath(input_path)), mgrid)
    field = SplineMagneticField.from_mgrid(mgrid, extcur=free_boundary.get("extcur"))
    constraints = (ForceBalance(eq=eq),) + tuple(
        con
        for con in get_fixed_boundary_constraints(eq)
        if not isinstance(con, (FixBoundaryR, FixBoundaryZ))
    )
    objective = ObjectiveFunction(BoundaryError(eq=eq, field=field, field_fixed=True))
    eq, _ = eq.optimize(
        objective,
        constraints,
        optimizer="proximal-lsq-exact",
        verbose=verbose,
        copy=True,
    )
    return eq


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
