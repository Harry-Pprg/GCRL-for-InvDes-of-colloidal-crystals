if __name__ == "__main__":

    """Generate target ensemble."""
    import argparse

    import freud
    import numpy
    import relentless
    import gsd.hoomd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice",
        dest="lattice",
        type=str,
        choices=[
            "checkerboard",
            "cubic-diamond",
            "hex-diamond",
            "cscl",
            "open-honeycomb",
            "triangular-binary",
            "rectangular-honeycomb",
            "fcc"
        ],
        required=True,
    )
    parser.add_argument("-n", dest="n", nargs=3, type=int, required=True)
    parser.add_argument("-o", dest="outf", type=str, default="target.json")
    parser.add_argument("-i", dest="init", type=str)
    parser.add_argument(
        "--hoomd", dest="hoomd_version", type=int, choices=[2, 3], default=2
    )
    args = parser.parse_args()

    num_repeat = numpy.array(args.n)
    lattice_type = args.lattice
    types = ("A", "B")
    typeids = {type_: i for i, type_ in enumerate(types)}

    num_samples = 100
    spring_constant = 1000

    rdf_bin_size = 0.05
    rdf_stop = 5.0

    if lattice_type == "checkerboard":
        # unit cell, with each row defined as a lattice vector
        unit_cell = 2 * numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        # fractional coordinates of particles in the unit cell
        unit_cell_coords = numpy.array(
            [[0, 0, 0], [1 / 2, 0, 0], [0, 1 / 2, 0], [1 / 2, 1 / 2, 0]]
        )
        # type of each particle in the unit cell
        unit_cell_types = ["A", "B", "B", "A"]
    elif lattice_type == "cubic-diamond":
        unit_cell = numpy.sqrt(16 / 3) * numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        unit_cell_coords = numpy.array(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 0],
                [1 / 2, 0, 1 / 2],
                [0, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
            ]
        )
        unit_cell_types = ["A", "A", "A", "A", "B", "B", "B", "B"]
    elif lattice_type == "hex-diamond":    
        a = 1  # lattice constant 'a'
        c = numpy.sqrt(8 / 3) * a  # lattice constant 'c'
        z = 0.375
        unit_cell = numpy.array([
            [a/2, -a*numpy.sqrt(3)/2, 0],
            [a / 2, numpy.sqrt(3) * a / 2, 0],
            [0, 0, c]
        ])
        unit_cell_coords = numpy.array([
            [1/3, 2/3, z],          # A1
            [2/3, 1/3, 1/2+z],    # A2
            [1/3, 2/3, 1/2-z],    # B1
            [2/3, 1/3, -z]         # B2
        ])
        '''unit_cell = numpy.array([
            [a, 0, 0],
            [-a / 2, numpy.sqrt(3) * a / 2, 0],
            [0, 0, c]
        ])
        unit_cell_coords = numpy.array([
            [0, 0, 0],          # A1
            [2/3, 1/3, 1/2],    # A2
            [1/3, 2/3, 1/8],    # B1
            [0, 0, 5/8]         # B2
        ])'''
        unit_cell_types = ["A", "A", "B", "B"]
    elif lattice_type == "cscl":
        unit_cell = 2 / numpy.sqrt(3) * numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        unit_cell_coords = numpy.array([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]])
        unit_cell_types = ["A", "B"]
    elif lattice_type == "open-honeycomb":
        unit_cell = 3 * numpy.array([[1, 0, 0], [1 / 2, numpy.sqrt(3) / 2, 0], [0, 0, 0]])
        unit_cell_coords = numpy.array(
            [
                [0, 0, 0],
                [2 / 3, 0, 0],
                [0, 1 / 3, 0],
                [1 / 3, 1 / 3, 0],
                [1 / 3, 2 / 3, 0],
                [2 / 3, 2 / 3, 0],
            ]
        )
        unit_cell_types = ["A", "B", "B", "A", "B", "A"]
    elif lattice_type == "triangular-binary":
        unit_cell = 3 * numpy.array([[1, 0, 0], [1 / 2, numpy.sqrt(3) / 2, 0], [0, 0, 0]])
        unit_cell_coords = numpy.array(
            [
                [0, 0, 0],
                [1 / 3, 0, 0],
                [2 / 3, 0, 0],
                [0, 1 / 3, 0],
                [1 / 3, 1 / 3, 0],
                [2 / 3, 1 / 3, 0],
                [0, 2 / 3, 0],
                [1 / 3, 2 / 3, 0],
                [2 / 3, 2 / 3, 0],
            ]
        )
        unit_cell_types = ["B", "A", "A", "A", "B", "A", "A", "A", "B"]
    elif lattice_type == "rectangular-kagome":
        unit_cell = 2 * numpy.array([[1, 0, 0], [1 / 2, numpy.sqrt(3) / 2, 0], [0, 0, 0]])
        unit_cell_coords = numpy.array([[0, 0, 0], [1 / 2, 0, 0], [0, 1 / 2, 0]])
        unit_cell_types = ["A", "A", "B"]
    elif lattice_type == 'fcc':
        unit_cell = 2.45 / numpy.sqrt(3) * numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        unit_cell_coords = numpy.array([[0, 0, 0], [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2]])
        unit_cell_types = ["A", "B", "A", "B"]
    else:
        raise ValueError(f"Unknown lattice {lattice_type}")

    # validate types
    assert all(t in types for t in unit_cell_types)

    # replicate the unit cell coordinates
    num_cells = numpy.prod(num_repeat)
    N = num_cells * unit_cell_coords.shape[0]
    box = freud.box.Box.from_matrix((unit_cell * num_repeat).T)
    num_dim = 2 if box.is2D else 3
    if num_dim == 2 and num_repeat[2] > 1:
        raise ValueError("Cannot replicate 2d lattice in z direction")
    r0 = numpy.zeros((N, 3), dtype=float)
    for i, pos in enumerate(numpy.ndindex(*num_repeat)):
        first = i * unit_cell_coords.shape[0]
        last = first + unit_cell_coords.shape[0]
        r0[first:last] = box.make_absolute((pos + unit_cell_coords) / num_repeat)
    typeid = numpy.tile([typeids[i] for i in unit_cell_types], num_cells)

    # initialize RDF calculators
    num_bins = numpy.round(rdf_stop / rdf_bin_size).astype(int)
    rdfs = {}
    pairs = [(t1, t2) for i, t1 in enumerate(types) for t2 in types[i:]]
    for pair in pairs:
        rdfs[pair] = freud.density.RDF(bins=num_bins, r_max=rdf_stop)

    # generate ensemble of structures from Einstein crystal and sample RDF
    rng = numpy.random.default_rng()
    for sample in range(num_samples):
        print(sample)
        disp = numpy.zeros((N, 3))
        disp[:, :num_dim] = rng.normal(0, numpy.sqrt(1 / spring_constant), (N, num_dim))
        r = r0 + disp

        for i, j in pairs:
            query_args = rdfs[i, j].default_query_args
            query_args.update(exclude_ii=(i == j))
            rdfs[i, j].compute(
                (box, r[typeid == typeids[i]]),
                r[typeid == typeids[j]],
                neighbors=query_args,
                reset=False,
            )

    # save target ensemble
    if num_dim == 3:
        V = relentless.model.TriclinicBox(
            Lx=box.Lx,
            Ly=box.Ly,
            Lz=box.Lz,
            xy=box.xy,
            xz=box.xz,
            yz=box.yz,
            convention="HOOMD",
        )
    else:
        V = relentless.model.ObliqueArea(
            Lx=box.Lx, Ly=box.Ly, xy=box.xy, convention="HOOMD"
        )
    ens = relentless.model.Ensemble(
        T=1.0,
        N={i: numpy.sum(typeid == typeids[i]) for i in types},
        V=V,
    )
    for pair in pairs:
        ens.rdf[pair] = relentless.model.ensemble.RDF(
            rdfs[pair].bin_centers, rdfs[pair].rdf
        )
    ens.save(args.outf)
    print(f"Length of the sim box is {box.Lx} x {box.Ly} x {box.Lz}")
    import hoomd
    import hoomd.data as data
    import hoomd.dump as dump
    import hoomd.group as group
    # write initial configuration to file
    if args.init is not None:
        hoomd.context.initialize("")

        # For 2D, use dummy Lz=1.0 and set xy tilt
        if num_dim == 2:
            boxdim = data.boxdim(Lx=box.Lx, Ly=box.Ly, Lz=1.0, xy=box.xy)
        else:
            boxdim = data.boxdim(Lx=box.Lx, Ly=box.Ly, Lz=box.Lz,
                                xy=box.xy, xz=box.xz, yz=box.yz)

        # Create snapshot
        snapshot = data.make_snapshot(
            N=N,
            box=boxdim,
            particle_types=list(types)
        )

        # Add particle data
        snapshot.particles.position[:] = r0
        snapshot.particles.typeid[:] = typeid

        # Initialize simulation with snapshot
        hoomd.init.read_snapshot(snapshot)

        # Write a single GSD frame
        dump.gsd(filename=args.init, period=None,
                group=group.all(), overwrite=True)