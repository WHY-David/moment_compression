import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

def hex_tile_irregular(n_rows=4, n_cols=5, s=1.0, jitter=0.12, seed=42):
    """
    Draw a tile of slightly irregular flat-top hexagons.
    Parameters
    ----------
    n_rows, n_cols : int
        Grid size of hex centers (total hexagons = n_rows * n_cols).
    s : float
        Side length of the underlying regular hexagons.
    jitter : float
        Max random displacement magnitude relative to s for each unique vertex.
        (Actual displacement sampled uniformly in a disk of radius = jitter*s.)
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Unit conversion to an exact integer lattice for vertices:
    # Represent coordinates as (x, y) = ((s/2)*U, (√3 s/2)*V) with U,V ∈ ℤ.
    def to_xy(U, V):
        return np.array([ (s/2.0)*U, (np.sqrt(3.0)*s/2.0)*V ])

    # Hex center coordinates on the exact integer lattice (U,V):
    # Flat-top hex layout
    centers_UV = []
    for i in range(n_cols):
        for j in range(n_rows):
            Uc = 3*i                 # since x_center = (3/2)s * i  => U = 2x/s = 3i
            Vc = 2*j + (i % 2)       # since y_center = √3 s * (j + 0.5*(i%2)) => V = 2j + (i%2)
            centers_UV.append((i, j, (Uc, Vc)))

    # Vertex offsets (U,V) around a flat-top hex, in CCW order
    # Offsets correspond to: (±s,0), (±s/2, ±√3/2 s) expressed in U,V units
    vert_offsets = np.array([
        [ +2,  0],
        [ +1, +1],
        [ -1, +1],
        [ -2,  0],
        [ -1, -1],
        [ +1, -1],
    ], dtype=int)

    # Collect all unique vertex lattice coords (U,V) and build a map -> index
    vertex_set = set()
    hex_to_vertices = []  # list of lists of (U,V) for each hex in CCW order
    for _, _, (Uc, Vc) in centers_UV:
        verts_uv = []
        for dU, dV in vert_offsets:
            U = Uc + dU
            V = Vc + dV
            verts_uv.append((U, V))
            vertex_set.add((U, V))
        hex_to_vertices.append(verts_uv)

    # Assign each unique (U,V) a perturbed 2D point
    vertex_list = sorted(list(vertex_set))  # stable ordering
    uv_to_idx = {uv: k for k, uv in enumerate(vertex_list)}

    def inside_with_margin(p, poly_pts, margin):
        """Return True iff point p is inside the convex polygon poly_pts (CCW)
        and at least `margin` away from every edge.
        Uses half-space test with oriented edges.
        """
        px, py = p
        for i in range(len(poly_pts)):
            v0 = poly_pts[i]
            v1 = poly_pts[(i + 1) % len(poly_pts)]
            ex, ey = v1[0] - v0[0], v1[1] - v0[1]
            elen = (ex*ex + ey*ey) ** 0.5
            if elen == 0:
                return False
            ex /= elen
            ey /= elen
            dx, dy = px - v0[0], py - v0[1]
            # 2D cross of unit edge with (p - v0); for CCW polygon this is the signed
            # distance (in data units) from p to the supporting line, positive inside.
            signed_dist = ex * dy - ey * dx
            if signed_dist < margin:
                return False
        return True

    # Sample random jitter in a disk of radius = jitter*s so edges still meet
    def sample_disk(n, radius):
        # Polar sampling for uniform points in disk
        r = radius * np.sqrt(rng.random(n))
        theta = 2*np.pi * rng.random(n)
        return np.column_stack([r*np.cos(theta), r*np.sin(theta)])

    base_xy = np.array([to_xy(U, V) for (U, V) in vertex_list])
    noise = sample_disk(len(vertex_list), jitter*s)
    perturbed_xy = base_xy + noise

    # Build hexagon polygon vertices in xy
    hex_polys = []
    hex_pts_list = []  # store the vertex arrays for each hex
    for verts_uv in hex_to_vertices:
        pts = np.array([perturbed_xy[uv_to_idx[uv]] for uv in verts_uv])
        hex_pts_list.append(pts)
        hex_polys.append(Polygon(pts))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    pc = PatchCollection(hex_polys, facecolor='white', edgecolor='black', linewidth=1.0, antialiased=True)
    ax.add_collection(pc)

    # # Randomly choose 7 hexagons and draw 5–12 small solid disks inside each
    # n_choose = 7
    # min_dots, max_dots = 5, 12  # inclusive range
    # dot_r = 0.08 * s            # disk radius in data units (relative to side length)

    # # Choose unique hex indices
    # chosen_idx = rng.choice(len(hex_polys), size=n_choose, replace=False)

    # for idx in chosen_idx:
    #     pts = hex_pts_list[idx]
    #     # Bounding box for rejection sampling
    #     min_xy = pts.min(axis=0)
    #     max_xy = pts.max(axis=0)

    #     # Number of dots for this hexagon
    #     n_dots = int(rng.integers(min_dots, max_dots + 1))
    #     placed = 0
    #     margin = 4*dot_r + 1e-9  # ensure the whole disk stays inside
    #     # Shrink the bounding box by margin on all sides to reduce rejections
    #     bb_min = min_xy + margin
    #     bb_max = max_xy - margin
    #     # Guard against degenerate shrunken boxes
    #     if np.any(bb_max <= bb_min):
    #         continue
    #     while placed < n_dots:
    #         cand = bb_min + (bb_max - bb_min) * rng.random(2)
    #         if inside_with_margin(cand, pts, margin):
    #             circ = Circle((cand[0], cand[1]), radius=dot_r, facecolor='0.6', edgecolor='none', zorder=3)
    #             ax.add_patch(circ)
    #             placed += 1

    # Set limits with a small margin
    all_pts = perturbed_xy
    minx, miny = all_pts.min(axis=0)
    maxx, maxy = all_pts.max(axis=0)
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.1*dx, maxx + 0.1*dx)
    ax.set_ylim(miny - 0.1*dy, maxy + 0.1*dy)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("fig1_tmp.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    # 4 x 5 = 20 hexagons
    hex_tile_irregular(n_rows=4, n_cols=5, s=1.0, jitter=0.2, seed=13)