import math

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


FRACTURE_GRID_SIZE = 56
FRACTURE_E = 1.0
FRACTURE_NU_MIN = 0.18
FRACTURE_NU_MAX = 0.35
FRACTURE_GC_MIN = 7e-5
FRACTURE_GC_MAX = 2.2e-4
FRACTURE_MIN_DEFECTS = 0
FRACTURE_MAX_DEFECTS = 5
FRACTURE_MAIN_CRACK_Y_MIN = 0.2
FRACTURE_MAIN_CRACK_Y_MAX = 0.8
FRACTURE_MAIN_CRACK_MIN_SEGMENTS = 2
FRACTURE_MAIN_CRACK_MAX_SEGMENTS = 5
FRACTURE_MAIN_CRACK_TOTAL_LENGTH_MIN = 0.14
FRACTURE_MAIN_CRACK_TOTAL_LENGTH_MAX = 0.30
FRACTURE_MAIN_CRACK_INITIAL_THETA_STD = 0.10
FRACTURE_MAIN_CRACK_THETA_STEP_STD = 0.20
FRACTURE_MAIN_CRACK_THETA_MIN = -0.7
FRACTURE_MAIN_CRACK_THETA_MAX = 0.7
FRACTURE_MAIN_CRACK_WIDTH_PX = 0.36
FRACTURE_DEFECT_CRACK_MIN_SEGMENTS = 1
FRACTURE_DEFECT_CRACK_MAX_SEGMENTS = 3
FRACTURE_DEFECT_CRACK_TOTAL_LENGTH_MIN = 0.025
FRACTURE_DEFECT_CRACK_TOTAL_LENGTH_MAX = 0.08
FRACTURE_DEFECT_CRACK_THETA_STEP_STD = 0.35
FRACTURE_DEFECT_CRACK_WIDTH_PX = 0.28
FRACTURE_LOW_STRAIN_MIN = 0.006
FRACTURE_LOW_STRAIN_MAX = 0.022
FRACTURE_HIGH_STRAIN_MIN = 0.055
FRACTURE_HIGH_STRAIN_MAX = 0.082
FRACTURE_BIAXIAL_STRAIN_MIN = 0.035
FRACTURE_BIAXIAL_STRAIN_MAX = 0.060
FRACTURE_STEPS = 18
FRACTURE_INNER_ITERS = 3
FRACTURE_ELL_FACTOR = 1.7
FRACTURE_KAPPA = 1e-6
FRACTURE_DAMAGE_VMIN = 0.0
FRACTURE_DAMAGE_VMAX = 1.0
FRACTURE_CONDITIONING_NAMES = ("nu", "G_c", "epsilon_h", "epsilon_v")
FRACTURE_CONDITIONING_TRANSFORMS = ("linear", "log", "linear", "linear")


def _uniform_mean_std(min_value, max_value):
    return (
        0.5 * (float(min_value) + float(max_value)),
        (float(max_value) - float(min_value)) / math.sqrt(12.0),
    )


def _log_uniform_mean_std(min_value, max_value):
    return (
        0.5 * (math.log(float(min_value)) + math.log(float(max_value))),
        (math.log(float(max_value)) - math.log(float(min_value))) / math.sqrt(12.0),
    )


def _strain_mean_std():
    modes = [
        _uniform_mean_std(FRACTURE_LOW_STRAIN_MIN, FRACTURE_LOW_STRAIN_MAX),
        _uniform_mean_std(FRACTURE_HIGH_STRAIN_MIN, FRACTURE_HIGH_STRAIN_MAX),
        _uniform_mean_std(FRACTURE_BIAXIAL_STRAIN_MIN, FRACTURE_BIAXIAL_STRAIN_MAX),
    ]
    means = np.asarray([mean for mean, _ in modes], dtype=np.float64)
    variances = np.asarray([std * std for _, std in modes], dtype=np.float64)
    mean = float(means.mean())
    variance = float(np.mean(variances + means * means) - mean * mean)
    return mean, math.sqrt(max(variance, 0.0))


FRACTURE_NU_MEAN, FRACTURE_NU_STD = _uniform_mean_std(FRACTURE_NU_MIN, FRACTURE_NU_MAX)
FRACTURE_LOG_GC_MEAN, FRACTURE_LOG_GC_STD = _log_uniform_mean_std(FRACTURE_GC_MIN, FRACTURE_GC_MAX)
FRACTURE_EPS_MEAN, FRACTURE_EPS_STD = _strain_mean_std()
FRACTURE_CONDITIONING_MEAN = (
    FRACTURE_NU_MEAN,
    FRACTURE_LOG_GC_MEAN,
    FRACTURE_EPS_MEAN,
    FRACTURE_EPS_MEAN,
)
FRACTURE_CONDITIONING_STD = (
    FRACTURE_NU_STD,
    FRACTURE_LOG_GC_STD,
    FRACTURE_EPS_STD,
    FRACTURE_EPS_STD,
)


def random_piecewise_linear_crack(rng):
    y = rng.uniform(FRACTURE_MAIN_CRACK_Y_MIN, FRACTURE_MAIN_CRACK_Y_MAX)
    x = 0.0
    pts = [(x, y)]

    nseg = int(rng.integers(FRACTURE_MAIN_CRACK_MIN_SEGMENTS, FRACTURE_MAIN_CRACK_MAX_SEGMENTS + 1))
    total_len = rng.uniform(FRACTURE_MAIN_CRACK_TOTAL_LENGTH_MIN, FRACTURE_MAIN_CRACK_TOTAL_LENGTH_MAX)
    seg_lens = rng.dirichlet(np.ones(nseg)) * total_len
    theta = rng.normal(0.0, FRACTURE_MAIN_CRACK_INITIAL_THETA_STD)

    for seg_len in seg_lens:
        theta = np.clip(
            theta + rng.normal(0.0, FRACTURE_MAIN_CRACK_THETA_STEP_STD),
            FRACTURE_MAIN_CRACK_THETA_MIN,
            FRACTURE_MAIN_CRACK_THETA_MAX,
        )
        x_new = np.clip(x + seg_len * np.cos(theta), x + 0.012, 0.82)
        y_new = np.clip(y + seg_len * np.sin(theta), 0.06, 0.94)
        pts.append((x_new, y_new))
        x, y = x_new, y_new

    return np.array(pts)


def random_internal_piecewise_crack(rng):
    x = rng.uniform(0.12, 0.88)
    y = rng.uniform(0.12, 0.88)
    pts = [(x, y)]

    nseg = int(rng.integers(FRACTURE_DEFECT_CRACK_MIN_SEGMENTS, FRACTURE_DEFECT_CRACK_MAX_SEGMENTS + 1))
    total_len = rng.uniform(FRACTURE_DEFECT_CRACK_TOTAL_LENGTH_MIN, FRACTURE_DEFECT_CRACK_TOTAL_LENGTH_MAX)
    seg_lens = rng.dirichlet(np.ones(nseg)) * total_len
    theta = rng.uniform(0.0, 2.0 * np.pi)

    for seg_len in seg_lens:
        theta = theta + rng.normal(0.0, FRACTURE_DEFECT_CRACK_THETA_STEP_STD)
        x_new = np.clip(x + seg_len * np.cos(theta), 0.03, 0.97)
        y_new = np.clip(y + seg_len * np.sin(theta), 0.03, 0.97)
        pts.append((x_new, y_new))
        x, y = x_new, y_new

    return np.array(pts)


def rasterize_polyline(pts, n, width_px):
    mask = np.zeros((n, n), dtype=bool)

    for a, b in zip(pts[:-1], pts[1:]):
        steps = int(np.ceil(max(abs(b - a)) * (n - 1) * 8)) + 1
        t = np.linspace(0.0, 1.0, steps)
        x = np.clip(np.round((a[0] * (1 - t) + b[0] * t) * (n - 1)).astype(int), 0, n - 1)
        y = np.clip(np.round((a[1] * (1 - t) + b[1] * t) * (n - 1)).astype(int), 0, n - 1)
        mask[y, x] = True

    dist = distance_transform_edt(~mask)
    damage = np.exp(-(dist**2) / (2.0 * float(width_px) ** 2))
    damage[mask] = 1.0
    return np.clip(damage, 0.0, 1.0)


def random_initial_damage_with_defects(rng, n):
    damage = rasterize_polyline(
        random_piecewise_linear_crack(rng),
        n=n,
        width_px=FRACTURE_MAIN_CRACK_WIDTH_PX,
    )
    num_defects = int(rng.integers(FRACTURE_MIN_DEFECTS, FRACTURE_MAX_DEFECTS + 1))

    for _ in range(num_defects):
        defect = rasterize_polyline(
            random_internal_piecewise_crack(rng),
            n=n,
            width_px=FRACTURE_DEFECT_CRACK_WIDTH_PX,
        )
        damage = np.maximum(damage, defect)

    return damage, num_defects


def random_material(rng):
    nu = rng.uniform(FRACTURE_NU_MIN, FRACTURE_NU_MAX)
    Gc = 10 ** rng.uniform(np.log10(FRACTURE_GC_MIN), np.log10(FRACTURE_GC_MAX))
    return nu, Gc


def random_biaxial_strain(rng):
    mode = rng.choice(["vertical_dominant", "horizontal_dominant", "biaxial"])
    if mode == "vertical_dominant":
        eps_h = rng.uniform(FRACTURE_LOW_STRAIN_MIN, FRACTURE_LOW_STRAIN_MAX)
        eps_v = rng.uniform(FRACTURE_HIGH_STRAIN_MIN, FRACTURE_HIGH_STRAIN_MAX)
    elif mode == "horizontal_dominant":
        eps_h = rng.uniform(FRACTURE_HIGH_STRAIN_MIN, FRACTURE_HIGH_STRAIN_MAX)
        eps_v = rng.uniform(FRACTURE_LOW_STRAIN_MIN, FRACTURE_LOW_STRAIN_MAX)
    else:
        eps_h = rng.uniform(FRACTURE_BIAXIAL_STRAIN_MIN, FRACTURE_BIAXIAL_STRAIN_MAX)
        eps_v = rng.uniform(FRACTURE_BIAXIAL_STRAIN_MIN, FRACTURE_BIAXIAL_STRAIN_MAX)
    return eps_h, eps_v, mode


def nominal_plane_stress(eps_h, eps_v, E=FRACTURE_E, nu=0.3):
    C11 = E / (1.0 - nu**2)
    C12 = E * nu / (1.0 - nu**2)
    sigma_h = C11 * eps_h + C12 * eps_v
    sigma_v = C12 * eps_h + C11 * eps_v
    return sigma_h, sigma_v


def make_mesh(n):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack([X.ravel(), Y.ravel()])

    elems = []
    for i in range(n - 1):
        for j in range(n - 1):
            n0 = i * n + j
            n1 = i * n + j + 1
            n2 = (i + 1) * n + j
            n3 = (i + 1) * n + j + 1
            elems.append([n0, n1, n3])
            elems.append([n0, n3, n2])

    return coords, np.array(elems)


def tri_B_area(coords_e):
    x = coords_e[:, 0]
    y = coords_e[:, 1]
    M = np.array(
        [
            [1.0, x[0], y[0]],
            [1.0, x[1], y[1]],
            [1.0, x[2], y[2]],
        ]
    )

    area = abs(np.linalg.det(M)) / 2.0
    inv = np.linalg.inv(M)
    bx = inv[1, :]
    by = inv[2, :]
    B = np.zeros((3, 6))

    for a in range(3):
        B[0, 2 * a] = bx[a]
        B[1, 2 * a + 1] = by[a]
        B[2, 2 * a] = by[a]
        B[2, 2 * a + 1] = bx[a]

    grads = np.column_stack([bx, by])
    return B, grads, area


def precompute_fe(n, E=FRACTURE_E, nu=0.3):
    coords, elems = make_mesh(n)
    nn = coords.shape[0]
    ndof = 2 * nn
    C = E / (1.0 - nu**2) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )

    erows = []
    ecols = []
    edata = []
    drows = []
    dcols = []
    ddata = []
    mass = np.zeros(nn)
    elem_nodes = []
    elem_dofs = []
    B_list = []
    area_list = []

    for nodes in elems:
        B, grads, area = tri_B_area(coords[nodes])
        Ke = area * (B.T @ C @ B)
        dofs = []
        for node in nodes:
            dofs += [2 * node, 2 * node + 1]

        elem_nodes.append(nodes)
        elem_dofs.append(dofs)
        B_list.append(B)
        area_list.append(area)

        for a in range(6):
            for b in range(6):
                erows.append(dofs[a])
                ecols.append(dofs[b])
                edata.append(Ke[a, b])

        Kg = area * (grads @ grads.T)
        for a in range(3):
            mass[nodes[a]] += area / 3.0
            for b in range(3):
                drows.append(nodes[a])
                dcols.append(nodes[b])
                ddata.append(Kg[a, b])

    Kgrad = coo_matrix((ddata, (drows, dcols)), shape=(nn, nn)).tocsr()
    return {
        "n": n,
        "C": C,
        "elem_nodes": np.array(elem_nodes),
        "elem_dofs": np.array(elem_dofs),
        "B": np.array(B_list),
        "area": np.array(area_list),
        "erows": np.array(erows),
        "ecols": np.array(ecols),
        "edata": np.array(edata),
        "Kgrad": Kgrad,
        "mass": mass,
        "ndof": ndof,
        "nn": nn,
    }


def assemble_elastic(d, pre, kappa=FRACTURE_KAPPA):
    elem_d = d[pre["elem_nodes"]].mean(axis=1)
    g = (1.0 - elem_d) ** 2 + kappa
    data = pre["edata"] * np.repeat(g, 36)
    return coo_matrix((data, (pre["erows"], pre["ecols"])), shape=(pre["ndof"], pre["ndof"])).tocsr()


def fixed_dofs_values_biaxial(n, eps_h, eps_v, load_scale):
    fixed = []
    values = []
    ux_right = load_scale * eps_h
    uy_top = load_scale * eps_v

    for i in range(n):
        left = i * n
        right = i * n + (n - 1)
        fixed.append(2 * left)
        values.append(0.0)
        fixed.append(2 * right)
        values.append(ux_right)

    for j in range(n):
        bottom = j
        top = (n - 1) * n + j
        fixed.append(2 * bottom + 1)
        values.append(0.0)
        fixed.append(2 * top + 1)
        values.append(uy_top)

    fixed = np.array(fixed)
    values = np.array(values)
    unique_fixed, idx = np.unique(fixed, return_index=True)
    return unique_fixed, values[idx]


def solve_elastic_biaxial(d, load_scale, eps_h, eps_v, pre, kappa=FRACTURE_KAPPA):
    K = assemble_elastic(d, pre, kappa=kappa)
    fixed, fixed_values = fixed_dofs_values_biaxial(pre["n"], eps_h, eps_v, load_scale)
    free_mask = np.ones(pre["ndof"], dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    rhs = -K[free][:, fixed] @ fixed_values
    u_free = spsolve(K[free][:, free], rhs)
    u = np.zeros(pre["ndof"])
    u[fixed] = fixed_values
    u[free] = u_free
    return u


def update_history(u, H, pre):
    ue = u[pre["elem_dofs"]]
    eps = np.einsum("eij,ej->ei", pre["B"], ue)
    psi = 0.5 * np.einsum("ei,ij,ej->e", eps, pre["C"], eps)
    accum = np.zeros_like(H)

    for a in range(3):
        np.add.at(accum, pre["elem_nodes"][:, a], psi * pre["area"] / 3.0)

    H_now = accum / (pre["mass"] + 1e-30)
    return np.maximum(H, H_now)


def solve_damage(H, d_old, d_init, pre, Gc, ell):
    mass = pre["mass"]
    A = Gc * ell * pre["Kgrad"] + diags(mass * (Gc / ell + 2.0 * H))
    b = mass * (2.0 * H)
    d = spsolve(A, b)
    d = np.clip(d, 0.0, 1.0)
    d = np.maximum(d, d_old)
    d = np.maximum(d, d_init)
    return d


def simulate_phase_field_biaxial(
    d_init_img,
    eps_h,
    eps_v,
    pre,
    steps=FRACTURE_STEPS,
    inner_iters=FRACTURE_INNER_ITERS,
    Gc=1.2e-4,
    ell=None,
    kappa=FRACTURE_KAPPA,
):
    n = pre["n"]
    h = 1.0 / (n - 1)
    if ell is None:
        ell = FRACTURE_ELL_FACTOR * h

    d_init = d_init_img.ravel().copy()
    d = d_init.copy()
    H = np.zeros(pre["nn"])

    for k in range(1, steps + 1):
        load_scale = k / steps
        for _ in range(inner_iters):
            u = solve_elastic_biaxial(d, load_scale, eps_h, eps_v, pre, kappa=kappa)
            H = update_history(u, H, pre)
            d_new = solve_damage(H, d, d_init, pre, Gc=Gc, ell=ell)
            if np.max(np.abs(d_new - d)) < 1e-4:
                d = d_new
                break
            d = d_new

    return d.reshape(n, n)


def _damage_to_rgb(damage, output_size):
    damage = np.clip(np.asarray(damage, dtype=np.float32), FRACTURE_DAMAGE_VMIN, FRACTURE_DAMAGE_VMAX)
    image = Image.fromarray(np.ascontiguousarray((damage * 255.0).round().astype(np.uint8)), mode="L")
    if image.size != (int(output_size), int(output_size)):
        image = image.resize((int(output_size), int(output_size)), Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB")).copy()


def generate_fracture_image_pairs(
    seeds,
    sim_nx=FRACTURE_GRID_SIZE,
    output_size=256,
    E=FRACTURE_E,
    ell_factor=FRACTURE_ELL_FACTOR,
    steps=FRACTURE_STEPS,
    inner_iters=FRACTURE_INNER_ITERS,
    kappa=FRACTURE_KAPPA,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if sim_nx < 8:
        raise ValueError("fracture_grid_size must be at least 8.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")

    records = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        nu, Gc = random_material(rng)
        pre = precompute_fe(int(sim_nx), E=E, nu=nu)
        d0, num_defects = random_initial_damage_with_defects(rng, int(sim_nx))
        eps_h, eps_v, strain_mode = random_biaxial_strain(rng)
        sigma_h, sigma_v = nominal_plane_stress(eps_h, eps_v, E=E, nu=nu)
        dT = simulate_phase_field_biaxial(
            d0,
            eps_h,
            eps_v,
            pre,
            steps=steps,
            inner_iters=inner_iters,
            Gc=Gc,
            ell=ell_factor / (int(sim_nx) - 1),
            kappa=kappa,
        )

        params = {
            "seed": int(seed),
            "pde": "fracture",
            "sim_nx": int(sim_nx),
            "output_size": int(output_size),
            "E": float(E),
            "nu": float(nu),
            "Gc": float(Gc),
            "epsilon_h": float(eps_h),
            "epsilon_v": float(eps_v),
            "strain_mode": str(strain_mode),
            "sigma_h0": float(sigma_h),
            "sigma_v0": float(sigma_v),
            "num_defects": int(num_defects),
            "ell": float(ell_factor / (int(sim_nx) - 1)),
            "ell_factor": float(ell_factor),
            "steps": int(steps),
            "inner_iters": int(inner_iters),
            "kappa": float(kappa),
            "solution_target": "phase_field_final_damage",
            "conditioning_names": FRACTURE_CONDITIONING_NAMES,
            "conditioning_values": (float(nu), float(Gc), float(eps_h), float(eps_v)),
        }
        records.append(
            (
                _damage_to_rgb(dT, output_size),
                _damage_to_rgb(d0, output_size),
                params,
            )
        )

    return records
