"""Reusable geometry, radar, and differentiable flow helpers.

Training orchestration intentionally lives in radar.ipynb.
"""

import csv
import math
from pathlib import Path

import torch


# =============================================================================
# Geometry: smooth 100-vertex profile with exact polygon area
# =============================================================================


def polygon_area(vertices):
    nxt = torch.roll(vertices, -1, dims=-2)
    return 0.5 * torch.sum(
        vertices[..., 0] * nxt[..., 1]
        - vertices[..., 1] * nxt[..., 0],
        dim=-1,
    )


def naca_closed_base(s):
    """NACA-like half-thickness with rounded nose and closed trailing edge."""
    s = s.clamp(0.0, 1.0)
    return (
        0.2969 * torch.sqrt(s.clamp_min(1e-12))
        - 0.1260 * s
        - 0.3516 * s.square()
        + 0.2843 * s.pow(3)
        - 0.1036 * s.pow(4)
    ).clamp_min(0.0)


def profile_vertices(
    params,
    target_area=0.205,
    n_vertices=100,
    chord_min=0.52,
    chord_max=0.96,
):
    """
    params: [B, 1 + M]

    The first parameter controls chord. Remaining parameters control M smooth
    log-thickness cosine modes. Exactly 100 polygon vertices are returned.
    Area is imposed exactly on the discrete polygon by scaling y.
    """
    if n_vertices != 100:
        raise ValueError("This profile parameterization uses exactly 100 vertices.")

    batch = params.shape[0]
    modes_count = params.shape[1] - 1

    chord = chord_min + (chord_max - chord_min) * torch.sigmoid(params[:, 0])
    coefficients = 0.22 * torch.tanh(params[:, 1:])

    # Counter-clockwise: upper TE -> LE, then lower LE -> TE.
    q = torch.linspace(
        0.0, math.pi, 51, device=params.device, dtype=params.dtype
    )
    s_upper = 0.5 * (1.0 + torch.cos(q))
    s_lower = 0.5 * (1.0 - torch.cos(q))[1:-1]
    s = torch.cat((s_upper, s_lower))
    sign = torch.cat((torch.ones_like(s_upper), -torch.ones_like(s_lower)))

    base = naca_closed_base(s)
    if modes_count:
        modes = torch.arange(
            1, modes_count + 1, device=params.device, dtype=params.dtype
        )
        basis = torch.cos(math.pi * modes[:, None] * s[None])
        modifier = torch.exp(torch.einsum("bm,mp->bp", coefficients, basis))
    else:
        modifier = torch.ones(batch, s.numel(), device=params.device, dtype=params.dtype)

    x = chord[:, None] * (s[None] - 0.5)
    y_unscaled = sign[None] * base[None] * modifier
    provisional = torch.stack((x, y_unscaled), dim=-1)

    area_unscaled = polygon_area(provisional)
    vertical_scale = target_area / area_unscaled
    y = y_unscaled * vertical_scale[:, None]
    vertices = torch.stack((x, y), dim=-1)

    return vertices, chord, coefficients, vertical_scale


def straight_through_solid_mask(soft_mask, threshold=0.5):
    """Use a sharp solid in the forward solve and the soft mask for gradients."""
    hard_mask = (soft_mask >= threshold).to(soft_mask.dtype)
    return soft_mask + (hard_mask - soft_mask).detach()


def profile_mask(
    params,
    solver,
    target_area=0.205,
    interface_cells=1.0,
    sharp_forward=True,
    grid_offsets=None,
):
    """
    Efficient differentiable rasterization of the same analytic profile used
    to construct the 100 polygon vertices.
    """
    if grid_offsets is None:
        offsets = torch.zeros(
            params.shape[0], 2, device=params.device, dtype=params.dtype
        )
    else:
        offsets = torch.as_tensor(
            grid_offsets, device=params.device, dtype=params.dtype
        ).reshape(-1, 2)
        if params.shape[0] == 1 and offsets.shape[0] > 1:
            params = params.expand(offsets.shape[0], -1)
        elif params.shape[0] != offsets.shape[0]:
            raise ValueError(
                "params batch must be one or match the number of grid offsets"
            )

    vertices, chord, coefficients, vertical_scale = profile_vertices(
        params, target_area=target_area
    )

    # Sub-cell translations expose grid-alignment errors. Averaging their drag
    # prevents the optimizer from tailoring a shape to one lattice placement.
    x = solver.xx[None] - offsets[:, 0, None, None] * solver.dx
    y = solver.yy[None] - offsets[:, 1, None, None] * solver.dx
    s = x / chord[:, None, None] + 0.5
    s_clamped = s.clamp(0.0, 1.0)

    base = naca_closed_base(s_clamped)
    if coefficients.shape[1]:
        modes = torch.arange(
            1,
            coefficients.shape[1] + 1,
            device=params.device,
            dtype=params.dtype,
        )
        phase = math.pi * modes[None, :, None, None] * s_clamped[:, None]
        modifier = torch.exp(
            torch.sum(coefficients[:, :, None, None] * torch.cos(phase), dim=1)
        )
    else:
        modifier = torch.ones_like(base)

    half_thickness = vertical_scale[:, None, None] * base * modifier
    eps = interface_cells * solver.dx

    vertical_inside = torch.sigmoid((half_thickness - y.abs()) / eps)
    left_inside = torch.sigmoid((x + 0.5 * chord[:, None, None]) / eps)
    right_inside = torch.sigmoid((0.5 * chord[:, None, None] - x) / eps)
    soft_mask = vertical_inside * left_inside * right_inside
    mask = (
        straight_through_solid_mask(soft_mask)
        if sharp_forward
        else soft_mask
    )

    return mask, vertices, chord, coefficients


def circle_vertices(device, dtype, target_area=0.205, n_vertices=100):
    theta = torch.arange(n_vertices, device=device, dtype=dtype)
    theta = theta * (2.0 * math.pi / n_vertices)
    radius = math.sqrt(target_area / math.pi)
    vertices = torch.stack(
        (radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1
    )
    return vertices[None]


def circle_mask(
    solver,
    target_area=0.205,
    interface_cells=1.0,
    sharp_forward=True,
    grid_offsets=None,
):
    radius = math.sqrt(target_area / math.pi)
    eps = interface_cells * solver.dx
    if grid_offsets is None:
        offsets = torch.zeros(1, 2, device=solver.device, dtype=solver.dtype)
    else:
        offsets = torch.as_tensor(
            grid_offsets, device=solver.device, dtype=solver.dtype
        ).reshape(-1, 2)
    x = solver.xx[None] - offsets[:, 0, None, None] * solver.dx
    y = solver.yy[None] - offsets[:, 1, None, None] * solver.dx
    radius_grid = torch.sqrt(x.square() + y.square())
    soft_mask = torch.sigmoid((radius - radius_grid) / eps)
    return (
        straight_through_solid_mask(soft_mask)
        if sharp_forward
        else soft_mask
    )


# =============================================================================
# Exact differentiable scalar first-Born RCS proxy
# =============================================================================


def monostatic_q(wavelengths, angles, device, dtype):
    wavelengths = torch.as_tensor(wavelengths, device=device, dtype=dtype).flatten()
    angles = torch.as_tensor(angles, device=device, dtype=dtype).flatten()
    k = 2.0 * math.pi / wavelengths[:, None]
    direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return (2.0 * k[..., None] * direction[None]).reshape(-1, 2)


def polygon_fourier_power(vertices, q):
    """Exact |integral_P exp(-i q.x) dx|^2 for a polygon."""
    nxt = torch.roll(vertices, -1, dims=-2)
    edge = nxt - vertices
    midpoint = 0.5 * (vertices + nxt)

    qe = torch.einsum("qd,bpd->bqp", q, edge)
    qm = torch.einsum("qd,bpd->bqp", q, midpoint)
    cross = (
        q[None, :, 0, None] * edge[:, None, :, 1]
        - q[None, :, 1, None] * edge[:, None, :, 0]
    )

    weight = cross * torch.sinc(qe / (2.0 * math.pi))
    q2 = q.square().sum(dim=-1).clamp_min(1e-20)[None]
    real = (weight * torch.sin(qm)).sum(dim=-1) / q2
    imag = (weight * torch.cos(qm)).sum(dim=-1) / q2
    return real.square() + imag.square()


# =============================================================================
# CUDA MRT D2Q9 external-flow solver
# =============================================================================


class MRTExternalFlow:
    """
    Low-Mach D2Q9 MRT lattice-Boltzmann solver.

    Physical domain is expressed relative to the equal-area diameter D:
      x in [-3D, 6D], y in [-3D, 3D].

    Boundary conditions:
      * left, top, bottom: free-stream non-equilibrium extrapolation,
      * right: prescribed-density non-equilibrium extrapolation,
      * outlet/far-field sponge,
      * differentiable link-wise halfway bounce-back at the body.

    Drag uses the momentum-exchange method, consistent with bounce-back.
    """

    C_LIST = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ]
    OPPOSITE_LIST = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    def __init__(
        self,
        cells_per_diameter=96,
        reynolds=40.0,
        inflow_speed=1.0 / 30.0,
        reference_area=0.205,
        domain_x_d=(-4.0, 10.0),
        domain_y_d=(-5.0, 5.0),
        device="cuda",
        dtype=torch.float32,
        compile_step=False,
        compile_mode="default",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.cells_per_diameter = int(cells_per_diameter)
        self.reynolds = float(reynolds)
        self.inflow_speed = float(inflow_speed)
        self.reference_area = float(reference_area)
        self.equivalent_diameter = math.sqrt(4.0 * reference_area / math.pi)
        self.domain_x_d = tuple(float(value) for value in domain_x_d)
        self.domain_y_d = tuple(float(value) for value in domain_y_d)

        if not (
            self.domain_x_d[0] < 0.0 < self.domain_x_d[1]
            and self.domain_y_d[0] < 0.0 < self.domain_y_d[1]
        ):
            raise ValueError("The external-flow domain must contain the origin.")

        # Integer multiples of D guarantee exactly square lattice cells.
        self.nx = round(
            (self.domain_x_d[1] - self.domain_x_d[0])
            * self.cells_per_diameter
        )
        self.ny = round(
            (self.domain_y_d[1] - self.domain_y_d[0])
            * self.cells_per_diameter
        )
        self.dx = self.equivalent_diameter / self.cells_per_diameter
        self.xmin = self.domain_x_d[0] * self.equivalent_diameter
        self.xmax = self.domain_x_d[1] * self.equivalent_diameter
        self.ymin = self.domain_y_d[0] * self.equivalent_diameter
        self.ymax = self.domain_y_d[1] * self.equivalent_diameter
        self.boundary_scheme = "sharp_ste_halfway_v1"

        self.cs2 = 1.0 / 3.0
        self.cs = 1.0 / math.sqrt(3.0)
        self.nu_lattice = (
            self.inflow_speed * self.cells_per_diameter / self.reynolds
        )
        self.tau_shear = 0.5 + 3.0 * self.nu_lattice
        self.s_nu = 1.0 / self.tau_shear

        if not (0.0 < self.s_nu < 2.0):
            raise ValueError(
                f"Invalid MRT shear relaxation s_nu={self.s_nu:.5f}. "
                "Change Re, U, or cells_per_diameter."
            )

        self.c = torch.tensor(self.C_LIST, device=self.device, dtype=self.dtype)
        self.w = torch.tensor(
            [
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        # d'Humieres/Lallemand-Luo moment basis.
        M = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 1, -1],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.M = M
        self.M_inv = torch.linalg.inv(M)

        # Conserved density/jx/jy modes have zero relaxation.
        self.relaxation = torch.tensor(
            [0.0, 1.64, 1.54, 0.0, 1.90, 0.0, 1.90, self.s_nu, self.s_nu],
            device=self.device,
            dtype=self.dtype,
        )[None, :, None, None]

        x = torch.linspace(
            self.xmin + 0.5 * self.dx,
            self.xmax - 0.5 * self.dx,
            self.nx,
            device=self.device,
            dtype=self.dtype,
        )
        y = torch.linspace(
            self.ymin + 0.5 * self.dx,
            self.ymax - 0.5 * self.dx,
            self.ny,
            device=self.device,
            dtype=self.dtype,
        )
        self.yy, self.xx = torch.meshgrid(y, x, indexing="ij")

        D = self.equivalent_diameter
        outlet = torch.clamp(
            (self.xx - (self.xmax - 2.0 * D)) / (2.0 * D), 0.0, 1.0
        ).square()
        top = torch.clamp(
            (self.yy - (self.ymax - 1.5 * D)) / (1.5 * D), 0.0, 1.0
        ).square()
        bottom = torch.clamp(
            ((self.ymin + 1.5 * D) - self.yy) / (1.5 * D), 0.0, 1.0
        ).square()
        inlet = torch.clamp(
            ((self.xmin + 1.0 * D) - self.xx) / D, 0.0, 1.0
        ).square()
        sponge_profile = torch.maximum(
            torch.maximum(outlet, top), torch.maximum(bottom, inlet)
        )
        self.sponge = (0.10 * sponge_profile)[None, None]

        rho = torch.ones(1, self.ny, self.nx, device=self.device, dtype=self.dtype)
        velocity = torch.zeros(1, 2, self.ny, self.nx, device=self.device, dtype=self.dtype)
        velocity[:, 0] = self.inflow_speed
        self.freestream_equilibrium = self.equilibrium(rho, velocity)

        rho_solid = torch.ones_like(rho)
        velocity_solid = torch.zeros_like(velocity)
        self.solid_equilibrium = self.equilibrium(rho_solid, velocity_solid)

        if compile_step:
            self.step = torch.compile(self._step_impl, mode=compile_mode)
        else:
            self.step = self._step_impl

    def equilibrium(self, rho, velocity):
        cu = torch.einsum("ic,bcyx->biyx", self.c, velocity)
        speed2 = velocity.square().sum(dim=1, keepdim=True)
        return self.w[None, :, None, None] * rho[:, None] * (
            1.0 + 3.0 * cu + 4.5 * cu.square() - 1.5 * speed2
        )

    def macros(self, populations, mask=None):
        rho = populations.sum(dim=1).clamp_min(1e-7)
        momentum = torch.einsum("ic,biyx->bcyx", self.c, populations)
        velocity = momentum / rho[:, None]
        if mask is not None:
            velocity = velocity * (1.0 - mask[:, None])
        return rho, velocity

    def initial_state(self, batch=1):
        return self.freestream_equilibrium.expand(batch, -1, -1, -1).clone()

    def collide(self, populations):
        rho, velocity = self.macros(populations)
        equilibrium = self.equilibrium(rho, velocity)
        moments = torch.einsum("ij,bjyx->biyx", self.M, populations)
        moments_eq = torch.einsum("ij,bjyx->biyx", self.M, equilibrium)
        moments_post = moments - self.relaxation * (moments - moments_eq)
        return torch.einsum("ij,bjyx->biyx", self.M_inv, moments_post)

    def stream_and_bounce(self, post_collision, mask):
        """
        Link-wise halfway bounce-back on the sharp forward solid.

        ``mask`` is binary in the forward pass, so fluid mass is not lost in a
        diffuse interface. Its straight-through derivative still supplies a
        geometry gradient. Momentum exchange gives the force on the body.
        """
        batch = post_collision.shape[0]
        streamed = torch.zeros_like(post_collision)
        body_force = torch.zeros(batch, 2, device=self.device, dtype=self.dtype)
        fluid_source = 1.0 - mask

        for i, (cx, cy) in enumerate(self.C_LIST):
            fi = post_collision[:, i]
            destination_solid = torch.roll(mask, shifts=(-cy, -cx), dims=(-2, -1))

            transmitted_weight = fluid_source * (1.0 - destination_solid)
            reflected_weight = fluid_source * destination_solid

            transmitted = fi * transmitted_weight
            reflected = fi * reflected_weight

            streamed[:, i] = streamed[:, i] + torch.roll(
                transmitted, shifts=(cy, cx), dims=(-2, -1)
            )
            streamed[:, self.OPPOSITE_LIST[i]] = (
                streamed[:, self.OPPOSITE_LIST[i]] + reflected
            )

            if cx != 0:
                body_force[:, 0] = body_force[:, 0] + 2.0 * cx * reflected.sum(dim=(-2, -1))
            if cy != 0:
                body_force[:, 1] = body_force[:, 1] + 2.0 * cy * reflected.sum(dim=(-2, -1))

        return streamed, body_force

    def apply_outer_boundaries(self, populations):
        out = populations.clone()
        rho, velocity = self.macros(out)

        # Bottom free-stream NEE.
        rho_b = rho[:, 1, :]
        u_adj = velocity[:, :, 1, :]
        u_target = torch.zeros_like(u_adj)
        u_target[:, 0] = self.inflow_speed
        feq_target = self.equilibrium(rho_b[:, None, :], u_target[:, :, None, :])[:, :, 0, :]
        feq_adj = self.equilibrium(rho_b[:, None, :], u_adj[:, :, None, :])[:, :, 0, :]
        out[:, :, 0, :] = feq_target + out[:, :, 1, :] - feq_adj

        # Top free-stream NEE.
        rho_t = rho[:, -2, :]
        u_adj = velocity[:, :, -2, :]
        u_target = torch.zeros_like(u_adj)
        u_target[:, 0] = self.inflow_speed
        feq_target = self.equilibrium(rho_t[:, None, :], u_target[:, :, None, :])[:, :, 0, :]
        feq_adj = self.equilibrium(rho_t[:, None, :], u_adj[:, :, None, :])[:, :, 0, :]
        out[:, :, -1, :] = feq_target + out[:, :, -2, :] - feq_adj

        # Left free-stream NEE.
        rho_l = rho[:, :, 1]
        u_adj = velocity[:, :, :, 1]
        u_target = torch.zeros_like(u_adj)
        u_target[:, 0] = self.inflow_speed
        feq_target = self.equilibrium(rho_l[:, :, None], u_target[:, :, :, None])[:, :, :, 0]
        feq_adj = self.equilibrium(rho_l[:, :, None], u_adj[:, :, :, None])[:, :, :, 0]
        out[:, :, :, 0] = feq_target + out[:, :, :, 1] - feq_adj

        # Right pressure outlet NEE: rho=1, velocity extrapolated.
        rho_r = torch.ones_like(rho[:, :, -2])
        u_adj = velocity[:, :, :, -2]
        feq_target = self.equilibrium(rho_r[:, :, None], u_adj[:, :, :, None])[:, :, :, 0]
        rho_adj = rho[:, :, -2]
        feq_adj = self.equilibrium(rho_adj[:, :, None], u_adj[:, :, :, None])[:, :, :, 0]
        out[:, :, :, -1] = feq_target + out[:, :, :, -2] - feq_adj

        return out

    def _step_impl(self, populations, mask):
        post_collision = self.collide(populations)
        streamed, body_force = self.stream_and_bounce(post_collision, mask)
        streamed = self.apply_outer_boundaries(streamed)

        batch = streamed.shape[0]
        freestream = self.freestream_equilibrium.expand(batch, -1, -1, -1)
        streamed = (1.0 - self.sponge) * streamed + self.sponge * freestream

        # Reset only the deep solid core; leave the diffuse interface governed
        # by link-wise bounce-back.
        solid_core = mask.pow(8)[:, None]
        solid_eq = self.solid_equilibrium.expand(batch, -1, -1, -1)
        streamed = (1.0 - solid_core) * streamed + solid_core * solid_eq

        return streamed, body_force

    def drag_coefficient(self, body_force):
        # 2-D coefficient per unit span, D_lattice = cells_per_diameter.
        return 2.0 * body_force[:, 0] / (
            self.inflow_speed**2 * self.cells_per_diameter
        )

    def diagnostics(self, populations, mask):
        rho, velocity = self.macros(populations, mask)
        fluid = 1.0 - mask
        speed = torch.linalg.vector_norm(velocity, dim=1)
        max_mach = (speed / self.cs).amax(dim=(-2, -1))
        mean_rho = (rho * fluid).sum(dim=(-2, -1)) / fluid.sum(dim=(-2, -1)).clamp_min(1.0)
        mass_error = (mean_rho - 1.0).abs()
        return {
            "max_mach": max_mach,
            "mass_error": mass_error,
            "mean_rho": mean_rho,
        }

    def validate_diagnostics(
        self,
        diagnostics,
        max_mach_limit=0.15,
        mass_error_limit=1e-3,
    ):
        max_mach = diagnostics["max_mach"].max().item()
        mass_error = diagnostics["mass_error"].max().item()
        problems = []
        if max_mach > max_mach_limit:
            problems.append(
                f"max Mach {max_mach:.4f} exceeds {max_mach_limit:.4f}"
            )
        if mass_error > mass_error_limit:
            problems.append(
                f"density error {mass_error:.3e} exceeds {mass_error_limit:.3e}"
            )
        if problems:
            raise RuntimeError("Unphysical LBM state: " + "; ".join(problems))


# =============================================================================
# Steady-state settling and implicit differentiation
# =============================================================================


@torch.no_grad()
def settle_to_steady(
    solver,
    state,
    mask,
    min_steps=400,
    max_steps=3000,
    chunk_steps=100,
    drag_tolerance=2e-4,
    velocity_tolerance=2e-4,
    stable_chunks_required=3,
    report_every=0,
    label="flow",
):
    previous_cd = None
    stable_chunks = 0
    total_steps = 0
    last_drag_change = float("inf")
    last_velocity_change = float("inf")
    last_cd = None

    while total_steps < max_steps:
        start_state = state
        force_sum = torch.zeros(
            state.shape[0], 2, device=state.device, dtype=state.dtype
        )

        for _ in range(chunk_steps):
            state, force = solver.step(state, mask)
            force_sum = force_sum + force

        total_steps += chunk_steps
        cd = solver.drag_coefficient(force_sum / chunk_steps)

        _, u0 = solver.macros(start_state, mask)
        _, u1 = solver.macros(state, mask)
        fluid = (1.0 - mask)[:, None]
        numerator = torch.sqrt(torch.sum(((u1 - u0) * fluid).square(), dim=(1, 2, 3)))
        denominator = torch.sqrt(torch.sum((u1 * fluid).square(), dim=(1, 2, 3))).clamp_min(1e-12)
        velocity_change = numerator / denominator

        if previous_cd is None:
            drag_change = torch.full_like(cd, float("inf"))
        else:
            drag_change = (cd - previous_cd).abs() / cd.abs().clamp_min(1e-8)

        last_drag_change = drag_change.max().item()
        last_velocity_change = velocity_change.max().item()
        last_cd = cd

        if report_every and total_steps % int(report_every) < chunk_steps:
            print(
                f"{label}: steps={total_steps}, Cd={cd.mean().item():.6f}, "
                f"dCd={last_drag_change:.2e}, du={last_velocity_change:.2e}"
            )

        converged = (
            total_steps >= min_steps
            and last_drag_change < drag_tolerance
            and last_velocity_change < velocity_tolerance
        )

        if converged:
            stable_chunks += 1
        else:
            stable_chunks = 0

        if stable_chunks >= stable_chunks_required:
            break

        previous_cd = cd

    diag = solver.diagnostics(state, mask)
    return state, {
        "settle_steps": total_steps,
        "cd": last_cd.detach(),
        "drag_relative_change": last_drag_change,
        "velocity_relative_change": last_velocity_change,
        "max_mach": diag["max_mach"].detach(),
        "mass_error": diag["mass_error"].detach(),
        "settled": stable_chunks >= stable_chunks_required,
    }


def _advance_with_mean_force(solver, state, mask, steps):
    force_sum = torch.zeros(
        state.shape[0], 2, device=state.device, dtype=state.dtype
    )
    for _ in range(int(steps)):
        state, force = solver.step(state, mask)
        force_sum = force_sum + force
    return state, force_sum / float(steps)


def _tensor_inner(left, right):
    return torch.sum(
        left.reshape(-1) * right.reshape(-1), dtype=torch.float64
    )


def _tensor_norm(value):
    return torch.sqrt(_tensor_inner(value, value).clamp_min(0.0))


def _gmres(
    matvec,
    rhs,
    restart=8,
    max_iterations=24,
    relative_tolerance=2e-3,
    absolute_tolerance=1e-7,
):
    """Restarted matrix-free GMRES for a single tensor-shaped unknown."""
    rhs = rhs.detach()
    rhs_norm = _tensor_norm(rhs)
    threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * rhs_norm.item(),
    )
    if rhs_norm.item() <= threshold:
        return torch.zeros_like(rhs), {
            "iterations": 0,
            "residual_norm": rhs_norm.item(),
            "relative_residual": 0.0,
            "converged": True,
        }

    solution = torch.zeros_like(rhs)
    iterations = 0
    residual = rhs

    while iterations < int(max_iterations):
        if iterations:
            residual = rhs - matvec(solution)
        beta = _tensor_norm(residual)
        if beta.item() <= threshold:
            break

        basis = [residual / beta.to(residual.dtype)]
        cycle_steps = min(int(restart), int(max_iterations) - iterations)
        hessenberg = torch.zeros(
            cycle_steps + 1,
            cycle_steps,
            device=rhs.device,
            dtype=torch.float64,
        )
        projected_rhs = torch.zeros(
            cycle_steps + 1, device=rhs.device, dtype=torch.float64
        )
        projected_rhs[0] = beta
        coefficients = None

        for column in range(cycle_steps):
            candidate = matvec(basis[column]).detach()
            # Two-pass modified Gram-Schmidt is materially more stable for the
            # highly non-normal LBM adjoint than a single orthogonalization.
            for _ in range(2):
                for row, vector in enumerate(basis):
                    coefficient = _tensor_inner(candidate, vector)
                    hessenberg[row, column] += coefficient
                    candidate = (
                        candidate
                        - coefficient.to(candidate.dtype) * vector
                    )

            next_norm = _tensor_norm(candidate)
            hessenberg[column + 1, column] = next_norm
            if next_norm.item() > 1e-14:
                basis.append(candidate / next_norm.to(candidate.dtype))

            active_h = hessenberg[: column + 2, : column + 1]
            active_rhs = projected_rhs[: column + 2]
            coefficients = torch.linalg.lstsq(
                active_h, active_rhs
            ).solution
            projected_residual = active_rhs - active_h @ coefficients
            iterations += 1

            if _tensor_norm(projected_residual).item() <= threshold:
                break
            if next_norm.item() <= 1e-14:
                break

        if coefficients is None:
            break

        update = torch.zeros_like(solution)
        for coefficient, vector in zip(coefficients, basis):
            update = update + coefficient.to(update.dtype) * vector
        solution = solution + update

        residual = rhs - matvec(solution)
        if _tensor_norm(residual).item() <= threshold:
            break

    residual = rhs - matvec(solution)
    residual_norm = _tensor_norm(residual).item()
    relative_residual = residual_norm / max(rhs_norm.item(), 1e-30)
    return solution, {
        "iterations": iterations,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "converged": residual_norm <= threshold,
    }


class _ImplicitSteadyDrag(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state,
        mask,
        solver,
        fixed_point_steps,
        restart,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    ):
        ctx.solver = solver
        ctx.fixed_point_steps = int(fixed_point_steps)
        ctx.restart = int(restart)
        ctx.max_iterations = int(max_iterations)
        ctx.relative_tolerance = float(relative_tolerance)
        ctx.absolute_tolerance = float(absolute_tolerance)
        ctx.save_for_backward(state.detach(), mask.detach())

        _, mean_force = _advance_with_mean_force(
            solver, state, mask, ctx.fixed_point_steps
        )
        return solver.drag_coefficient(mean_force)

    @staticmethod
    def backward(ctx, drag_gradient):
        state, mask = ctx.saved_tensors
        solver = ctx.solver
        fixed_point_steps = ctx.fixed_point_steps

        def state_vjp(vector):
            with torch.enable_grad():
                state_input = state.detach().requires_grad_(True)
                next_state, _ = _advance_with_mean_force(
                    solver,
                    state_input,
                    mask,
                    fixed_point_steps,
                )
                return torch.autograd.grad(
                    next_state,
                    state_input,
                    grad_outputs=vector,
                    retain_graph=False,
                    create_graph=False,
                )[0].detach()

        with torch.enable_grad():
            state_input = state.detach().requires_grad_(True)
            _, mean_force = _advance_with_mean_force(
                solver, state_input, mask, fixed_point_steps
            )
            drag = solver.drag_coefficient(mean_force)
            weighted_drag = torch.sum(drag * drag_gradient)
            drag_state_gradient = torch.autograd.grad(
                weighted_drag,
                state_input,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

        def adjoint_operator(vector):
            return vector - state_vjp(vector)

        adjoint, info = _gmres(
            adjoint_operator,
            drag_state_gradient,
            restart=ctx.restart,
            max_iterations=ctx.max_iterations,
            relative_tolerance=ctx.relative_tolerance,
            absolute_tolerance=ctx.absolute_tolerance,
        )
        solver.last_adjoint_info = info
        if not info["converged"]:
            raise RuntimeError(
                "Steady adjoint GMRES did not converge: "
                f"relative residual={info['relative_residual']:.3e} after "
                f"{info['iterations']} iterations"
            )

        with torch.enable_grad():
            mask_input = mask.detach().requires_grad_(True)
            next_state, mean_force = _advance_with_mean_force(
                solver,
                state,
                mask_input,
                fixed_point_steps,
            )
            drag = solver.drag_coefficient(mean_force)
            augmented_objective = (
                torch.sum(drag * drag_gradient)
                + _tensor_inner(next_state, adjoint).to(drag.dtype)
            )
            mask_gradient = torch.autograd.grad(
                augmented_objective,
                mask_input,
                retain_graph=False,
                create_graph=False,
            )[0]

        return (
            None,
            mask_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def implicit_steady_drag(
    solver,
    steady_state,
    mask,
    fixed_point_steps=8,
    restart=64,
    max_iterations=224,
    relative_tolerance=7e-3,
    absolute_tolerance=1e-7,
    primal_relative_tolerance=5e-3,
):
    """Drag with an implicit gradient of the converged LBM fixed point."""
    with torch.no_grad():
        advanced_state, _ = _advance_with_mean_force(
            solver, steady_state, mask, fixed_point_steps
        )
        primal_residual = _tensor_norm(advanced_state - steady_state).item()
        state_norm = _tensor_norm(steady_state).item()
        primal_relative_residual = primal_residual / max(state_norm, 1e-30)
    solver.last_primal_fixed_point_residual = primal_relative_residual
    if primal_relative_residual > float(primal_relative_tolerance):
        raise RuntimeError(
            "Implicit drag requires a converged primal state: "
            f"relative fixed-point residual={primal_relative_residual:.3e} "
            f"exceeds {float(primal_relative_tolerance):.3e}"
        )

    return _ImplicitSteadyDrag.apply(
        steady_state.detach(),
        mask,
        solver,
        fixed_point_steps,
        restart,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )


@torch.no_grad()
def average_steady_drag(solver, state, mask, windows=8, steps_per_window=100):
    values = []
    for _ in range(windows):
        force_sum = torch.zeros(
            state.shape[0], 2, device=state.device, dtype=state.dtype
        )
        for _ in range(steps_per_window):
            state, force = solver.step(state, mask)
            force_sum = force_sum + force
        values.append(solver.drag_coefficient(force_sum / steps_per_window))

    values = torch.stack(values, dim=0)
    return state, values.mean(dim=0), values.std(dim=0, unbiased=False)


# =============================================================================
# References, checkpoints, logging
# =============================================================================


def disk_reference(solver, cfg, outdir):
    x_tag = "_".join(f"{value:g}" for value in solver.domain_x_d)
    y_tag = "_".join(f"{value:g}" for value in solver.domain_y_d)
    offsets = cfg.get("grid_offsets") or [(0.0, 0.0)]
    offset_tag = "_".join(
        f"{float(x):g}-{float(y):g}" for x, y in offsets
    )
    cache_name = (
        f"disk_{solver.boundary_scheme}_cpd{solver.cells_per_diameter}_"
        f"Re{solver.reynolds:g}_U{solver.inflow_speed:g}_"
        f"x{x_tag}_y{y_tag}_offsets{offset_tag}.pt"
    )
    cache_path = outdir / cache_name

    if cache_path.exists() and not cfg["ignore_disk_cache"]:
        cached = torch.load(cache_path, map_location=solver.device)
        return cached["state"], cached["cd_mean"], cached["cd_std"]

    mask = circle_mask(
        solver,
        target_area=cfg["target_area"],
        interface_cells=cfg["interface_cells"],
        grid_offsets=cfg.get("grid_offsets"),
    )
    state = solver.initial_state(mask.shape[0])
    state, settle_diag = settle_to_steady(
        solver,
        state,
        mask,
        min_steps=cfg["settle_min_steps"],
        max_steps=cfg["reference_settle_max_steps"],
        chunk_steps=cfg["settle_chunk_steps"],
        drag_tolerance=cfg["reference_drag_tolerance"],
        velocity_tolerance=cfg["reference_velocity_tolerance"],
    )
    solver.validate_diagnostics(
        settle_diag,
        max_mach_limit=cfg.get("max_mach", 0.15),
        mass_error_limit=cfg.get("max_mass_error", 1e-3),
    )
    if cfg.get("require_settled", True) and not settle_diag["settled"]:
        raise RuntimeError(
            "Disk reference did not reach steady state within "
            f"{settle_diag['settle_steps']} steps."
        )
    state, cd_mean, cd_std = average_steady_drag(
        solver,
        state,
        mask,
        windows=cfg["average_windows"],
        steps_per_window=cfg["average_window_steps"],
    )
    # Combine temporal and sub-cell alignment variation into one reference.
    cd_offset_mean = cd_mean.mean().reshape(1)
    cd_combined_std = torch.sqrt(
        (cd_std.square() + (cd_mean - cd_offset_mean).square()).mean()
    ).reshape(1)
    cd_mean = cd_offset_mean
    cd_std = cd_combined_std

    torch.save(
        {
            "state": state.detach(),
            "cd_mean": cd_mean.detach(),
            "cd_std": cd_std.detach(),
            "settle_diag": settle_diag,
        },
        cache_path,
    )
    return state, cd_mean, cd_std


def save_checkpoint(path, params, optimizer, flow_state, history, cfg, step):
    torch.save(
        {
            "params": params.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "flow_state": flow_state.detach().cpu(),
            "history": history,
            "config": cfg,
            "step": step,
        },
        path,
    )


def append_history_csv(path, record):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def initial_parameters(device, dtype, modes_count, chord=0.70):
    chord_min = 0.52
    chord_max = 0.96
    fraction = (chord - chord_min) / (chord_max - chord_min)
    chord_logit = math.log(fraction / (1.0 - fraction))
    params = torch.zeros(1, 1 + modes_count, device=device, dtype=dtype)
    params[:, 0] = chord_logit
    params[:, 1:] = 1e-3 * torch.randn(
        1, modes_count, device=device, dtype=dtype
    )
    return params


# =============================================================================
# Grid-convergence evaluation
# =============================================================================


def evaluate_grid(cfg):
    device = torch.device(cfg["device"])
    checkpoint_data = torch.load(cfg["checkpoint"], map_location=device)
    params = checkpoint_data["params"].to(device)

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "grid_convergence.csv"

    rows = []
    for cpd in cfg["grid_cells_per_diameter"]:
        solver = MRTExternalFlow(
            cells_per_diameter=cpd,
            reynolds=cfg["reynolds"],
            inflow_speed=cfg["inflow_speed"],
            reference_area=cfg["target_area"],
            domain_x_d=cfg.get("domain_x_d", (-4.0, 10.0)),
            domain_y_d=cfg.get("domain_y_d", (-5.0, 5.0)),
            device=device,
            dtype=torch.float32,
            compile_step=cfg["compile_step"],
            compile_mode=cfg.get(
                "compile_mode", "default"
            ),
        )

        mask, vertices, chord, _ = profile_mask(
            params,
            solver,
            target_area=cfg["target_area"],
            interface_cells=cfg["interface_cells"],
            grid_offsets=cfg.get("grid_offsets"),
        )
        state = solver.initial_state(mask.shape[0])
        state, settle_diag = settle_to_steady(
            solver,
            state,
            mask,
            min_steps=cfg["settle_min_steps"],
            max_steps=cfg["grid_settle_max_steps"],
            chunk_steps=cfg["settle_chunk_steps"],
            drag_tolerance=cfg["grid_drag_tolerance"],
            velocity_tolerance=cfg["grid_velocity_tolerance"],
        )
        state, cd_mean, cd_std = average_steady_drag(
            solver,
            state,
            mask,
            windows=cfg["average_windows"],
            steps_per_window=cfg["average_window_steps"],
        )
        cd_offset_mean = cd_mean.mean()
        cd_combined_std = torch.sqrt(
            (cd_std.square() + (cd_mean - cd_offset_mean).square()).mean()
        )

        disk_state, disk_cd, disk_cd_std = disk_reference(solver, cfg, outdir)

        row = {
            "cells_per_diameter": cpd,
            "nx": solver.nx,
            "ny": solver.ny,
            "tau_shear": solver.tau_shear,
            "cd": float(cd_offset_mean.cpu()),
            "cd_std": float(cd_combined_std.cpu()),
            "disk_cd": float(disk_cd.cpu()),
            "disk_cd_std": float(disk_cd_std.cpu()),
            "relative_cd": float((cd_offset_mean / disk_cd).cpu()),
            "area": float(polygon_area(vertices).mean().cpu()),
            "chord": float(chord.mean().cpu()),
            "settle_steps": settle_diag["settle_steps"],
            "settled": bool(settle_diag["settled"]),
            "drag_relative_change": settle_diag["drag_relative_change"],
            "velocity_relative_change": settle_diag["velocity_relative_change"],
            "max_mach": float(settle_diag["max_mach"].max().cpu()),
            "mass_error": float(settle_diag["mass_error"].max().cpu()),
        }
        rows.append(row)
        print(row)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) >= 2:
        print("successive relative Cd changes:")
        for coarse, fine in zip(rows[:-1], rows[1:]):
            change = abs(fine["cd"] - coarse["cd"]) / abs(fine["cd"])
            print(
                f"  D/dx {coarse['cells_per_diameter']} -> "
                f"{fine['cells_per_diameter']}: {100.0 * change:.4f}%"
            )
