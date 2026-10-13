# Controller figures

The three figures use 9 pt DejaVu Sans, charcoal, muted blue, sage, and one orange accent. They contain solid curves and references, with no decorative dotted lines. PDF and SVG preserve vector geometry and editable text; PNG files are 300 dpi previews.

- [Rotary-pendulum configuration](human-readables/01-rotary-pendulum.pdf): two separate views show two arm angles in the horizontal plane and two pendulum angles in the vertical swing plane. The body legend identifies the charcoal arm and blue pendulum; faded rods show alternative poses. The right-angle mark identifies the arm-normal swing plane, which rotates and translates with the arm and is therefore not an inertial reference plane. Constants and dimensions are omitted.
- [Energy request and physical compatibility](human-readables/02-energy-plane.pdf): the pale plane fixes requested total energy; its blue region contains the mechanically realizable energy distributions. The legend defines the requested energy plane and realizable region. A single 3D view shows the requested plane, its curved physical subset, and the upright-rest target. The current state is omitted to keep attention on the plane structure. The curved boundary matters: nonnegative energy alone is insufficient.
- [Torque response and solutions](human-readables/03-torque-solutions.pdf): the left panel shows endpoints of independent held-torque predictions; the right panel exposes the work-matching zeros. Open circles are approximate work matches; the star is the controller's selected action.

Figures 2 and 3 use the same recorded state and the same requested plane. Figure 2 is a single 3.4-inch-column view emphasizing the plane structure and upright target. Figure 3 retains its 7-inch two-panel layout for the complete torque sweep and work-matching curve. Both share the same 9 pt typography, palette, and quiet coordinate backgrounds.

## Selected example and interpretation

All 715 complete decisions in the eight preserved episodes were compared. Of these, 37 met the geometric and numerical illustration criteria. The three highest-ranked candidates were visually compared. The Figure 3 example is the downward-start replay with the longest capture time in its source seed, at 12.1 s, decision 121, lane 177, seed identifier 20261022. It is selected to explain the method, not to estimate typical performance. The full original raw batches are absent locally; the verified portable records are the available source.

The matching torques are approximately −11.366 and +0.231 mN m. The recorded and reproduced choice is +3.4425 mN m. The selected action differs from both matches because the controller jointly scores endpoint potential energy, work mismatch, and arm excursion.

Figure 2 uses the same 12.1 s record as Figure 3. Its requested plane has total energy 0.783767 times the upright reference. At this energy the compatibility boundary is visibly curved, while the upright-rest target is clearly separated from the plane. The figure has one equal-scale coordinate frame spanning zero to 1.05, one target marker, and a compact two-item legend. There is no current-state marker, inset, or separate overview. Source values and geometry are saved in the [plane example](machine-scannables/plane_example.json). Physical compatibility remains distinct from finite-duration reachability.

## Caption material

**Figure 1. Rotary-pendulum rotations.** (a) Two illustrative arm configurations have angles $\theta_1$ and $\theta_2$, measured from the same fixed horizontal reference. Motor torque $u$ and signed arm velocity $\omega=\dot\theta$ act toward increasing arm angle. (b) Two illustrative pendulum configurations have signed angles $\alpha_1<0$ and $\alpha_2>0$, measured from the downward vertical. Signed pendulum velocity $\nu=\dot\alpha$ is positive in the indicated direction, and $g$ denotes downward gravitational acceleration. Subscripts identify alternative configurations of the same mechanism. The pale planes are the horizontal arm plane and the arm-attached vertical swing plane. The right-angle mark indicates that the arm is normal to the swing plane. As the arm angle changes, the hinge translates and the swing plane turns; this is a moving, non-inertial reference. Charcoal identifies the arm and blue the pendulum, with faded rods denoting alternative configurations. The poses are chosen for illustration and are not sampled states or simultaneous rods. No length, mass, or inertia annotations are shown.

**Figure 2. Requested energy plane and physical compatibility.** The pale requested plane and its blue realizable subset are shown relative to the upright-rest target. The curved blue edge highlights the mechanical compatibility constraint. All three axes have equal linear scales. The energy coordinates are arm-body kinetic energy $K_a$, full pendulum-body kinetic energy $K_p$, and downward-referenced potential energy $V$, normalized by the derived upright-rest energy $E_\star=m_pgl$. The marked state $e_\star=(0,0,E_\star)$ denotes upright rest. The requested signed work $w_k$ is computed by the existing clipped proportional energy policy, and $E_k$ is the sum of the current energy components. For endpoint energy coordinates $z_a,z_p,z_V$, the requested plane obeys

$$
z_a+z_p+z_V=E_k+w_k.
$$

The blue region is the part of this plane satisfying the mechanical energy-compatibility inequality. Its curved edge is the minimum pendulum kinetic energy compatible with arm motion and elevation. The region asserts existence of a physical state with unrestricted velocities, not finite-duration reachability or trajectory containment.

**Figure 3. Held-torque endpoints and work matching.** With the recorded physical state fixed, each bounded torque $u$ is held for 0.1 s. The blue curve comprises predicted endpoint energies, rather than successive points along one trajectory. The plane region has the same meaning as in Figure 2. The usable bound is $U=0.01836$ N m. Open circles identify approximate work-matching torques $u_1,u_2$; the star marks the selected torque $u^\star$. Work error is predicted shaft work minus the recorded request, with shaft work computed from held torque times unwrapped arm displacement. All predictions use five 20 ms fourth-order Runge–Kutta steps. The selected action jointly minimizes the implemented potential, work, and sampled arm-excursion penalties over grid and refined candidates; it need not match the request. The arm envelope remains a soft penalty.

## Verification and reproduction

The replay hashes and current physical/controller inputs were checked. The independently reproduced selected endpoint agrees with the recorded prediction within $2.1\times10^{-17}$ J. The maximum energy-balance residual over the displayed sweep is approximately $1.48\times10^{-7}$ J. The physical boundary was reconstructed and re-encoded within $1.1\times10^{-17}$ J. Numerical residuals are kept in the [example audit](machine-scannables/example.json).

The three panels were visually inspected after label and viewpoint revisions. Source preflight has no failures; its raster warnings are accepted because the publication assets are vector and the PNGs are previews. Its uncertainty warning is inapplicable: these are deterministic predictions at one recorded state, not seed averages. The PDF glyph-size check found a minimum of 6.3 pt for mathematical subscripts, with all base text at 9 pt. The source files pass Ruff.

Run from the repository root using the existing virtual environment:

```bash
MPLCONFIGDIR=/tmp/burst-coast-matplotlib PYTHONPATH=src \
  .venv/bin/python -m rotary_pendulum.visualization.controller_figures.render
```

The four source files, containing eight functions, reside in `src/rotary_pendulum/visualization/controller_figures/`. The entry point recomputes the diagnostic figures from the preserved records and writes source tables, hashes, numerical audits, and the nine figure files. It does not rerun a performance campaign or modify the manuscript. The NumPy dynamics are the explicit figure-prediction backend; no JAX installation or GPU is needed.
