# Bringing the arm toward center during swing-up

All six previously unsuccessful starting states achieved a one-second upright hold with the revised momentum target. They completed the required hold between 5.82 and 18.03 s after the start of the simulation, with a median of 12.12 s. All six completed 40 seconds and still met the holding conditions at the end. The largest recorded arm angle magnitude was 134.09°, below the 135° limit.

The previous controller asked each swing-up pulse to leave zero angular momentum about the arm axis. In the nominal model, zero momentum stops the arm from drifting on average over a swing. It can leave the arm far from center. The revised controller requests momentum in the direction of center, proportional to the arm's displacement from center.

$$
p_{\mathrm{target}}=-J_b\lambda\theta .
$$

Here $\theta$ is the signed arm angle from center in radians, $J_b$ is the model's arm-axis inertia with the pendulum hanging down, and $\lambda=1\ \mathrm{s}^{-1}$ is the centering rate chosen for this test. The target is in kg·m²/s. The pulse selection uses this target in place of zero. The same torque limits, arm angle checks, and conditions for starting upright feedback apply. Only the six previously unsuccessful cases were rerun, so the result describes this subset.

The first near-upright time is the first recorded instant within 5° of upright, including the initial state. Four runs already started within 5°. They first needed to move the arm inward before a later approach. Upright feedback could start once the arm was within 85.9° of center, the pendulum was within 11.5° of upright, both angular speeds were below their entry limits of 2 and 3 rad/s, and the predicted response met the existing checks.

| Starting state ID | Arm angle magnitude at first near-upright time, previous → revised | Arm angle magnitude when upright feedback began | Time to one-second upright hold |
| ---: | ---: | ---: | ---: |
| 0 | 103.9° → 30.7° | 34.8° | 15.47 s |
| 3 | 114.1° → 84.4° | 85.4° | 5.82 s |
| 15 | 113.5° → 113.5° | 84.2° | 15.62 s |
| 19 | 112.8° → 112.8° | 50.8° | 8.76 s |
| 23 | 103.1° → 103.1° | 68.4° | 6.94 s |
| 31 | 106.9° → 106.9° | 83.4° | 18.03 s |

The [first-10-second torque plot](../../human-readables/actuation-first-10s.pdf) uses the three reference runs from the preceding report. Shading marks the time the controller was commanding a pulse, braking, or applying upright feedback, including any zero torque during that time. Adjacent active commands count as one interval. Each interval contributes one switch-on and its duration within the plotted 10 seconds.

[Paired arm measurements](../nominal/machine-scannables/paired-results.csv) · [Six-run settings](../nominal/machine-scannables/design.json) · [First-10-second counts](../../machine-scannables/actuation-first-10s.csv)
