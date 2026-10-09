# Reading the Section III.A plan

Read outline.md for the proposed sequence: mechanical derivation, energy representation, Jacobian, and feasibility. A short Section III introduction explains why energy-based work selection still requires prediction from the physical state. The plan contains one introduction and four subsection paragraphs, combining the two mechanical topics into one derivation.

The key mathematical distinction is that the energy map describes physical energy but does not replace the full state. Its Jacobian converts physical-state rates into energy rates. It neither makes the map invertible nor establishes an autonomous energy-state model. Physically realizable energy vectors also need not be reachable from a given state over a prescribed actuation interval.

I checked the power identity, explicit Jacobian, and completed-square expression symbolically against the current nominal model. The model configuration specifies zero damping. The supporting derivation in machine-scannables/mathematical_checks.md records the exact identities and assumptions. No experiment was rerun and no controller performance claim was added.

The Section III introduction and III.A in main.tex now follow this structure. The text contains four subsection paragraphs and seven equation groups. The Jacobian is displayed directly as a three-by-four matrix, with rows corresponding to energies and columns to physical-state components. The hinge inertia and its contributions, the cross-acceleration terms, the instantaneous vector field, and the constant-input state-transition map are defined explicitly. The duplicate energy-definition label has been removed. The edited passage uses standard control terminology and distinguishes the energy representation from a closed state model.

Compilation succeeded and the rendered equations were inspected. The existing unresolved fig:method reference remains in III.B. A source comparison confirmed that Sections I-II and III.B onward are unchanged. No Chinese files were created or modified.
