# Execution Plan Policy

Every multi-milestone change must have a living plan in `.agent/execplans/`.
The plan is part of the change and must remain useful to a contributor who has
only the repository checkout.

An ExecPlan must record:

- the complete goal, scope, and non-goals;
- frozen contracts and decisions, with links to ADRs;
- milestone status and the next concrete action;
- discoveries that change implementation assumptions;
- exact commands and summarized validation results;
- failures, their evidence, and the chosen recovery;
- unrun matrix entries without presenting them as passes;
- external or irreversible actions that still require authorization.

Update the plan after each milestone and whenever evidence invalidates a prior
assumption. Never rewrite a failed command into a successful-looking history.
Keep commands reproducible from the repository root, and identify environment
variables that affect numerical or backend behavior.

An ExecPlan is complete only when its requirement-by-requirement audit points
to current evidence for every release gate. A green narrow test is not evidence
for a broader compatibility claim.
