[x] In outer approximation (and GBD) we should add the infeasible cut when the NLP is infeasible!! \
[x] Do we still need to enforce only the constraints as `-inf <= g(x) <= lbg` ? Or can they be double sided? \
[ ] Need a double flag for s-b-miqp: one for computing the LB from the relaxed solution, one for warmstarting with the relaxed solution. \
    - add a setting `s.warm_start_relaxed`
[ ] warm start as a function
[x] What should we do with all the derivation of the sequential benders algorithm? \
[x] How can we pass the hyperparams to MinlpSolver -> GenericDecomposition -> ..., for instance: `first_relaxed` for the decomposition. \
[ ] check reset function
[ ] implementation of `+` for warm starting
[ ] refactoring pumps
[ ] refactoring bonmin
[ ] refactoring cia
----> Publish v0.0.1
[ ] add algorithm form Alberto
