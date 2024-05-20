[x] In outer approximation (and GBD) we should add the infeasible cut when the NLP is infeasible!! \
[x] Do we still need to enforce only the constraints as `-inf <= g(x) <= lbg` ? Or can they be double sided? \
[x] Need a double flag for s-b-miqp: one for computing the LB from the relaxed solution, one for warmstarting with the relaxed solution.
-  add a setting `s.warm_start_relaxed`

[x] warm start as a function
[x] What should we do with all the derivation of the sequential benders algorithm? \
[x] How can we pass the hyperparams to MinlpSolver -> GenericDecomposition -> ..., for instance: `first_relaxed` for the decomposition. \
[x] check reset function \
[x] implementation of `+` for warm starting \
[x] refactoring pumps \
[x] refactoring bonmin \
[x] refactoring cia \
[x] add plain relaxed nlp solver \
[x] how to save stats? \
[x] check stats \
[x] adapt tests \
[x] add example file \
[x] external/ampl it works? --> added disclaimer \
[x] delete old files \
[x] add license header everywhere \
[x] add algorithm from Alberto \
[ ] update readme \
[ ] remove file `benders_lbqp_master.py` from the public version
[ ] create a branch `public` for tracking the public GitHub version, keep the current syscop_gitlab main for developing.

----> Publish v0.0.1 \
