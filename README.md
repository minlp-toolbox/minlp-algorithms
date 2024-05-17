# Usage

Set up and activate a fresh Python virtual environment (Python >= 3.7 should work)

From the root of the repository
```
pip3 install -e .
python3 minlp_algorithms <mode> <solver> <problem>
```

More info by running
```
python3 minlp_algorithms -h
```



You can enable or change options using environmental variables:
| Environmental variable |     Value    | Description                 |
| ---------------------- | ------------ | ----------------------------|
|         DEBUG          |  True/False  | Toggle debugging output     |
|         LOG_DATA       |  True/False  | Toggle saving statistics     |
|        MIP_SOLVER      | gurobi/highs | Configure MIP solver        |



## Available MINLP solvers/algorithms

**New?**: the algorithm is novel and created by the authors of this software.\
**CVX guarantee?**: the algorithm converge to the global optimum when a *convex* MINLP is given.

| Solvers | Description                                                  | New?                                              | CVX guarantee? |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| gbd        | Generalized Benders Decomposition ([Geoffrion, 1972](https://www.researchgate.net/profile/Arthur-Geoffrion/publication/230872895_Generalized_Benders_Decomposition/links/554a25f20cf29752ee7b8013/Generalized-Benders-Decomposition.pdf)) |                    | x |
| gbd-qp      | Adaptation of the Generalized Benders Decomposition with an hessian to the cost function | x |  |
| oa | Outer approximation ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  | x |
| oa-qp | Quadratic outer approximation ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  |  |
| oa-i | Outer approximation improved with safeguard for nonlinear constraints| x | x |
| oa-qp-i | Quadratic outer approximation improved with safeguard for nonlinear constraints| x |  |
| s-v-miqp | Sequential Voronoi-based MIQP with exact or Gauss-Newton Hessian ([Ghezzi et al, 2023](https://publications.syscop.de/Ghezzi2023a.pdf)) |  |  |
| s-b-miqp | Sequential Benders-based MIQP ([Ghezzi, Van Roy, et al, 2024](https://arxiv.org/pdf/2404.11786)) | x | x |
| fp | Feasibility Pump for MINLP ([Bertacco, et al, 2007](https://doi.org/10.1016/j.disopt.2006.10.001)) |  |  |
| ofp | Objective Feasibility Pump for MINLP ([Sharma, et al, 2016](https://doi.org/10.1007/s10589-015-9792-y)) |  |  |
| rofp | Random Objective Feasibility Pump | x |  |
| bonmin | ([Bonami, et al, 2006](https://doi.org/10.1016/j.disopt.2006.10.011)) -- Same as bonmin-bb |  | x |
| bonmin-bb | A nonlinear branch-and-bound algorithm based on solving a continuous  nonlinear  program  at  each  node  of  the  search  tree  and  branching on variables  ([Gupta, Ravindran, 1980](https://www.coin-or.org/Bonmin/bib.html#Gupta80Nonlinear)) |  | x |
| bonmin-hyb | A  hybrid  outer-approximation  /  nonlinear  programming  based     branch-and-cut algorithm  ([Bonami et al. 2008](http://domino.research.ibm.com/library/cyberdig.nsf/1e4115aea78b6e7c85256b360066f0d4/fdb4630e33bd2876852570b20062af37?OpenDocument)) |  | x |
| bonmin-oa | An  outer-approximation  based  decomposition  algorithm  ([Duran, Grossmann, 1986](https://www.coin-or.org/Bonmin/bib.html#DG)), ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  | x |
| bonmin-qg | An outer-approximation based branch-and-cut algorithm  ([Quesada, Grossmann, 1994](http://dx.doi.org/10.1016/0098-1354(92)80028-8)) |  | x |
| bonmin-ifp | An iterated feasibility pump algorithm   ([Bonami, et al, 2009](http://dx.doi.org/10.1007/s10107-008-0212-2)) |  |  |
| cia | Combinatorial Integral Approximation ([Sager, et al, 2011](https://link.springer.com/article/10.1007/s00186-011-0355-4)) using `pycombina` ([Buerger, et al, 2020](https://publications.syscop.de/Buerger2020a.pdf)) -- installation instructions below|  |  |
| nlp | Solve the canonical relaxation of the MINLP (integers are relaxed to continuous variables) |  |  |
| nlp-fxd | Fix the integer variables of the MINLP and solve the corresponding NLP|  |  |
| ~~ | BELOW TO UPDATE | ~~ |
| milp-tr | MILP-based trust region approach ([De Marchi, 2023](https://doi.org/10.48550/arXiv.2310.17285)) | |
| benderseq | *Experimental version based on GBD where a solution with a cost equal to the relaxed solution cost is searched* | x |
|  | **Other 'solver'-like options:** |  |
|ampl | Export to ampl format (experimental) |  |
|test | Test a problem by listing all objective values around the relaxed solution for every perturbation of 1 variables (making it integer) together with the gradient value | |

### Warm starting
It is possible to warm start every solver with the solution of another by concatenating the desired solvers when executing `python3 minlp_algorithms`.
For instance, to combine use the solution of the feasibility pump as a warm start to sequential Benders-based MIQP, execute the following:
```
python3 minlp_algorithms <mode> fp+s-b-miqp <problem>

```

## Install pycombina

- Install gcc

- Set up and activate a fresh Python virtual environment (Python >= 3.7 should work)

- If you want to use pycombina for comparison, install the dependencies listed at https://pycombina.readthedocs.io/en/latest/install.html#install-on-ubuntu-18-04-debian-10, then clone and build pycombina by running:


        git clone https://github.com/adbuerger/pycombina.git
        cd pycombina
        git submodule init
        git submodule update
        python setup.py install

# Docs

In the folder `docs/` we provide two python scripts `example.py` and `stats_analysis.py`.
- `example.py` shows how to a user can define its own MINLP and call one of the algorithm implemented in this library to solve it.
- `stats_analysis.py` shows how one can retrieve the statistics stored by running the algorithms. More advanced statistics analysis is left to the user.\
  **Note that:** to save stats set the env variable `LOG_DATA=1` by runnning `export LOG_DATA=1` from a terminal console.


# Citing

If you find this project useful, please consider giving it a :star: or citing it if your work is scientific:
```bibtex
@software{minlp-algorithms,
  author = {Ghezzi, Andrea and Van Roy, Wim},
  license = {GPL-3.0},
  month = may,
  title = {minlp-algorithms: {P}ython-{C}as{AD}i-based package containing several algorithms for solving mixed-integer nonlinear programs ({MINLP}s)},
  url = {https://github.com/minlp-toolbox/minlp-algorithms},
  version = {0.0.1},
  year = {2024}
}
```

# Contributing
Contributions and feedback are welcomed via GitHub PR and issues!

# License
This software is under xxx, please check xxxx, for more details.