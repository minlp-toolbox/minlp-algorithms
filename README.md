# Preparation

- Install gcc

- Set up and activate a fresh Python virtual environment (Python >= 3.7 should work)

- If you want to use pycombina for comparison, install the dependencies listed at https://pycombina.readthedocs.io/en/latest/install.html#install-on-ubuntu-18-04-debian-10, then clone and build pycombina by running:


        git clone https://github.com/adbuerger/pycombina.git
        cd pycombina
        git submodule init
        git submodule update
        python setup.py install

# Usage

From the root of repo

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
|        MIP_SOLVER      | gurobi/highs | Configure MIP solver        |



There is a wide list of solvers included:

| Solvers | Description                                                  | New?                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| gbd        | Generalized Benders Decomposition $^{1}$                   |                    |
| gbd-qp      | Adaptation of the Generalized Benders Algorithm with an added hessian to the cost function | x |
| oa | Outer approximation$^{2}$ |  |
| oa-qp | Quadratic outer approximation$^{2}$ |  |
| oa-i | Outer approximation improved with safeguard for nonlinear constraints| x |
| oa-qp-i | Quadratic outer approximation improved with safeguard for nonlinear constraints| x |
| s-v-miqp | Sequential Voronoi-based MIQP with exact or Gauss-Newton Hessian ([Ghezzi et al, 2023](https://publications.syscop.de/Ghezzi2023a.pdf)) |  |
| s-b-miqp | Sequential Benders-based MIQP ([Ghezzi, Van Roy, et al, 2024](https://arxiv.org/pdf/2404.11786)) | x |
| fp | Feasibility Pump for MINLP$^{3}$ |  |
| ofp | Objective Feasibility Pump for MINLP $^{4}$ |  |
| rofp | Random Objective Feasibility Pump | x |
| ~~ | BELOW TO UPDATE | ~~ |
| benders_old_tr | Benders Region without gradient corrections (old implementation) starting from an integer point | x |
| benders_old_tr_rel | Sequential-MIQP with Benders Region without gradient corrections (old implementation) starting from the relaxed point | x |
| benders_tr | Sequential-MIQP with Benders Region including gradient correction | x |
| benders_tr_fp | **benders_tr**, warm-started with a feasibility pump | x |
| benders_trm | Alternating Sequential-MIQP with Benders Region together with a LB-MILP including gradient correction. | x |
| benders_trm_fp | **benders_trm**, warm-started with a feasibility pump | x |
| benders_trl | Alternative for benders_trm where we use the MIQP with a corrected hessian as a lower bound function | x |
| benderseq | *Experimental version based on GBD where a solution with a cost equal to the relaxed solution cost is searched* | x |
| bonmin | Same as bonmin-bb |  |
| bonmin-bb | A simple branch-and-bound algorithm based on solving a continuous  nonlinear  program  at  each  node  of  the  search  tree  and  branching on variables  [[Gupta 1980\]](https://www.coin-or.org/Bonmin/bib.html#Gupta80Nonlinear) |  |
| bonmin-hyb | A  hybrid  outer-approximation  /  nonlinear  programming  based     branch-and-cut algorithm  [[Bonami et al. 2008\]](http://domino.research.ibm.com/library/cyberdig.nsf/1e4115aea78b6e7c85256b360066f0d4/fdb4630e33bd2876852570b20062af37?OpenDocument) |  |
| bonmin-oa | An  outer-approximation  based  decomposition  algorithm  [[Duran, Grossmann, 1986](https://www.coin-or.org/Bonmin/bib.html#DG),[Fletcher Leyffer 1994](http://dx.doi.org/10.1007/BF01581153)] |  |
| bonmin-qg | An outer-approximation based branch-and-cut algorithm  [[Quesada Grossmann 1994\]](http://dx.doi.org/10.1016/0098-1354(92)80028-8) |  |
| bonmin-ifp | An iterated feasibility pump algorithm   [[Bonami Cornuéjols Lodi Margot 2009\]](http://dx.doi.org/10.1007/s10107-008-0212-2)  . |  |
| relaxed | Relaxed solution using ipopt |  |
| cia | Combinatorial Integral Approximation using pycombina [[Burger A. et al., 2020]](https://link.springer.com/article/10.1007/s00186-011-0355-4) |  |
| milp_tr | MILP-based trust region approach [[De Marchi, 2023]](https://doi.org/10.48550/arXiv.2310.17285) | |
|  | **Other 'solver'-like options:** |  |
|ampl | Export to ampl format (experimental) |  |
|test | Test a problem by listing all objective values around the relaxed solution for every perturbation of 1 variables (making it integer) together with the gradient value | |



$^{1}$A. Geoffrion, “Generalized Benders Decomposition,” Journal of Optimization Theory and
Applications, vol. 10, pp. 237–260, 1972.

$^{2}$ R. Fletcher and S. Leyffer, “Solving Mixed Integer Nonlinear Programs by Outer Approxi-
mation,” Mathematical Programming, vol. 66, pp. 327–349, 1994.

$^{3}$ Bertacco, L., Fischetti, M., & Lodi, A. (2007). A feasibility pump heuristic for general mixed-integer problems. Discrete Optimization, 4(1), 63-76.

$^{4}$ Sharma, S., Knudsen, B. R., & Grimstad, B. (2016). Towards an objective feasibility pump for convex MINLPs. Computational Optimization and Applications, 63, 737-753.
