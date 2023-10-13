.. _libmpc++-cite:

*************
Cite libmpc++
*************

If you use libmpc++ in work that leads to a publication, we would appreciate it if you would kindly cite libmpc++ in your manuscript. 
Please cite both the libmpc++ library and the authors of the specific algorithm(s) that you employed in your work. 
Cite libmpc++ as something like

::

    @misc{libmpc++,
    author = {Piccinelli, Nicola},
    title = {Libmpc++: A library to solve linear and non-linear MPC},
    howpublished = {https://github.com/nicolapiccinelli/libmpc}
    }

External dependecies
====================

To cite appropriately the other libraries used in libmpc++ in the following you can find the BibTeX entries requested by the
libraries authors.

Eigen3
------

For detailed information on how cite the library properly please refers to https://eigen.tuxfamily.org/index.php?title=BibTeX.

::

    @misc{eigenweb,
    author = {Ga\"{e}l Guennebaud and Beno\^{i}t Jacob and others},
    title = {Eigen v3},
    howpublished = {http://eigen.tuxfamily.org},
    year = {2010}
    }

NLopt
-----

For detailed information on how cite the library properly please refers to https://nlopt.readthedocs.io/en/latest/Citing_NLopt/,
the default solver used for non-linear programming is SLSQP.

::

    @misc{nlopt,
    author = {Johnson, Steven G.},
    title = {The NLopt nonlinear-optimization package}
    howpublished = {http://github.com/stevengj/nlopt}
    }

OSQP
----

For detailed information on how cite the library properly please refers to https://osqp.org/docs/citing/index.html.

::

    @article{osqp,
    author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
    title   = {{OSQP}: an operator splitting solver for quadratic programs},
    journal = {Mathematical Programming Computation},
    volume  = {12},
    number  = {4},
    pages   = {637--672},
    year    = {2020},
    doi     = {10.1007/s12532-020-00179-2},
    url     = {https://doi.org/10.1007/s12532-020-00179-2},
    }
