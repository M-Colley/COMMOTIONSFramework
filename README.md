This repository contains a Python 3 implementation of a modular framework for modeling road user interactions, developed in the EPSRC-funded project [COMMOTIONS](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/S005056/1). 

* This root folder contains some base modules: `commotions.py` provides general basics (including the cooperative agent base classes `BaseAgent`, `AgentWithGoal`, and `AgentWithIntermittentActions`), and `sc_scenario*.py` provides an implementation of the framework for two-agent "straight crossing paths" scenarios on perpendicular or arbitrary non-parallel crossing paths (see `sc_scenario.get_crossing_geometry()`).

* The main current entry point to this repository is the "*SCPaper*" folder, which makes use of the modules in this root folder to implement a number of tests of different model variants obtainable from the framework, in vehicle-pedestrian interactions, including comparisons of model predictions to empirical data. This work has been written up as a paper, and the "*SCPaper*" folder contains the code needed to regenerate all of the results and figures in that paper. `SCPaperScenario` and `sc_run.run_simulation()` default to the perpendicular (90°) crossing geometry used in the paper, and also accept an optional `crossing_angle` argument (in degrees) for non-perpendicular crossings.

* The "*diary notebooks*" folder contains Jupyter notebooks which are effectively diary entries, documenting the work on developing and testing the code in this repository, providing further insight into the motivations behind the many modelling decisions made along the way.

* The "*handwritten notes*" folder contains scans of some mathematical derivations referenced in the code and diary notebooks.

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png
