# Planning files

`domain.pddl` defines the symbolic planning domain used by the neuro-symbolic integration pipeline.

`problem.pddl` contains the default CIFAR-100 world state used to initialise planning. Goals are supplied dynamically through the Python API rather than being fixed in this file.

The recovered base-state file contained unused `sponge` facts from an earlier domain variant. They were removed from this portfolio copy because the final domain declares `knife` and `dslr` as its tool constants.
