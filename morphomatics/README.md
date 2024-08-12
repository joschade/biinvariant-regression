# GDAP library

Code Repository for algorithms / prototypes developed within the geometric data analysis and processing (GDAP) group at [Zuse Institute Berlin](https://www.zib.de/visual/geometric-data-analysis-and-processing).

## Dependencies
* jax, jaxlib
* jraph, dm-haiku, optax
* pyvista, pyvistaqt, pyqtconsolepi

Optional
* pymanopt
* sksparse

## Recommended development setup
* IDE: PyCharm (Community Edition)
* Package/Environment management: Anaconda


## Known issues
* ```jaxlib``` on GPUs (Nvidia) is only supported for Linux, there are [experimental options for Windows and Mac](https://jax.readthedocs.io/en/latest/installation.html), though 
* Reading OBJs with pyvista may fail for wrong locale (add/set LC_NUMERIC=C to environment)
* Conflicting Qt libs in system (Workaround: Shadowing of system libs)
    ```bash
    export LD_LIBRARY_PATH=$(python -c "import PyQt5 as _; print(_.__path__[0])")/lib:${LD_LIBRARY_PATH}
    ```

## How to run GUI

```bash
LC_NUMERIC=C;PYTHONPATH=. python morphomatics/gui/Morphomatics.py
```
