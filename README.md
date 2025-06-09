> Under Construction

# Quantum DeepONet: Neural operators accelerated by quantum computing

The data and code for the paper [P. Xiao, M. Zheng, A. Jiao, X. Yang, & L. Lu. Quantum DeepONet: Neural operators accelerated by quantum computing. *Quantum*, 9, 1761, 2025.](https://doi.org/10.22331/q-2025-06-04-1761)

## Datasets

Data generation scripts are available in the [`data`](data) folder:

- [Antiderivative](data/ode_generation.py)
- [Poisson's equation](data/poisson_generation.py)
- [Advection equation](data/advection_generation.py)
- [Burgers equation](data/burgers_generation.py)

Each script generates training and testing data for the respective problem.

## Code

All code is in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.10.1. 

To install dependencies: 

```bash
pip install -r requirements.txt
```

To train a model for a specific task, navigate to the corresponding example directory and run:

```bash
python training.py
```

After training, simulate the trained quantum DeepONet using [Qiskit](https://www.ibm.com/quantum/qiskit). The simulation scripts are located in the same folder as the training code. To run the simulation, use:

```bash
python simulation.py
```
> *Note: Some tasks may use different script names for simulation; please check the example folder for details.*


### Data-driven

- [Function 1](src/data_driven/simple_function)
- [Function 2](src/data_driven/complex_function)
- [Antiderivative](src/data_driven/antiderivative)
- [Advection equation](src/data_driven/advection)
- [Burgers' equation](src/data_driven/burgers)

### Physics-informed

- [Antiderivative](src/physics_informed/antiderivative/)
- [Poisson's equation](src/physics_informed/poisson/)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{Xiao2025quantumdeeponet,
  doi = {10.22331/q-2025-06-04-1761},
  url = {https://doi.org/10.22331/q-2025-06-04-1761},
  title = {Quantum {D}eep{ON}et: {N}eural operators accelerated by quantum computing},
  author = {Xiao, Pengpeng and Zheng, Muqing and Jiao, Anran and Yang, Xiu and Lu, Lu},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {9},
  pages = {1761},
  month = jun,
  year = {2025}
}
```

## Question

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
