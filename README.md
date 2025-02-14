# Quantum algorithm for partial differential equations of nonconservative systems with spatially varying parameters

[![python](https://img.shields.io/badge/python-v3.10-blue)](https://www.python.org/downloads/release/python-3109/)

This repository provides codes to reproduce results of the paper:
```
Yuki Sato, Hiroyuki Tezuka, Ruho Kondo, and Naoki Yamamoto,
Quantum algorithm for partial differential equations of nonconservative systems with spatially varying parameters,
Physical Review Applied 23, 014063, 2025.
```

## Requirement

|  Software  |  Version  |
| :----: | :----: |
|  python  |  3.10  |
|  tqdm  |  4.66  |
|  qiskit  |  1.0  |
| qiskit-aer | 0.13 |
 
## Contents
 
Section IV A, Acoustic simulation: [Section4A-waveEq.ipynb](/Section4A-waveEq.ipynb)

Section IV B, LCU coefficient: [Section4B-LCU.ipynb](/Section4B-LCU.ipynb)

Section IV B, Heat simulation: [Section4B-heatEq.ipynb](/Section4B-heatEq.ipynb)

Note: codes for Hamiltonian simulation are based on those in [our another repository](https://github.com/ToyotaCRDL/hamiltonian-simulation-for-hyperbolic-pde)
  
## Citation

If you find it useful to use these codes in your research, please cite the following paper.

```
Yuki Sato, Hiroyuki Tezuka, Ruho Kondo, and Naoki Yamamoto, Quantum algorithm for partial differential equations of nonconservative systems with spatially varying parameters, Physical Review Applied 23, 014063, 2025.
```

In bibtex format:
```
@article{sato2025quantum,
  title={Quantum algorithm for partial differential equations of nonconservative systems with spatially varying parameters},
  author={Sato, Yuki and Tezuka, Hiroyuki and Kondo, Ruho and Yamamoto, Naoki},
  journal={Physical Review Applied},
  volume={23},
  number={1},
  pages={014063},
  year={2025},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.23.014063},
  url = {https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.23.014063}
}
```


Since the idea of constructing the quantum circuits for time evolution operators is based on our another paper, please also consider to cite the following paper.
```
Yuki Sato, Ruho Kondo, Ikko Hamamura, Tamiya Onodera, and Naoki Yamamoto, Hamiltonian simulation for hyperbolic partial differential equations by scalable quantum circuits, Physical Review Research, 6, 033246, 2024.
```

In bibtex format:
```
@article{sato2024hamiltonian,
  title={Hamiltonian simulation for hyperbolic partial differential equations by scalable quantum circuits},
  author={Sato, Yuki and Kondo, Ruho and Hamamura, Ikko and Onodera, Tamiya and Yamamoto, Naoki},
  journal={Physical Review Research},
  volume={6},
  number={3},
  pages={033246},
  year={2024},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.6.033246},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.6.033246}
}
```
 
# License

See the [LICENSE](/LICENSE.txt) file for details
