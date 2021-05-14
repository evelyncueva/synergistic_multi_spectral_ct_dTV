### Data & codes: Synergistic multi-spectral CT reconstruction with dTV
**Authors:** Evelyn Cueva & Matthias J. Ehrhardt

All codes are implemented in Python and use Operator Discretization Library [(ODL)](https://github.com/odlgroup/odl).


#### 1. Read and/or generate data

For real  data (bird dataset), use the file `real_generate_data.py` to read data, reconstruct reference images and to reconstruct side information. The values of regularizer parameters $\alpha$, the number of angles, detectors and reconstruction size are specified within the file. The resulting `npy` files are saved in the folder `data` in the subfolder related to dataset name `bird`. The files needed to do the reconstructions presented in the article are already save in `data/bird` folder.

For synthetic data (XCAT dataset), use the file `xcat_generate_data.py` to create the synthetic data, save reference images and reconstruct side information for different values of regularization parameter $\alpha$. As before, the resulting data files are saved in `data/xcat` path. 

Before run any experiment (explained in section below), we have to decide the data and reconstruction sizes, since references and side information need to be generated under those specifications, *i.e.*, if we want to reconstruct a $512\times 512$ image using a sinogram of size $60\times 552$, we have to generate sinograms, references and side information using these dimensions. 

#### 2. Run experiments using Forward-backward splitting

To run an instance of forward-backward splitting algorithm, we can use the file `alpha_opt_fbs.py`.
This file is written to run different experiments in parallel, for example, if we want to run different values of $\alpha$ for `xcat` data, for $E_1$, using 60 angles and dTV regularizer, we go to the bottom of this file to modify it accordingly as shown in the next block of code:

```{python, eval=FALSE, python.reticulate=FALSE}
dataset = 'xcat'

if dataset == 'bird':
    alphas = np.logspace(-3, 1, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)
elif dataset == 'xcat':
    alphas = np.logspace(-3, 2, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)

start = time.perf_counter()

if __name__ == '__main__':

    processes = []
    for alpha in alphas:
        print('Solving with alpha = {}'.format(alpha))
        p = multiprocessing.Process(target=minimization_alpha,
                                    args=['xcat', 'dTV', alpha, 'E1', 60])
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    finish = time.perf_counter()

    print('All process end in {} seg.'.format(round(finish-start, 2)))

```

In we want to run an experiment for different energies and regularizers, the next example shows how we should proceed in this case:

```{python, eval=FALSE, python.reticulate=FALSE}

start = time.perf_counter()

energies = ['E0', 'E0', 'E1', 'E1', 'E2', 'E2']
regs = ['TV', 'dTV', 'TV', 'dTV', 'TV', 'dTV']


if __name__ == '__main__':

    processes = []
    for (energy, reg) in zip(energies, regs):
        print('Solving energy {} with  regularizer {}'.format(energy, reg))
        p = multiprocessing.Process(target=minimization_alpha,
                                    args=['xcat', reg, 1e-2, energy, 60])
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    finish = time.perf_counter()

    print('All process end in {} seg.'.format(round(finish-start, 2)))
```

Once this file ends its execution, the last iteration for each combination of parameters is saved in:

`./results/{dataset}/fbs/60_angles/{energy}_alphas_{reg}/alpha_{alpha}.npy `


#### 3. Run experiments using Bregman iterations

To run an instance of Bregman iteration algorithm, we use  the file `iter_opt_bregman.py`. As before, different experiments are running in parallel, modifying the last part of the code as shown below. Here, we show how to run an experiment for $E_1$ using TV and dTV regularizers, we use $\alpha=1e1$ for both. The dataset name `bird` and number of angles 60 are fixed for this example. 

```{python, eval=FALSE, python.reticulate=FALSE}
start = time.perf_counter()
energies = ['E1', 'E1']
regs = ['TV', 'dTV']
alphas = [1e1, 1e1]

if __name__ == '__main__':

    processes = []
    for (energy, reg, alpha) in zip(energies, regs, alphas):
        print('Running {} with {} and alpha={}'.format(energy, reg, alpha))
        p = multiprocessing.Process(target=minimization_alpha,
                                    args=['bird', 60, reg, energy, alpha])
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    finish = time.perf_counter()
    total = round(finish - start, 2)
    print('All process end in {} seg.'.format(total))
```

In this case, the results are saved in each iteration in the following path:

`./results/{dataset}/bregman/60_angles/{energy}_alpha_{alpha}_{reg}`
Similarity measures as PSNR and SSIM are located in a `meas_per_iter` folder and every 10 iterations the solution in save in `solutions_per_iter`. 

The files 

- `read_FBS_files.py`, `read_bregman_files.py` and `generate_optimal_results.py` help to read the generated information for graphic representation purposes.

- `misc.py` and `misc_dataset.py` are miscellaneous files, where we included all functions needed in `alpha_opt_fbs.py` and `iter_opt_bregman.py` to run the algorithms.

- `pdhg_tnv.npy` solves our problem including total nuclear variation using primal dual hibryd gradient algorith. 


#### 3. Plot (paper) figures

The file `article_figures.py` allows us to run all figures presented in the submitted article. 


