# Parametrizing GP Trees for Better Symbolic Regression Performance through Gradient Descent

Python implementation for the paper "Parametrizing GP Trees for Better Symbolic Regression Performance through Gradient Descent"
Federico Julian Camerota Verdù, Gloria Pietropolli, Luca Manzoni, Mauro Castelli. 
_The Genetic and Evolutionary Computation Conference_ __GECCO 2023__

### Abstract
Symbolic regression is a common problem in genetic programming (GP), but the syntactic search carried out by the standard GP algorithm often struggles to tune the learned expressions. On the other hand, gradient-based optimizers can efficiently tune parametric
functions by exploring the search space locally. While there is a large amount of research on the combination of evolutionary algorithms and local search (LS) strategies, few of these studies deal with GP. To get the best from both worlds, we propose embedding learnable parameters in GP programs and combining the standard GP evolutionary approach with a gradient-based refinement of the
individuals employing the Adam optimizer. We devise two different algorithms that differ in how these parameters are shared in the expression operators and report experimental results performed on a set of standard real-life application datasets. Our findings show that the proposed gradient-based LS approach can be effectively combined with GP to outperform the original algorithm.

link to the paper: https://drive.google.com/file/d/1AFN1ubYF5um8AVzI8l6RzPgzMufe16vZ/view?usp=sharing

link to the poster: https://drive.google.com/file/d/17h8WIzMIrItwWuC3YuoZ6W8pcDYJ16if/view?usp=sharing

__contacts__: gloria.pietropolli@phd.units.it

## Instructions

Code runs with Python 3.8 on Ubuntu 20.04.

To install the required libraries, enter the following command: 

```bash
pip install -r requirements.txt 
```

To run the code, enter the following command:

```bash
python main.py --alg --dataset --save_dir --hyperparams_file --comp_budget --e_in_evo --e_after_evo --lr
```

where the inputs arguments stand for: 
* `--alg` is the algorithm considered (that can be: _gp_, _gpgda_, _gpgdc_, _opgda_, _opgdc_)  
* `--dataset` is the dataset selected for the training
*  `--save_dir` is the directory where results are saved
*  `--hyperparams_file` is the file containing hyperparameters for the training 
*  `--comp_budget` is the total computational budget for the training
*  `--e_in_evo` is the gradient-based optimization performed during the evolution
*  `--e_in_evo` is the gradient-based optimization performed after the evolution
*  `--lr` is the learning rate that governs the gradient-based optimizer algorithm

that will return fitness results for the 150 runs performed and save them in the `save_dir` folder.

The code to reproduce all the box-plot of the paper (for all of the datasets) is contained in the folder `analysis`, it is sufficient to run:
```bash
python3 boxplot.py 
```
