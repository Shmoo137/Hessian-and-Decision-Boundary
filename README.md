# Unveiling the Hessian’s Connection to the Decision Boundary

This repository contains the code for the main Figures from 'Unveiling the Hessian’s Connection
to the Decision Boundary' by Mahalakshmi Sabanayagam, Freya Behrens, Urte Adomaityte and Anna Dawid [[arxiv](https://arxiv.org/abs/2306.07104)].

## General structure

To facilitate experimentw with different datasets and training setups, the code is structured, such that every setup with dataset+training+architecture is pre-defined in a json configuration file in the directory `configs`, ordered by the datasets and experiments conducted.

The training to obtain a model checkpoint, can be done using the `run_training.py` with the appropriate config as a command line argument, e.g. `python3 run_training.py --config=intro1D/normal_training.json`. This saves the model checkpoint in `results/models`.

To obtain the analysis of the Hessian andto compute its eigenvectors, the script `run_hessian_analysis.py` is used analogously. The Hessian and its eigendecomposition is stored in `results/grads`. Be careful, these files are quite large and might take up a big amount of space (as the Hessian's size scales quadratically in the number of model parameters).

Finally, the results are visualized to produce the figures from the paper using the scripts in the `plotting` directory.

Overall, the structure hopefully allows you to also easily play around with your own models.

In the following, we detail the steps to produce the exact figures from our paper.

## Reproducing the Experiments

First creating the data that will be visualized, then creating the plots from it.
config > training > hessian_analysis > plotting
interdependencies between the configs (e.g. adversarial init initialized at random training model checkpoint)
all configs for paper available, but only show the figures from the main + normal training for all datasets, scripts for more available upon request. 

### Training

This is the basic code required to execute all traning and analysis, on which the plotting functions are based. As later figures reuse results from previous figures, the *order matters* in which the commands are executed.
Executing all commands on a local desktop machine should not take longer than 2 hrs.

```
# Fig. 1
python3 run_training.py --config=intro1D/normal_training.json
python3 run_hessian_analysis.py --config=intro1D/normal_training.json

# Fig. 2
python3 run_training.py --config=gauss/normal_training.json
python3 run_hessian_analysis.py --config=gauss/normal_training.json

# Fig. 3
python3 run_training.py --config=gauss/random_label_training.json
python3 run_training.py --config=gauss/adversarial_init_training.json
python3 run_hessian_analysis.py --config=gauss/adversarial_init_training.json
python3 run_training.py --config=gauss/large_norm_training.json
python3 run_hessian_analysis.py --config=gauss/large_norm_training.json

# Fig. 4
# None

# Fig. 5
python3 run_training.py --config=gauss_checkerboard_linear/normal_training_close.json
python3 run_training.py --config=gauss_checkerboard_noisy/normal_training_close.json
python3 run_training.py --config=gauss_checkerboard_linear/noisy_init_training_close.json

# Fig. 6
python3 run_training.py --config=mnist2D/normal_training_c017.json
python3 run_hessian_analysis.py --config=mnist2D/normal_training_c017.json
python3 run_training.py --config=mnist2D/random_label_training_c017.json
python3 run_training.py --config=mnist2D/adversarial_init_training_c017.json
python3 run_hessian_analysis.py --config=mnist2D/adversarial_init_training_c017.json

# Additional Datasets
python3 run_training.py --config=circle/normal_training.json
python3 run_hessian_analysis.py --config=circle/normal_training.json
python3 run_training.py --config=hierachical/normal_training.json
python3 run_hessian_analysis.py --config=hierachical/normal_training.json


```

### Plotting

The plots are saved in the directory `figures`, we provide the expected outcomes in this repository.

```
# Fig. 1: 1D example.
python3 plotting/intro_explain_plot.py --config=intro1D/normal_training.json 


# Fig. 2: Experimental results on gaussian dataset.
python3 plotting/hessian_encode_boundary.py --config gauss/normal_training.json --precomputed_hessian


# Fig. 3:  Decision boundaries of different complexities for gaussian.
python3 plotting/complex_boundaries_take_more_vectors.py --precomputed_hessian
python3 plotting/generalization_histogram_vertical.py --data=gauss


# Fig. 4: Alignment of all training data with the top 25 Hessian eigenvectors.
python3 plotting/overlap_per_gradient_horizontal.py --config gauss/normal_training.json gauss/adversarial_init_training.json gauss/large_norm_training.json --confignames normal adversarial large_norm --precomputed_hessian


# Fig. 5: Simplicity bias and margin estimation for checkerboard. 
python3 plotting/simplicity_bias.py --config gauss_checkerboard_linear/normal_training_close.json gauss_checkerboard_linear/noisy_init_training_close.json 
python3 plotting/margin_estimation.py --config gauss_checkerboard_linear/normal_training_close.json gauss_checkerboard_linear/noisy_init_training_close.json 


# Fig. 6: Normal and adversarial initialization training for MNIST-017 with t-SNE visualization.
python3 plotting/mnist_validation.py --config mnist2D/normal_training_c017.json mnist2D/adversarial_init_training_c017.json --precomputed_hessian 

# Additional data:
python3 plotting/hessian_encode_boundary.py --config gauss/normal_training.json circle/normal_training.json half_moon/normal_training.json hierachical/normal_training.json gauss_checkerboard_linear/noisy_init_training_close.json --precomputed_hessian 

```


## Contact

If you have questions, we are happy to help. Please contact one of the authors via email.
