# FABSBM
An implementation of a tractable fully Bayesian method for the stochastic block model. For more details, see our paper:

A Tractable Fully Bayesian Method for the Stochastic Block Model.
Kohei Hayashi, Takuya Konishi, Tatsuro Kawamoto.
[arXiv:1602.02256](http://arxiv.org/abs/1602.02256), 2015

## Requirements
* Numpy >= 1.8.2
* Scipy >= 0.13.3

If you want to reproduce the figure of our experiment, following R packages are required:

* ggplot2
* scales
* Hmisc

## Demo
You can run the algorithm with toy data (the true number of clusters is 4) by a command line something like this:
```
python run.py balanced $(N) $(K) $(seed) $(method)
```
* $(N) is the number of nodes of data, 
* $(K) is the number of clusters of the algorithm, 
* $(seed) is the random seed, and 
* $(method) is the name of the algorithm. 
 
For example, if you want to use our algorithm, try
```
python run.py balanced 100 4 0 FVAB
```
The output is formatted as follows:
```
balanced $(N) $(true K) $(seed) $(method) $(selected K) $(runtime) $(training log-likelihood) $(testing log-likelihood)
```



## Reproducing experiment
You can also reproduce the toy experiment of our paper. For this, try
```
make -j all
make plot
```
Note: this can be time consuming (~1day). 
