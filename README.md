# dlab-public
Code for the paper "DLAB - Deep learning methods for structure-based virtual screening of antibodies" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.12.430941v1))[1]

## Install
Install libmolgrid [2] by building from source according to the instructions on the authors github [repository](https://github.com/gnina/libmolgrid). Use the source code provided in this repository, as a small change was made to accomodate the centering method employed for DLAB.

## Data preparation
After generating docking poses using [ZDock](http://zdock.umassmed.edu/software/), use the data\_prep\_pipeline.py script to generate types files. This uses python implementation of the atomtyper functionality in libmolgrid [2]:
	data_prep_pipeline.py -c data_prep_config.yaml
Where data\_prep\_config.yaml defines input and output folder (see folder example\_yamls)

## References
[1] DLAB - Deep learning methods for structure-based virtual screening of antibodies. C Schneider, A Buchanan, B Taddese, CM Deane, bioRxiv, 2021
[2] libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications. J Sunseri, DR Koes. Journal of Chemical Information and Modeling, 2020
