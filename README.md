# dlab-public
Code to run the antibody virtual screening pipeline described in the paper "DLAB - Deep learning methods for structure-based virtual screening of antibodies" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.12.430941v1))[1]

## Install
1. Install libmolgrid [2] by building from source according to the instructions on the authors github [repository](https://github.com/gnina/libmolgrid). Use the source code provided in this repository, as a small change was made to accomodate the centering method employed for DLAB.
2. Get [ZDock](http://zdock.umassmed.edu/software/) and copy the source code files into the folder external/zdock-3.0.2-src.
3. Install the python requirements.

## Test your install
The folder tests contains a number of tests you can use to test if your folder structure and python environment are set up correctly. The can be run either using pytest or by running each file individually.

## Data preparation
1. Prepare your input antibody and antigen structures by running the mark_sur script as per the ZDock README. If you want to limit docking to the (predicted) interaction site, block atoms outside the interaction site by changing column 55-56 after running mark_sur to 19 as per the ZDock README. There is more detail on this in the paper.
2.  Generate a .csv file containing all antibody and antigen pairings you want to investigate. An example is shown in example_input_files/pairings.csv.
3.  Use the data\_prep\_pipeline.py script to run ZDock and generate types files. This uses python implementation of the atomtyper functionality in libmolgrid [2]:

		python data_prep_pipeline.py -c data_prep_config.yaml
		
    Where data\_prep\_config.yaml configures the pipeline (see example_input_files/data_prep_config.yaml).

## Running DLAB
For this script, you will need to be on a machine with GPU and CUDA, which is why this is seperate from the data preperation script (docking and preperation can be run on cpu compute servers before running GPU computations). Run DLAB (both rescoring and virtual screening in one go) using this command:
		
	python dlab_re_vs_pipeline.py -c dlab_config.yaml

Where the yaml file defines the locations and types of models used as well as the input data and output file. An example can be found in example_input_files/dlab_config.yaml.

## Analysing the output

## References
[1] DLAB - Deep learning methods for structure-based virtual screening of antibodies. C Schneider, A Buchanan, B Taddese, CM Deane, bioRxiv, 2021
[2] libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications. J Sunseri, DR Koes. Journal of Chemical Information and Modeling, 2020
