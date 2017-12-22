# ARHMM-C elegans

Code to reproduce results in:

*Quantifying the behavioral dynamics of C. Elegans with autoregressive hidden Markov models*
<br /> by E. Kelly Buchanan, Akiva Lipschitz, Scott Linderman and Liam Paninski

presented at Worm's Neural Information Processing (WNIP) Workshop 
<br /> at Neural Information Processing Systems (NIPS), 2017.

## Getting Started
List of input files:
* andre_ftp_matfiles.txt : List of files

Jupyter Notebooks:
* F1-Select-worms : Select worms
* F2-Preprocess-worms : Preprocess worms following Stephens et al., 2008.
* F3-Fit-all-worms-once : Fit all worms at once
* F4-Make_final_plots : Make plots from results

Run Experiments (run between F3 and F4):
* fit_worm10.py :

Helper functions:
* preprocessing.py
* tmputil.py
* tmputil_plots.py

Results directory:
* results_strains/

## Authors

* **E. Kelly Buchanan** - *Initial work*
* **Scott Linderman** - *Initial work* - [PyHSMM-spiketrains](https://github.com/slinderman/pyhsmm_spiketrains)

See also the list of [contributors](https://github.com/ekellbuch/arhmm-celegans/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **Andre Brown** for providing the data.
