# Python csv->biobib conversion

This repo contains python scripts that can convert csv files into latex tables for use in a bio-bib file for merit and promotion at UCSB.

To use these python scripts, you must first generate a set of csv files, which are contained in the CV/ folder.

The `tables.py` file contains object models for the table class, which is sub-classed to make the various tables used in the bio-bib. Each `Table` class or subclass is instanced by reading in a csv file. 

The data from the csv file is stored as a pandas dataframe within the `Table` object. 

This dataframe is then cleaned to format the data properly, and - if necessary - filtered based on the evaluation period, which is stored as a boolean column in the csv files.  

Once the data are formatted properly, they are written out to `.tex` files, that are then imported into the `biobib.tex` file during creation of the `biobib.pdf`.




