# Python csv->biobib conversion

## General Description

This repo contains python scripts that can convert csv files into latex tables for use in a bio-bib file for merit and promotion at UCSB.

To use these python scripts, you must first generate a set of csv files, which are contained in the CV/ folder.

The `tables.py` file contains object models for the table class, which is sub-classed to make the various tables used in the bio-bib. Each `Table` class or subclass is instanced by reading in a csv file. 

The data from the csv file is stored as a pandas dataframe within the `Table` object. 

This dataframe is then cleaned to format the data properly, and - if necessary - filtered based on the evaluation period, which is stored as a boolean column in the csv files.  

Once the data are formatted properly, they are written out to `.tex` files, that are then imported into the `biobib.tex` file during creation of the `biobib.pdf`.

## Usage:

1. Clone the repo
2. Edit the `biobib_kkc.tex` file. The header, title, and first page information is specific to me, not you. There are also other text blocks and footnotes that are specific to me that must be edited.
2. install python requirements using `pip install -f requirements.txt`
3. Edit `build_biobib.py` as needed (e.g. changing csv filenames, or whatever you need to do)
4. Run `build_biobib.py` (e.g. `> python build_biobib.py`)
5. Compile the `biobib_kkc.tex` file (hopefully you've renamed it).
6. Watch the sweet `biobib_kkc.pdf` file come roaring to life (again, rename it).

## Notes:

1. I keep track of conference presentations separate from lectures and seminars. But apparently, these are considered the same thing in the UCSB biobib universe. So, when building the `Lectures.tex` file, I have to side-load the `Conference Abstract-Table.csv` during initialization of the `Lecture` object (cf. L401-404 in `tables.py`). This is super janky, but it works.

2. While I tired to make the `tables.py` structure general and extensible, it's not really well factored, so there is a fair amount of repetition related to cleaning up dataframes and formating of table output.

3. This effort was inspired by Stuart Sweeney's similar system for generating biobib files using R scripts. 

4. Pull requests welcome. 




