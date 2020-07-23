# README #

This is a model to calculate the snow water equivalent out of given snow heights. The model is ported from R
code which was written from Winkler et al. 2020 "Snow Water Equivalents exclusively from Snow Heights and their
temporal Changes: The ∆SNOW.MODEL".

The code structure was adapted a bit but the calculations, input and output variables stayed the same.

Ported by Manuel Theurl who is taking no warranties for the correctness of the R to python port.


### How do I get set up? ###

## Installation ##
Clone the repository using the **Clone button**.
The software needs specific modules which can be downloaded automatically using the following commands in the Software directory of the repo:

Windows Terminal: 
```
#!python

pip install -r requirements.txt
```

Linux Terminal:  

```
#!python

pip3 install -r requirements.txt
```


You can test the program by typing

Windows Terminal: 
```
#!python

python main.py
```

Linux Terminal: 
```
#!python

python3 main.py
```

## Usage ##

First create a SnowToSwe() object with the required (adpated) attributes.
Then either use convert_csv method or convert_list method. Both methods are documented by doc string inside the script. Please refer to that for further details. 



