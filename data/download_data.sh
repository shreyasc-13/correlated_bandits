#!/bin/bash

# Download the cropped Goodreads poetry file from Drive
# Complete file and other topics can be found at https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
gdown --id 1uEx7GMLs9nciBozzleVx4j966MuOyskb


# Download Movielens from website
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
