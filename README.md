# Automated Playlist Continuation

**Team Members:** Landon Edwards ,  Yash Lalwani

## Introduction

This repository contains a Python implementation of our music recommendation system for the  [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

## Development Environment

Python Anaconda  v4.10.1

tf-nightly-gpu  v2.6.0.dev20210415  (to be updated when tensorflow 2.6 is released) 

CUDA v11.3

CUDNN v8.2.0

GeForce RTX 3090

## Dataset

Spotify has produced the Million Playlist Dataset . This data can be obtained by signing up for the[ aicrowd competition ](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge),and download the zips conatined  within the Resources tab.

## Directory Structure

The scripts read inputs by default based on the following organization of files.

1. Unzip the file spotify_million_playlist_dataset.zip and move all the files located within [] to the folder **data/**
2. Unzip the the file spotify_million_playlist_dataset_challenge.zip  and move the file located within [] to the folder **challenge_data/**

## Data Preprocessing

Run the script  `DataPreProcess.py` to produce the directory preprpocessed_data which is required for use in following scripts.

Run the script  `trainingValidationSplit.py` to produce training and validation sets for use in training.

## Executing

A model can be trained and saved by running `train.py`

*Arguments of* `train.py`:

--mode :  One of  **train** (Begin New training session), **resume** (Resume a previous checkpoint), **load** (Load a previously trained model and begin new training session) .

--model_name : A unique identifier to  create a folder in which to save checkpoints and training results under.

--models_dir : directory path of where to save model.

--resume_dir : If selected mode is resume, specify location of directory where model checkpoint is stored

