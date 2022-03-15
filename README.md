# Workshop: Practical Anomaly Detection


## Target Audience

Entry to mid-level data scientists / machine learning engineers / ...


## Goals

This workshop introduces the participants to the topic of anomaly detection.
Inparticular, it provides initial answers for the following questions:

1. What is an anomaly?
1. Where can anomaly detection be applied?
1. What methods are available?
1. How can I evaluate the performance of an anomaly detection system?
1. What is Extreme Value Theory (EVT)?
1. How EVT contributes to anomaly detection?
 
Successful participants will acquire the basic theoretical knowledge and the
practical skills to perform anomaly detection on simple use cases using
state-of-the-art methods. While mostly covering introductory material, the
workshop tries to provide enough depth to be interesting for participants with
some experience in anomaly detection.    	

## Agenda

# Schedule

The workshop is planned for up to two full days. A possible schedule is depicted below:  

|Day No.|Title |Duration|Schedule|
|:------:|-----|:------:|:------:|
| 1| Introduction to Anomaly Detection| 90 min|9:00 - 10:30|
| 1| Coffee Break| 15 min| 10:30 - 10:45|
| 1| Metrics and Outlier Scores| 75min| 10:45 - 12:00|
| 1| Lunch Break |  60 min |   12:00 - 13:00|
| 1| Kernel Density Estimation | 90 min   |   13:00 - 14:30|
| 1| Isolation Forest | 30 min | 14:30 - 15:00|
| 1| Coffee Break |   15 min    |   15:00 - 15:15|
| 1| Autoencoder | 60 min  | 15:15 - 16:15|
| 1| Time Series Analysis | 45 min | 16:15 - 17:00 |
| 2| Time Series Forecasting | 90 min  | 9:00 - 10:30|
| 2| Coffee Break          |   15 min  |    10:30 - 10:45 |
| 2| Introduction to EVT |   75 min  |   10:45 - 12:00|
| 2| Lunch Break           |   60 min  |    12:00 - 13:00 |
| 2| Generalized Extreme Value Distributions  |   120 min  |    13:00 - 15:00 |
| 2| Coffee Break          |   15 min  |    15:00 - 15:15 |
| 2| Generalized Pareto Distributions |   105 min  |    15:15 - 17:00 |


## Prerequisites

We try to keep the prerequisites as low as possible. Some experience with
standard machine learning methods and basic knowledge of the python machine
learning stack are however highly recommended.

## Setup

We recommend to install rise with conda (installation with pip may cause problems). We also use the
[spellchecker](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/spellchecker/README.html)
and [equation-numbering](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/equation-numbering/readme.html)
extensions. 

To configure everything, activate a conda env and run
```bash
conda install -c conda-forge notebook rise jupyter_contrib_nbextensions jupyter_nbextensions_configurator
python ./configure_spellcheck_dict.py
jupyter nbextension enable spellchecker/main
jupyter nbextension enable equation-numbering/main
```

Use the extension-configurator for customizing your slideshow as described 
[here](https://rise.readthedocs.io/en/stable/customize.html).

The hide_code extension is useful to see the slides in presentation mode. You can install it by typing the following lines
```bash
pip install hide_code
jupyter nbextension install --py hide_code
jupyter nbextension enable --py hide_code
jupyter serverextension enable --py hide_code
```

Finally, clone repository, change into the directory of the cloned repository and type
```bash
pip install -r requirements.txt
pip install -e src
```

### HTML Animations
Some notebook cells use HTML animations. If you encounter an error 'Requested MovieWriter (ffmpeg) not available' try following the advices given 
[here](https://suryadayn.medium.com/error-requested-moviewriter-ffmpeg-not-available-easy-fix-9d1890a487d3).
