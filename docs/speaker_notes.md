# Practical anomaly detection: Speaker notes

## Content: Day 1

### Introduction

The introduction is meant to get to know the participants. Try to get a 
feeling for their experience level. For groups with little experience it might 
be necessary to add some additional information here and there. Try to keep 
everyone on board throughout the workshop.

Develop the informal notion of anomaly together with the participants. Raise 
awareness for the specialities of anomaly detection like unsupervised learning, 
one class problem, class imbalances, etc.

### Preparation
The workshop covers a relatively broad field. Read carefully through the slides and make sure that you feel comfortable explaining everything. Ideally, you should be able to explain even smaller side notes that appear in the slides in more detail on a white board. Expect that the participants are curios and will ask questions!

### Algorithms

The slides contain a concise introductions to each algorithm. Each algorithm is 
chosen to represent a certain approach towards anomaly detection. The goal is 
to convey a sound mathematical description while still mostly focusing on the 
intuition. We start with a high level discussion of the general idea and go 
through the definition. Take your time and encourage people to ask questions. 
Having a whiteboard available to clarify some things will be helpful. The 
introduction of an algorithm is usually followed by an accompanying exercise.

### Exercises

Most exercises are  done directly in the slide notebooks.

* Some cells can be run as-is by the participants. They usually set up the 
  environment or visualize some results.
* Some cells  contain gaps which need to be filled by 
  the participants. As we want to have low technical prerequisites for the 
  workshop, most answers are fairly straightforward and usually similar 
  examples can be found in previous cells.
* It is also possible to go through the notebook together with the 
  participants on the projector if this seems more suitable for the group.

### Comments

1. Introduction     
   * Short introduction AppliedAI, UnternehmerTUM, and the speakers
   * Introduction round with the participants
     * Role
     * Interests
     * Experience
     * Expectations
   Get a feeling for the interests and experience level of the participants. Use 
     it to set weights on the topics and to moderate the exercises.
1. Introduction to Anomaly Detection     
High-level discussion to set the ground for the upcoming formal frameworks. 
   Some topics that should be brought up:
   * What is an anomaly? Intuitively clear but formal definition open problem.
   * Anomaly = not nominal $\Rightarrow$ defined through nominal class 
     $\Rightarrow$ One class classification
   * Problems of supervised learning approaches, e.g. restricting the anomaly 
     class to previously seen anomaly types (in e.g. attack scenarios), lack of 
     labels, etc.
   * Anomaly detection (business view, semantics) through outlier detection (data 
     view, topology)
1. Practical Relevance of Anomaly Detection
The goal is to substantiate the ideas of the previous section through real 
   life applications. Key messages:
   * Anomaly detection is used to ensure business processes run safely and 
     smoothly throughout the entire value chain (monitoring).
   * Often applications are highly safety relevant. Therefore, reliability and 
     robustness are key concerns.
1. Exercise: Use Case Identification 
The idea is to connect the newly learned intuition about anomaly detection to 
   applications from the participants' domains of expertise. Probably some of 
   them already have ideas in mind and the discussion of the proposals 
   hopefully yields bits and pieces that can be referred to later during the 
   practical part.
   * Discuss use cases
   * Write proposal
   * Discuss proposals
1. Contamination Framework 
This is the formal framework which we will use to understand the algorithms. 
   Key messages:
   * All presented algorithms use some subset of the given assumptions in various 
     strictness levels.
   * Wether the assumptions are valid depends strongly on the application and 
     should be considered when chosing an algorithm.
In the next sections several different approaches to anomaly detection are 
   presented. The plan is to first give a high-level description of the 
   approach followed by a short description the algorithm. An in-depth 
   exploration is carried out in the exercises.
1. Anomaly Detection via Density Estimation (Kernel Density 
   Estimation)    
1. Exercise: Choosing the Right Bandwidth  
This is a short exercise to get used to the working mode. While it is tempting 
   to experiment with different kernels when working with KDE on an anomaly 
   detection problem, the bandwidth is by far the more important parameter in 
   order to achieve high quality results. Key Learnings:
   * Sensitivity of result to different bandwidth parameters
   * Connection Bandwidth and dimensionality (curse of dimensionality)
1. Anomaly Detection via Isolation (Isolation Forest)  
1. Exercise: KDDCup99    
The final exercise. Participants are expected to perform anomaly detection on 
   this data set mostly on their own. Some groups might need assistance here. 
   However, the data set is known to be relatively easy and rather good results 
   can be expected.
1. Anomaly Detection via Reconstruction Error (Autoencoder) 
1. Exercise: MNIST     
   * We use the MNIST data set because we can nicely visualize the autoencoder on 
     image data set and because of its manageable complexity.
   * Apply convolutional autoencoder
1. Feedback     
Ask participants for feedback
   * Did they like the workshop?
   * Anything missing or not interesting?
   * What could be improved?
   * Interested in advanced anomaly detection workshop?

## Content: Day 2

### Comments

1. Anomaly Detection in Time Series Data    
   * Discuss possible scenarios where univariate time-series data occurs and 
     where identifying anomalies is crucial.
    * Get to know some basic concepts from time series analysis such as covarianve stationarity,
     trends, seasonalities, differencing, etc.
    * Review the SARIMA time series model. Show how one can choose a suitable SARIMA model using the acf and pacf plot.
   * Apply the method on the New York taxi data set. 
1.* Extreme Value Theory
  * Explain the intuition of extreme values as a special case of anomalies. Emphazise that the notion of extreme value is a quite restricted notion of anomaly. 
  * Describe how to detect more general anomalies by applying EVT to outlier scores
  * Carefully go through the main theorems of EVT. Explain the convergence condition and that not every distribution posesses a normalizing sequence that makes it converge => the theorems can not be applied.
  * Some of the exercises in this part can be too challenging for the participants. make sure that everyone can follow. For a rather inexperience audience it might be advisable to skip some exercises.
