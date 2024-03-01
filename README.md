
# Table of Contents

1.  [Description](#org4162102)
2.  [Usage](#org6ddeaae)
    1.  [Dockerize](#orgd231f32)
    2.  [Install the `learners` package locally](#org105ce33)
        1.  [Linux (pip)](#orgfc448b7)
    3.  [Mock run](#orgb9c82e3)
    4.  [Load a pickled learner](#orgd24724b)



<a id="org4162102"></a>

# Description


<a id="org6ddeaae"></a>

# Usage


<a id="orgd231f32"></a>

## Dockerize

To dockerize,

1.  run `cp .env_template .env` and edit the environment variables if needed then
2.  run `docker compose build learners`.


<a id="org105ce33"></a>

## Install the `learners` package locally


<a id="orgfc448b7"></a>

### Linux (pip)

`sh devops.sh`


<a id="orgb9c82e3"></a>

## Mock run

Once inside the container, run `learners-mock`.


<a id="orgd24724b"></a>

## Load a pickled learner

To load a learner that has been pickled, place yourself in the same directory as `learner.py` then run the following:

    import pickle
    from learners.learner import Learner

    learner = Learner(lesson_dir=None)
    learner = pickle.load(open('../lesson/learner.pkl', 'rb'))
    report = learner.report

Note that the path `../lesson/learner.pkl` may need some editing.

