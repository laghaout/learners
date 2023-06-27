
# Table of Contents

1.  [Dockerize](#org5e2e795)
2.  [Load a pickled learner](#org3792869)



<a id="org5e2e795"></a>

# Dockerize

To dockerize,

1.  Run `mv .env_template .env` and edit the environment variables if needed.
2.  Run `sh dockerize.sh`.


<a id="org3792869"></a>

# Load a pickled learner

Place yourself in the same directory as `learner.py` then run the following:

    import pickle
    from learners.learner import Learner

    learner = Learner()
    learner = pickle.load(open("../lesson/learner.pkl", "rb"))
    report = learner.report

Note that the path `../lesson/learner.pkl` may need some editing.

