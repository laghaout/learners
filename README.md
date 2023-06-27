
# Table of Contents

1.  [Dockerize](#org900fd20)
2.  [Load a pickled learner](#orge199cdb)



<a id="org900fd20"></a>

# Dockerize

To dockerize,

1.  Run `cp .env_template .env` and edit the environment variables if needed.
2.  Run `sh dockerize.sh`.


<a id="orge199cdb"></a>

# Load a pickled learner

Place yourself in the same directory as `learner.py` then run the following:

    import pickle
    from learners.learner import Learner

    learner = Learner(lesson_dir=None)
    learner = pickle.load(open('../lesson/learner.pkl', 'rb'))
    report = learner.report

Note that the path `../lesson/learner.pkl` may need some editing.

