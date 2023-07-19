
# Table of Contents

1.  [Dockerize](#org7801b25)
2.  [Load a pickled learner](#orgb7ab378)



<a id="org7801b25"></a>

# Dockerize

To dockerize,

1.  run `cp .env_template .env` and edit the environment variables if needed then
2.  run `sh dockerize.sh`.


<a id="orgb7ab378"></a>

# Load a pickled learner

Place yourself in the same directory as `learner.py` then run the following:

    import pickle
    from learners.learner import Learner
    
    learner = Learner(lesson_dir=None)
    learner = pickle.load(open('../lesson/learner.pkl', 'rb'))
    report = learner.report

Note that the path `../lesson/learner.pkl` may need some editing.

