
# Table of Contents

1.  [Dockerize](#org40e193b)
2.  [Load a pickled learner](#orgd7cdf2c)



<a id="org40e193b"></a>

# Dockerize

To dockerize, run `sh dockerize.sh`


<a id="orgd7cdf2c"></a>

# Load a pickled learner

Place yourself in the same directory as `learner.py`.

    import pickle

    # Go to the learner.py location
    from learners.learner import Learner

    learner = Learner()
    learner = pickle.load(open("../lesson/learner.pkl", "rb"))
    report = learner.report

