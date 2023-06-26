#+TITLE: README

* Dockerize

To dockerize, run =sh dockerize.sh=

* Load a pickled learner
Place yourself in the same directory as =learner.py=.
#+begin_src python
import pickle

# Go to the learner.py location
from learners.learner import Learner

learner = Learner()
learner = pickle.load(open("../lesson/learner.pkl", "rb"))
report = learner.report
#+end_src

