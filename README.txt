#+TITLE: README

* Dockerize
To dockerize,
1. run =cp .env_template .env= and edit the environment variables if needed then
2. run =sh dockerize.sh=.

* Load a pickled learner
Place yourself in the same directory as =learner.py= then run the following:
#+begin_src python
import pickle
from learners.learner import Learner

learner = Learner(lesson_dir=None)
learner = pickle.load(open('../lesson/learner.pkl', 'rb'))
report = learner.report
#+end_src
Note that the path =../lesson/learner.pkl= may need some editing.
