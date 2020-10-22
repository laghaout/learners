The README is availabe as a <a href='README.ipynb'>Jupyter notebook</a>.

Log in to the container to work interactively:
```
docker run -it -w /home -v $(pwd):/home learners bash
```

Run a predefined learner from the current directory `$(pwd)`:
```
docker run -it -w /home -v $(pwd):/home learners python main.py $1
```