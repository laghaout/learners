#!/bin/sh

CONTAINER=learners
PACKAGE=mlo

rm ./*~
rm .gitignore~
rm -rf ./$CONTAINER/*~
rm -rf ./$CONTAINER/lesson/
rm -rf ./lesson/
rm -rf ./__pycache__/
rm -rf ./$CONTAINER/__pycache__/
rm -rf ./.ipynb_checkpoints/
rm -rf ./build/ ./dist/ ./$PACKAGE.egg-info/

