#!/bin/bash

#repo_root_Legion="/home/mtesta/creativeAI"
repo_root_colab="/contents/creativeAI"
code="main.py"


cd code_root/ || exit
echo "calling ${code}, with repo_root: ${repo_root_colab}"


python3 ${code} --verbose --repo_root ${repo_root_colab}

