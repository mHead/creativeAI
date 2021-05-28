#!/bin/bash

#repo_root_Legion="/home/mtesta/creativeAI"
repo_root_local="/Users/head/Documents/GitHub/creativeAI"
code="main.py"
cd code_root/ || exit
echo "calling ${code}, with repo_root: ${repo_root_local}"


python3 ${code} --verbose --repo_root ${repo_root_local}
