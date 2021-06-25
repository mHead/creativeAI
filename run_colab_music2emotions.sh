#!/bin/bash
HOME=${HOME}
#repo_root_Legion="/home/mtesta/creativeAI"
repo_root_colab="/content/creativeAI"
code="main.py"


cd creativeAI/ || exit
echo "calling ${code}, with repo_root: ${repo_root_colab}, code root: ${repo_root_colab}/creativeAI"


python3 ${code} --verbose --repo_root ${repo_root_colab}

