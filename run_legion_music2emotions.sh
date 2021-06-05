#!/bin/bash
#SBATCH --job-name=music2emotion
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --partition=cuda
#SBATCH --mem=90GB
#SBATCH --time=72:00:00
#SBATCH --output=mus2emo_%j_out.txt
#SBATCH --error=mus2emo_%j_err.txt

#load modules
ml nvidia/cudasdk/10.1
ml intel/python/3/2019.4.088

cd /home/mtesta/ || exit

# %% Dropbox Uploader
#git clone https://github.com/thatbrguy/Dropbox-Uploader.git

#cd ./Dropbox-Uploader/ || exit
#chmod +x dropbox_uploader.sh

#dropbox_uploader.sh
#echo "wZJ15QXvZzMAAAAAAAAAAZabVRJ84LA_mJiIY2-khYD9ZHLY9Ot_1I54QXvaJA_X" > token.txt
#dropbox_uploader.sh
# %%
# if exists remove repo and reclone updated
rm -Rf creativeAI/
git clone https://github.com/mHead/creativeAI.git

cd ./creativeAI/code_root || exit

repo_root_legion="/home/mtesta/creativeAI"
code="main.py"

echo "installing modules"

echo "pip install Tornado --user"
pip install Tornado --user
echo "pip install tensorflow --user"
pip install tensorflow --user


echo "calling ${code}, with repo_root: ${repo_root_legion}. PWD is: $PWD"


python3 ${code} --verbose --repo_root ${repo_root_legion} --legion
#/Users/head/PycharmProjects/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav