#!/bin/bash
#SBATCH --job-name=music2emotion
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --partition=cuda
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=mus2emo_%j_out.txt
#SBATCH --error=mus2emo_%j_err.txt

ml purge
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
#rm -Rf creativeAI/
#git clone https://github.com/mHead/creativeAI.git

echo "cloning creativeAI repository DONE!"
echo "moving the dataset into the data_root inside the repository"
mkdir -p /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav

if [ -z "$(ls -A /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav)" ]; then
  echo "The folder: creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav is empty... going to move the dataset into the data_root inside the repository"
  cp -R /home/mtesta/data/clips_30seconds_preprocessed_BIG /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav
else
  echo "The raw files are already inside the repo path"
fi

cd ./creativeAI || exit
#chmod +x run_legion_music2emotions.sh

cd ./creativeAI || exit

repo_root_legion="/home/mtesta/creativeAI"
code="main.py"

echo "installing modules"
pip install Tornado --user
pip install tensorflow --user
pip install ffmpeg --user
pip install pydub --user
pip install scipy --user
pip install torch --user
pip install torchaudio --user
pip install torchvision --user
pip install tensorboard --user


echo "calling ${code}, with repo_root: ${repo_root_legion}. PWD is: $PWD"


python3 ${code} --verbose --legion --repo_root ${repo_root_legion}
#/Users/head/PycharmProjects/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav