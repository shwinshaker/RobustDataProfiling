##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

# ---------------
trim() {
  local s2 s="$*"
  until s2="${s#[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  until s2="${s%[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  echo "$s"
}

config=$1 
[[ -z $config ]] && config="config.yaml"

# gpu_id='5'
# gain=1.0
# echo $gain" - "$gpu_id
# sed -i "s/^gain: [0-9]\.[0-9]/gain: $gain/g" config.yaml
# # s=${gain//\./_}
# # echo $s
# sed -i "s/^gpu_id: [0-9]/gpu_id: $gpu_id/g" config.yaml
# # ./launch.sh

python config.py -c $config
[[ $? -ne 0 ]] && echo 'exit' && exit 2
checkpoint=$(cat tmp/path.tmp)
path="checkpoints/$checkpoint"
echo $path
cp $config $path"/config.yaml"
cp main.py $path
cp robustCertify.py $path
cp config.py $path
cp -r src $path
subset_path="$(grep '^train_subset_path' $config | awk '{print$2}' | sed -e "s/^'//" -e "s/'$//")"
if [ ! -z $subset_path ]; then
    cp $subset_path $path
fi
weight_path="$(grep '^alpha_sample_path' $config | awk '{print$2}' | sed -e "s/^'//" -e "s/'$//")"
if [ ! -z $weight_path ]; then
    cp $weight_path $path
fi
cur=$(pwd)
cd $path
if [ $(cat config.yaml | grep "^test:" | awk '{print$2}') == 'True' ]; then
    python main.py | tee train.out
else
    python main.py > train.out 2>&1 &
fi
pid=$!
gpu_id=$(trim $(grep 'gpu_id' config.yaml | awk -F ':' '{print$2}'))
echo "[$pid] [$gpu_id] [Path]: $path"
echo "s [$pid] [$gpu_id] $(date) [Path]: $path" >> $cur/log.txt
cd $cur
