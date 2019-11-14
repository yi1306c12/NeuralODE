SCRIPT=$1
X=$2
SAVEPATH=$3
DATE=$(date +%Y%m%d_%H%M%S)

git rev-parse HEAD
#seq -w $X | xargs -P0 -IXXX python3 $SCRIPT --save_path $SAVEPATH"/"$DATE"/"XXX
mkdir -p $SAVEPATH"/"$DATE
seq $X | parallel --no-notice --joblog $SAVEPATH"/"$DATE"/parallel.log" --eta python3 $SCRIPT --save_path $SAVEPATH"/"$DATE"/"{#} --n_steps 10000