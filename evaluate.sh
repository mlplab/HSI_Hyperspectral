#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
dataset="Harvard"
concat="False"
model_name=("HSCNN DeepSSPrior HyperReconNet Ghost")
block_num=9
ratios=(2 3 4)
modes=("mix1 mix2")


while getopts d:c:m:b: OPT
do
    echo "$OPTARG"
    case $OPT in
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        b) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $dataset
echo $concat
echo $block_num


model_name=( `echo $model_name | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for name in $model_name[@]; do
    if [ $name = "Ghost" ]; then
        for ratio in $ratios[@]; do
            for mode in $modes[@]; do
                echo $mode
                python evaluate_reconst_sh.py -d $dataset -c $concat -b $block_num -m $name -r $ratio -md $mode
            done
        done
    else
        python evaluate_reconst_sh.py -d $dataset -c $concat -b $block_num -m $name
    fi
done
