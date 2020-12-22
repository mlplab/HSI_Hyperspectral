#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
datasets=("CAVE" "Harvard")
concat="False"
model_name=("HSCNN HyperReconNet DeepSSPrior")
block_num=9
sRatio=(2 3 4)


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        s) sRatio=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $batch_size
echo $epoch
echo $datasets
echo $block_num
echo $sRatio


model_name=( `echo $model_name | tr ' ' ' '` )


for name in $model_name[@]; do
    echo $name
done


for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        if [[ $name = "Ghost" ]]; then
            for s in $sRatio[@]; do
                python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -s $s
            done
        else
            python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num
        fi
    done
done
