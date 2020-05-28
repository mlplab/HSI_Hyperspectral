#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
dataset="Harvard"
model_name=("HSCNN HSI_Network Attention_HSI_None Attentio_HSI_GAP Attention_HSI_GVP")


while getopts m: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) dataset=$OPTARG ;;
        m) model_name=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


model_name=( `echo $model_name | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for name in $model_name[@]; do
    python train_sh.py -b $batch_size, -e $epoch -d $dataset -m $name
done


# python train_sh.py -b $batch_size -e $epoch -d $dataset -m $model_name