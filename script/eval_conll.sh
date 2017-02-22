#!/bin/bash

awk '{if(length($0)==0){print}else{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8}}' $1 > $1.gold
awk '{if(length($0)==0){print}else{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$9"\t"$10}}' $1 > $1.system
awk '{if(length($0)==0){print}else{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t_"}}' $1 > $1.gold.unl
awk '{if(length($0)==0){print}else{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$9"\t_"}}' $1 > $1.system.unl


root=`dirname $0`;
#perl $root/eval.pl -g $1.gold -s $1.system
#perl $root/eval.pl -g $1.gold.unl -s $1.system.unl
perl $root/eval.pl -g $1.gold -s $1.system | egrep 'Unlabeled' | awk '{print $10}'

rm $1.gold $1.system $1.gold.unl $1.system.unl
