#!/bin/bash

pHrange=(3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0)

rm -f results.txt

for pH in ${pHrange[*]}; do

  cd ${pH}/NpT

  echo Analyzing pH=${pH} ...
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o total.xvg -b 400 -ref "name WN" -sel "name P2" 

  python3 ../../../scripts/statistical_analysis.py -f total.xvg -min 100 -o stat.dat -eq

  DOP=$(awk -F "," '{if(NR==2) print $2}' stat.dat)
  ERR=$(awk -F "," '{if(NR==2) print $3}' stat.dat)
  echo ${pH}  ${DOP}  ${ERR} >> ../../results.txt

  cd ../..
done

python3 ../scripts/plot_results.py -f results.txt
