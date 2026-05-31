#!/bin/bash

pHrange=(3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0)

rm -f results_G?.txt results_radius_of_gyration.txt

for pH in ${pHrange[*]}; do

  cd ${pH}/NpT

  echo Analyzing pH=${pH} ...
  # Calculate degree of deprotonation
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o G1.xvg -b 400 -ref "name WN NG2 NG3 NG4 NG5" -sel "name NG1" 
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o G2.xvg -b 400 -ref "name WN NG1 NG3 NG4 NG5" -sel "name NG2" 
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o G3.xvg -b 400 -ref "name WN NG1 NG2 NG4 NG5" -sel "name NG3" 
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o G4.xvg -b 400 -ref "name WN NG1 NG2 NG3 NG5" -sel "name NG4" 
  python3 ../../../scripts/degree_of_deprot.py -f traj_comp.xtc -s topol.tpr -o G5.xvg -b 400 -ref "name WN NG1 NG2 NG3 NG4" -sel "name NG5" 

  for i in $(seq 1 5); do
    python3 ../../../scripts/statistical_analysis.py -f G${i}.xvg -min 100 -o stat_G${i}.dat -eq
    DOP=$(awk -F "," '{if(NR==2) print $2}' stat_G${i}.dat)
    ERR=$(awk -F "," '{if(NR==2) print $3}' stat_G${i}.dat)
    echo ${pH}  ${DOP}  ${ERR} >> ../../results_G${i}.txt
  done

  # Calculate radius of gyration
  echo 2 | gmx polystat -f traj_comp.xtc > polystat.out 2>&1 
  gmx analyze -f polystat.xvg -ee > analyze.out 2>&1
  err=$(grep 'Set   2:' analyze.out | awk '{print $4}')
  val=$(grep 'SS2' analyze.out | awk '{print $2}')
  echo ${pH} ${val} ${err} >> ../../results_radius_of_gyration.txt

  cd ../..
done

# Plot results
python3 ../scripts/plot_ppi_degree_of_deprot.py
python3 ../scripts/plot_radius_of_gyration.py
