#!/bin/bash 

pHrange=(3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0)

for pH in ${pHrange[*]}; do

  rm -rf ${pH}
  mkdir ${pH}
  cd ${pH}
  
  sed "s/<value>/${pH}/g" ../system.top > system.top 
  
  mkdir min
  mkdir eq
  mkdir NpT
  
  cd min 
  
  gmx grompp -f ../../min.mdp -c ../../start.gro -p ../system.top -maxwarn 1 >> grompp.out 2>&1
  gmx mdrun  >> mdrun.out 2>&1
  
  cd ../eq
  
  gmx grompp -f ../../NVT.mdp -c ../min/confout.gro -p ../system.top >> grompp.out 2>&1
  gmx mdrun  -nsteps 200000 >> mdrun.out 2>&1
  
  cd ../NpT
  
  gmx grompp -f ../../NpT.mdp -c ../eq/confout.gro -p ../system.top >> grompp.out 2>&1
  gmx mdrun  -nsteps 2000000 >> mdrun.out 2>&1
  
  cd ../..

done
