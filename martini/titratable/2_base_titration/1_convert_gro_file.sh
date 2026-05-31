#!/bin/bash

run="python3 ../scripts/convert_gro_to_titratable.py"

$run -f standard.gro -o temp.gro -sel "name P2" -bead base 
$run -f temp.gro -o start.gro -sel "name WN" -bead water

rm -f temp.gro
