#!/usr/bin/env python
# coding: utf-8


### #NB!!!!! 
## FAR FROM FLAWLESS !! You may need to correct the output pdb file a bit manually



import os
import re
import MDAnalysis as md


def get_pdb_line (pdb, sel):
    L = []
    for idx, line in enumerate(pdb):
        if re.match('^ATOM', line):
            atom = line.split()[2]
            if atom == sel :
                L.append(line)
    return L  


######### INPUT #######################################
mapping = open('mapping.ndx', 'r').readlines() #an index file with the mapping, in terms of atom names
pdb = open('MC3.pdb', 'r').readlines() #a pdb of the molecule in question
#######################################################

######## OUTPUT #######################################
mapp = open('MC3_map.pdb', 'w') #Output mapped pdb file
#######################################################



for idx, line in enumerate(mapping):
    #print (line.split()[0])
    if re.match('\[', line.split()[0]):
        bead_name = line.split()[0].strip('\[\]')
        bead      = line.split()[0]
        atoms     = mapping[idx+1]
        
        if bead_name != 'None':
            mapp.write('{}\n'.format(bead))
            atoms_taken = []
            for at in atoms.split(' '): 
                a = at[:4]
                if a in atoms_taken:
                    #Checks if there is dublicates
                    #The CHARMM atom names sometimes are longer than 4 spaces, leading to eg several
                    #hydrogen atoms named the same in the pdb but not in the gro file
                    continue
                else:
                    atoms_taken.append(a)
                    line = get_pdb_line(pdb, a)
                    for l in line:
                        mapp.write(f'{l}')
mapp.close()

