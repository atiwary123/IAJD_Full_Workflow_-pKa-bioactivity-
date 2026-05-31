#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import MDAnalysis as md
from MDAnalysis.analysis import align
from tqdm.notebook import tqdm
from MDAnalysis.lib import transformations, mdamath
import sys
import scipy.optimize
import itertools
import tqdm.notebook as tqdm
from MDAnalysis.analysis.distances import dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#from termcolor import colored
import os
import subprocess
import re

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization


##########################################################################
## A bunch of functions to easier parameterize small fragments using    ##
## python. Import these in a notebook and use for iterative optimization##
## of the parameters                                                    ##
##########################################################################


mdp_loc = '/data1/lisbeth/Params/FRAGMENTS/AA_SIMS_WATER_IONS/MDPs'


def draw_fragment (SMILE):
    '''Function to draw the fragment given a SMILE string. 
    It will also show the fragment in 3D interactively.'''
    m = Chem.MolFromSmiles(SMILE)
    fig = Draw.MolToMPL(m)
    AllChem.EmbedMolecule(m)
    m
    return (m)



def write_SASA_CG (dir_out, bead_names, bead_sizes):
    '''fuction to write the CG vdwradii.dat file for calculating SASA.'''
    dir_sizes = {'R':0.264,
                 'S':0.230,
                 'T':0.191}
    
    sasa = open(dir_out+'/vdwradii.dat', 'w')
    sasa.write(';CG van der walls radii\n')
    for idx, bead in enumerate(bead_names):
        size = dir_sizes[bead_sizes[idx]]
        sasa.write('???  {}    {}\n'.format(bead, size))
    sasa.close()
    return


def write_initial_CGitp (resname, bead_names, bead_sizes, bead_types, bead_charges, D_bond, D_ang):
    '''Function to write the inital CG itp file, given the defined angles and bonds.
    No dihedral included here but can be added.
    The fc is by default 500 and 25 for bonds and angles, respectively. '''
    mass_dir = {'R':72,
            'S':54,
            'T':36}
    initial = open('initial_CG.itp', 'w')
    initial.write('[ moleculetype ]\n')
    initial.write('{}  1'.format(resname))
    initial.write('\n')
    initial.write('[ atoms ]\n')
    initial.write('; nr type resnr residue atom cgnr charge mass\n')
    for idx, bead in enumerate(bead_names):
        initial.write("{}    {}    {}    {}    {}    {}    {}    {}\n"
                      .format(idx+1, bead_types[idx], 0, resname, bead, idx+1, bead_charges[idx], mass_dir[bead_sizes[idx]] ))
    initial.write('\n')
    initial.write('[ bonds ]\n')
    initial.write('; i  j  funct length\n')
    for i in D_bond:
        #print (D_bond[i])
        initial.write('{}    {}    {}    {}    {}\n'
                      .format(D_bond[i][0], D_bond[i][1], 1,  0.47, 5000))
    initial.write('\n')
    initial.write('[ angles ]\n')
    initial.write(';  i  j  k   funct   angle   force.c.\n')
    for i in D_ang:
        #print (D_ang[i])
        initial.write('{}    {}    {}    {}    {}    {}\n'
                      .format(D_ang[i][0], D_ang[i][1], D_ang[i][2], 2,  130, 25))
    initial.close()
    return

def write_initial_CGitp_old (resname, bead_names, bead_sizes, bead_types, D_bond, D_ang):
    '''Function to write the inital CG itp file, given the defined angles and bonds. 
    No dihedral included here but can be added.
    The fc is by default 500 and 25 for bonds and angles, respectively. '''
    initial = open('initial_CG.itp', 'w')
    initial.write('[ moleculetype ]\n')
    initial.write('{}  1'.format(resname))
    initial.write('\n')
    initial.write('[ atoms ]\n')
    initial.write('; nr type resnr residue atom cgnr charge mass\n')
    for idx, bead in enumerate(bead_names):
        initial.write("{}    {}    {}    {}    {}    {}    {}    {}\n"
                      .format(idx+1, bead_types[idx], 0, resname, bead, idx+1, 0, bead_sizes[idx] ))
    initial.write('\n')
    initial.write('[ bonds ]\n')
    initial.write('; i  j  funct length\n')
    for i in D_bond:
        #print (D_bond[i])
        initial.write('{}    {}    {}    {}    {}\n'
                      .format(D_bond[i][0], D_bond[i][1], 1,  0.47, 5000))
    initial.write('\n')
    initial.write('[ angles ]\n')
    initial.write(';  i  j  k 	funct 	angle 	force.c.\n')
    for i in D_ang:
        #print (D_ang[i])
        initial.write('{}    {}    {}    {}    {}    {}\n'
                      .format(D_ang[i][0], D_ang[i][1], D_ang[i][2], 2,  130, 25))
    initial.close()
    return



def map_aa2cg (u, resname, bead_assignments, bead_names, bead_sizes, AA_traj, GRO):
    '''Provide the universe along with the lists containing the resnames of eg the lipids,
    the bead_assignements (which atoms goes into which bead), and a list of the bead_names'''
    lip_resnames            = []
    lip_bead_assignments    = []
    lip_bead_names          = []
    
    
    lip_resnames.append(resname)
    lip_bead_assignments.append(bead_assignments)
    lip_bead_names.append(bead_names)
    
    #Mapping the trajectory, generating the mapping index files
    bead_agg = []
    for i, molecule in enumerate(lip_resnames):
        #print ('resname {}'.format(molecule)
        targets = u.select_atoms('resname {}'.format(molecule)).residues
        #print ('targets')
        #print (targets)
        for j in range(len(targets)):
            target = targets[j].atoms
            #print (target)
            for bead in lip_bead_assignments[i]:
                #print (bead)
                agg_whole = u.select_atoms('name empty')
                for atom in bead:
                    #print (bead)
                    agg = target.select_atoms('name {}'.format(atom))
                    agg_whole += agg
                    #print (agg_whole)
                bead_agg.append(agg_whole)
    with md.selections.gromacs.SelectionWriter('index.ndx', mode='w') as ndx:
        for idx, agg in enumerate(bead_agg):
            ndx.write(agg)
    #### Create VDWRadii to use with SASA
    sizedict = {
                 "R": 0.264,
                 "S": 0.230,
                 "T": 0.191
    }
    with open('/data1/lisbeth/Params/Pzifer_alcoholamine/7_SASA_longer_bonds/vdwradii_CG.dat') as filein:
        vdw = filein.readlines()
    vdw = vdw[:15]
    for idx, beadname in enumerate(bead_names):
        vdw.append(f'???  {beadname}   {sizedict[bead_sizes[idx]]}\n')
    with open('vdwradii_CG.dat', 'w+') as file_out:
        for line in vdw:
            file_out.write(line)
    for atom in u.atoms:
        atom.name = 'H'
    u.atoms.write('all_hydrogen.gro')
    os.system('cp /data1/lisbeth/Params/Pzifer_alcoholamine/7_SASA_longer_bonds/vdwradii_AllAtom_rowland.dat .')
    
    #Using gromacs to map the AA simulation into CG
    p = subprocess.Popen("gmx traj -f {} -s all_hydrogen.gro -n index.ndx -com -ng {} -oxt cg.gro".format( GRO, len(bead_agg))
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('\n'.join(map(str,list(range(len(bead_agg))))))
    p.wait()

    p = subprocess.Popen("gmx traj -f {} -s all_hydrogen.gro -n index.ndx -com -ng {} -oxt cg.xtc".format( AA_traj, len(bead_agg))
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('\n'.join(map(str,list(range(len(bead_agg))))))
    p.wait()

    ### Fix names in GRO for easier parameterization
    names = []
    for i, molecule in enumerate(lip_resnames):
        targets = u.select_atoms('resname {}'.format(molecule)).residues
        gro_names    = lip_bead_names[i]*len(targets)
        names += gro_names

    with open('cg.gro') as cggro_in:
        gro_file = cggro_in.readlines()
    for idx, line in enumerate(gro_file[2:-1]):
        gro_file[2+idx]=line.replace(line[12:15],names[idx])
    with open('cg_good.gro', 'w+') as file_out:
        for line in gro_file:
            file_out.write(line)
    return



def run_SASA(name, target, resname, dir_out, bead_names, bead_sizes):
    ''''Give name of simulation, AA for atomistic, resname in the gro file, and an out_put directory, in here folders will be made'''
    try:
        dir_writing = '{}/{}'.format(dir_out, name)
        os.mkdir(dir_writing)
    except FileExistsError:
        pass
    gro = target[0]
    xtc = target[1]
    u = md.Universe(gro, xtc)

    ## Create index file and spit out a gro
    ## Copy vdwradii to target directory.
    if name == 'AA':
        tgt = u.select_atoms('resname {}'.format(resname)).residues[0].atoms.select_atoms('all')
        subprocess.call('cp /data1/lisbeth/Params/Pzifer_alcoholamine/7_SASA_longer_bonds/vdwradii_AllAtom_rowland.dat {}/vdwradii.dat'.format(dir_writing)
                    , shell=True)
    else:
        tgt = u.select_atoms('resname {}'.format(resname)).residues[0].atoms.select_atoms('all')
        #subprocess.call('cp /data1/lisbeth/Params/Pzifer_alcoholamine/7_SASA_longer_bonds/vdwradii_CG.dat {}/vdwradii.dat'.format(dir_writing)
        #            , shell=True)
        write_SASA_CG(dir_writing, bead_names, bead_sizes)
    
    tgt.write("{}/index.ndx".format(dir_writing), mode="w", name= 'TGT')
    tgt.atoms.write(f"{dir_writing}/gro.gro")
    
    ## Calculate SASA & connoly surface
    os.chdir(dir_writing)
    print ('entering {}'.format(dir_writing))
    p = subprocess.Popen(f"gmx sasa -f ../{xtc} -s ../{gro} -n {dir_writing}/index.ndx -ndots 4800 -probe 0.191 -or {dir_writing}/resarea_SASA.xvg -o {dir_writing}/SASA.xvg "
                , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('TGT')
    p.wait()

    p = subprocess.Popen(f"gmx sasa -s gro.gro -o temp.xvg -probe 0.191 -ndots 240 -q surface.pdb"
            , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('System')
    p.wait()
    os.chdir('../')
    return




def prepare_setup_BL(itp_loc, mdp_loc, lipids='POPC DPPC DIPC DPSM CHOL DBPC DOPC'):
    '''Prepare setup of Bilayer system. Give itp_loc, mdp_loc and a list of the lipids in the system'''
    ###fix topology
    #itpDir='/data1/lisbeth/Martini_ITPs/DEV15/martini3-lipids-master-top/top'
    itpDir='/data1/lisbeth/Martini_ITPs/'

    with open('topol.top') as topinput:
        top = topinput.readlines()
    #top[0]= f'#include "{itpDir}/martini_v3.0.0.itp"\n#include "{itpDir}/martini_v3.0_ffbonded_dev_v15.itp"\n#include "{itp_loc}"\n#include "{itpDir}/martini_v3.0_phospholipids_PC_dev_v15.itp"\n#include "{itpDir}/martini_v3.0.0_solvents_v1.itp"\n#include "{itpDir}/martini_v3.0.0_ions_v1.itp"\n'
    top[0]= f'#include "{itpDir}/martini_v3.0.0.itp"\n#include "{itpDir}/Martini_v1_tails.itp"\n#include "{itp_loc}"\n#include "{itpDir}/martini_v3.0.0_phospholipids_v1.itp"\n#include "{itpDir}/martini_v3.0.0_solvents_v1.itp"\n#include "{itpDir}/martini_v3.0.0_ions_v1.itp"\n'
    with open('topol.top', 'w+') as topout:
        for line in top:
            topout.write(line)
    
    ### Neutralize
    #print('gmx grompp -f {}/min.mdp -c mem.gro -p topol.top -o memion.tpr -maxwarn 1'.format(mdp_loc))
    subprocess.call('gmx grompp -f {}/min.mdp -c mem.gro -p topol.top -o memion.tpr -maxwarn 1'.format(mdp_loc)
                , shell=True)
    #print ('gmx genion -s memion.tpr -o memion.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -neutral')
    p = subprocess.Popen('gmx genion -s memion.tpr -o memion.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -neutral'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()
    
    ### add 150mM NaCl
    Waterline =  top[-1].split(' ')
    Waternumber = int( Waterline[-1].rstrip() )
    naclNUM = int((0.15 * Waternumber*4)/55.5)
    subprocess.call('gmx grompp -f {}/min.mdp -c memion.gro -p topol.top -o memion_2.tpr -maxwarn 1'.format(mdp_loc)
                , shell=True)
    p = subprocess.Popen('gmx genion -s memion_2.tpr -o memion_2.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -np {} -nn {}'.format(naclNUM, naclNUM)
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()
    
    ### Minimize the system
    subprocess.call("gmx grompp -f {}/min.mdp -c memion_2.gro -p topol.top -o m.tpr -maxwarn 1".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm m -nt 5"
                    , shell = True)

    ### Create the index file
    gro = 'm.gro'
    u = md.Universe(gro)
    lipidsagg=u.select_atoms('resname {}'.format(lipids))
    solventagg=u.select_atoms('not resname {}'.format(lipids))
    lipidsagg.write("index.ndx", mode="w", name= 'Lipids')
    solventagg.write("index.ndx", mode="a", name= 'Solvent')
    u.atoms.write("index.ndx", mode="a", name= 'System')
    
    ### Relax the system
    subprocess.call("gmx grompp -f {}/rel_310.mdp -c m.gro -p topol.top -o r.tpr -n index.ndx -maxwarn 1".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm r -nt 5 -rdd 1.6"
                , shell = True)
    
    ### Prepare the simulation
    subprocess.call("gmx grompp -f {}/prod_310.mdp -c r.gro -p topol.top -o p.tpr -n index.ndx -maxwarn 1".format( mdp_loc)
                    , shell = True)
    ### Prepare the simulation
    subprocess.call("gmx mdrun -v -deffnm p  -nt 5 -rdd 1.6".format(mdp_loc)
                    , shell = True)
    return


def re_run_BL (resname, mdp_loc='/data1/lisbeth/Params/BILAYERS/MDPs', maxwarn=1, lipids='POPC DPPC DIPC DPSM CHOL DBPC DOPC'):
    '''Re-runs min, and short production run. By default it uses the mdp files
       located here: /data1/lisbeth/Params/BILAYERS/MDPs.'''

    ### Minimize the system
    subprocess.call("gmx grompp -f {}/min.mdp -c memion_2.gro -p topol.top -o m.tpr -maxwarn 1".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm m -nt 5"
                    , shell = True)

    ### Create the index file
    gro = 'm.gro'
    u = md.Universe(gro)
    lipidsagg=u.select_atoms('resname {}'.format(lipids))
    solventagg=u.select_atoms('not resname {}'.format(lipids))
    lipidsagg.write("index.ndx", mode="w", name= 'Lipids')
    solventagg.write("index.ndx", mode="a", name= 'Solvent')
    u.atoms.write("index.ndx", mode="a", name= 'System')
    
    ### Relax the system
    subprocess.call("gmx grompp -f {}/rel_310.mdp -c m.gro -p topol.top -o r.tpr -n index.ndx -maxwarn 1".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm r -nt 5 -rdd 1.6"
                , shell = True)
    
    ### Prepare the simulation
    subprocess.call("gmx grompp -f {}/prod_310.mdp -c r.gro -p topol.top -o p.tpr -n index.ndx -maxwarn 1".format( mdp_loc)
                    , shell = True)
    ### Prepare the simulation
    subprocess.call("gmx mdrun -v -deffnm p  -nt 5 -rdd 1.6".format(mdp_loc)
                    , shell = True)
    return 



def prepare_setup_water(mdp_loc):
    '''Function to setup and run simulation of the molecule in water'''
    u=md.Universe('cg_good.gro', '../cg.xtc')
    sel =u.select_atoms('all')
    resname = np.unique(sel.resnames)[0]
    #itpDir='/data1/lisbeth/Martini_ITPs/DEV15/martini3-lipids-master-top/top'
    itpDir='/data1/lisbeth/Martini_ITPs'
    u.select_atoms('resname {}'.format(resname)).residues[0].atoms.write('res.gro')
    
    ##Write topol.top
    #topinput = open('topol.top', 'w')
    #topinput.write('#include "{}/martini_v3.0.0.itp"\n'.format(itpDir))
    #topinput.write('#include "{}/martini_v3.0_ffbonded_dev_v15.itp"\n'.format(itpDir))
    #topinput.write('#include "initial_CG.itp"\n')
    #topinput.write('#include "{}/martini_v3.0.0_solvents_v1.itp"\n'.format(itpDir))
    #topinput.write('#include "{}/martini_v3.0.0_ions_v1.itp"\n'.format(itpDir))
    #topinput.write('[system]\n')
    #topinput.write('one molecule\n')
    #topinput.write("\n")
    #topinput.write("[ molecules ]\n")
    #topinput.write('{}   1\n'.format(resname))    
    #topinput.close()


    #Write topol.top
    topinput = open('topol.top', 'w')
    topinput.write('#include "{}/martini_v3.0.0.itp"\n'.format(itpDir))
    topinput.write('#include "{}/Martini_v1_tails.itp"\n'.format(itpDir))
    topinput.write('#include "initial_CG.itp"\n')
    topinput.write('#include "{}/martini_v3.0.0_solvents_v1.itp"\n'.format(itpDir))
    topinput.write('#include "{}/martini_v3.0.0_ions_v1.itp"\n'.format(itpDir))
    topinput.write('[system]\n')
    topinput.write('one molecule\n')
    topinput.write("\n")
    topinput.write("[ molecules ]\n")
    topinput.write('{}   1\n'.format(resname))    
    topinput.close()



    os.system('cp topol.top topol_old.top')
    
    ###
    ### editconf
    subprocess.call('gmx editconf -f res.gro -o box.gro -box 5 5 5 -bt cubic'
                , shell=True)
    
    ###
    ### gmx solvate
    subprocess.call(f'gmx solvate -cp box.gro -cs {mdp_loc}/water.gro -o watered.gro -p topol.top'
                , shell=True)
    
    u = md.Universe('watered.gro')
    Waternumber = int( (u.select_atoms('resname W w PW').resids.shape[0]) / 3 )


    ### Neutralize
    subprocess.call('gmx grompp -f {}/min.mdp -c watered.gro -p topol.top -o memion.tpr -maxwarn 1'.format(mdp_loc)
                , shell=True)
    p = subprocess.Popen('gmx genion -s memion.tpr -o memion.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -neutral'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()
   
    ### add 150mM NaCl
    naclNUM = int((0.15 * Waternumber*4)/55.5)
    subprocess.call('gmx grompp -f {}/min.mdp -c memion.gro -p topol.top -o memion_2.tpr -maxwarn 4'.format(mdp_loc)
                , shell=True)
    p = subprocess.Popen('gmx genion -s memion_2.tpr -o memion_2.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -np {} -nn {}'.format(naclNUM, naclNUM)
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()

    ### Minimize the system
    subprocess.call("gmx grompp -f {}/min.mdp -c memion_2.gro -p topol.top -o m.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm m "
                    , shell = True)
    ### Relax the system
    subprocess.call("gmx grompp -f {}/eq.mdp -c m.gro -p topol.top -o r.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm r  -nt 10"
                , shell = True)
    
    ### Prepare the simulation
    subprocess.call("gmx grompp -f {}/run.mdp -c r.gro -p topol.top -o p.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    ### Prepare the simulation
    subprocess.call("gmx mdrun -v -deffnm p -nt 10".format(mdp_loc)
                    , shell = True)

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.xtc -pbc mol'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.gro -pbc mol -b 0 -e 0'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()
    return 

def prepare_setup_water_old(mdp_loc):
    '''Function to setup and run simulation of the molecule in water'''
    u=md.Universe('cg_good.gro', '../cg.xtc')
    sel =u.select_atoms('all')
    res = u.select_atoms('all').resids[0]
    u.select_atoms('resid {}'.format(res)).write('molecule.pdb')
    resname = np.unique(sel.resnames)[0]
    
    u.select_atoms('resname {}'.format(resname)).residues[0].atoms.write('res.gro')
    
    #Write topol.top
    topinput = open('topol.top', 'w')
    topinput.write('#include "{}/martini_v3.0.0.itp"\n'.format(mdp_loc))
    topinput.write('#include "initial_CG.itp\n')
    topinput.write('#include "{}/martini_v3.0.0_solvents_v1.itp"\n'.format(mdp_loc))
    topinput.write("\n")
    topinput.write('[system]\n')
    topinput.write('one molecule\n')
    topinput.write("\n")
    topinput.write("[ molecules ]\n")
    topinput.write('{}   1\n'.format(resname))    
    topinput.close()
    
    os.system('cp topol.top topol_old.top')
    
    ###
    ### editconf
    subprocess.call('gmx editconf -f molecule.pdb -o box.gro -box 5 5 5 -bt cubic'
                , shell=True)
    
    ###
    ### gmx solvate
    subprocess.call(f'gmx solvate -cp box.gro -cs {mdp_loc}/water.gro -o watered.gro -p topol.top'
                , shell=True)
    
    ### Neutralize
    subprocess.call('gmx grompp -f {}/min.mdp -c watered.gro -p topol.top -o memion.tpr -maxwarn 1'.format(mdp_loc)
                , shell=True)
    p = subprocess.Popen('gmx genion -s memion.tpr -o memion.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -neutral'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()
    
    ### Minimize the system
    subprocess.call("gmx grompp -f {}/min.mdp -c memion.gro -p topol.top -o m.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm m "
                    , shell = True)
    ### Relax the system
    subprocess.call("gmx grompp -f {}/eq.mdp -c m.gro -p topol.top -o r.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm r  -nt 10"
                , shell = True)
    
    ### Prepare the simulation
    subprocess.call("gmx grompp -f {}/run.mdp -c r.gro -p topol.top -o p.tpr -maxwarn 5".format(mdp_loc)
                    , shell = True)
    ### Prepare the simulation
    subprocess.call("gmx mdrun -v -deffnm p -nt 10".format(mdp_loc)
                    , shell = True)

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.xtc -pbc mol'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.gro -pbc mol -b 0 -e 0'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()
    return 




###Define function to return a dihedral object for a specified set of 4 beads
def beadstodihedral(u, resid1, resname1, name1, 
                       resid2, resname2, name2, 
                       resid3, resname3, name3, 
                       resid4, resname4, name4):
    selection = 'resname {} and resid {} and name {}'    
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid2, name2)
    C = selection.format(resname3, resid3, name3)
    D = selection.format(resname4, resid4, name4)
    psi = [A, B, C, D]
    psi_angle = sum([u.select_atoms(atom) for atom in psi])  # sum of Atoms creates an AtomGroup
    psi_angle = psi_angle.dihedral  # convert AtomGroup to Dihedral object
    
    return psi_angle

###Define function to get the angle object for a specified set of 3 beads
def beadstoangle(u, resid1, resname1, name1, 
                    resid2, resname2, name2, 
                    resid3, resname3, name3):
    selection = 'resname {} and resid {} and name {}'
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid2, name2)
    C = selection.format(resname3, resid3, name3)

    psi = [A, B, C]
    psi_angle = sum([u.select_atoms(atom) for atom in psi])  # sum of Atoms creates an AtomGroup
    psi_angle = psi_angle.angle  # convert AtomGroup to Dihedral object
    
    return psi_angle

###Define function to get the distances between 2 specified beads
def beadstodistance(u, resid1, resname1, name1,
                       resid2, resname2, name2):
    selection = 'resname {} and resid {} and name {}'
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid1, name2)
    
    Ag = u.select_atoms(A)
    Bg = u.select_atoms(B)
    dist = np.linalg.norm(Ag.positions - Bg.positions)
    #dist_arr = dist(Ag, Bg)
    
    #return dist_arr[2]
    return dist


def make_angle_bond_dicts(ang_tgts, dist_tgts, bead_names):
    D_ang = {}
    for idx, a in enumerate(ang_tgts):
        loc_ang = []
        for i in a:
            for idj, j in enumerate(bead_names):
                if i in j:
                    loc_ang.append(idj+1)
        D_ang[idx+1] = loc_ang 
    
    D_bond = {}
    for idx, a in enumerate(dist_tgts):
        loc = []
        for i in a:
            for idj, j in enumerate(bead_names):
                if i in j:
                    loc.append(idj+1)
        D_bond[idx+1] = loc 
    return D_ang, D_bond


def plot_SASA (AA_xvg, CG_xvg, resname):
    '''AA_xvg is the /path/resarea_SASA.xvg'''
    AA_Values = []
    CG_Values = []
    
    
    cols = []
    with open(AA_xvg) as f:
        for line in f:
            cols.append(line.split())
    SASA_AA = np.array(cols[25:])[:,1:][0].astype('float64')
    
    cols = []
    with open(CG_xvg) as f:
        for line in f:
            cols.append(line.split())
    SASA_CG = np.array(cols[25:])[:,1:][0].astype('float64')
    
    print('SASA AA: '+ str(SASA_AA))
    print('SASA CG: '+ str(SASA_CG))


    fig = plt.figure(figsize=(7*2,7*2.2), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)
    
    #######
    #######
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])
    ax = fig.add_subplot(gs0[1])
    names = [resname]
    x     = np.arange(len(names))  # the label locations
    width = 0.35                   # the width of the bars
    rects1 = ax.bar(0 - width/2, SASA_AA[0], width, label='Atomistic',
                    color='tab:red', yerr=SASA_AA[1], error_kw=dict(lw=2, capsize=5, capthick=2), zorder=15)
    rects2 = ax.bar(0 + width/2, SASA_CG[0], width, label='CG Mapped',
                    color='tab:blue', yerr=SASA_CG[1], error_kw=dict(lw=2, capsize=5, capthick=2), zorder=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in names], fontweight='bold', fontsize=14)
    ax.tick_params(bottom=True, labelbottom=True)
    ax.legend(loc='upper center', ncol=2)
    #ax.set_ylim(0,8)
    ax.set_xlim(-1,1)
    ax.set_ylabel('SASA (nm$^{2}$)', fontweight='bold', fontsize=14)
    ax.text(-0.05, 1.15,'a', fontweight='bold', fontsize=25,
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
    
    print ('percent difference between the models.')
    print ((1 - (SASA_CG[0]/SASA_AA[0]))*100)
    return SASA_CG, SASA_AA
    

def measure_ang_dist (dist_tgts, ang_tgts, u_aa, u_cg, bead_names, resname, stride1=100, stride2=1):
    '''Measures angles and distances - not dihedrals ! This can be added.
    dist_tgts and ang_tgts are the defined distances and angles, respectively.
    u_aa and u_cg is the mapped AA simulation and the simulated CG simulation.
    Stride1 and stride2 is for u_aa and u_cg, respectively.'''


    bins_dihed = np.arange(-181,181,2)
    bins_angl = np.arange(-1,181,2)
    bins_dist = np.arange(0,15,0.2)



    D_ang, D_bond = make_angle_bond_dicts(ang_tgts, dist_tgts, bead_names)

    ang_out_cg  = []
    ang_hist_cg = []
    ang_out_cg_wat  = []
    ang_hist_cg_wat = []
    ang_out_aa  = []
    ang_hist_aa = []
    ang_out_aa_wat  = []
    ang_hist_aa_wat = []
    
    
    
    dist_out_cg = []
    dist_hist_cg = []
    dist_out_cg_wat = []
    dist_hist_cg_wat = []
    dist_out_aa = []
    dist_hist_aa = []
    dist_out_aa_wat = []
    dist_hist_aa_wat = []
        
    for tgt in ang_tgts:
        ang_out_cg.append([])
        ang_out_aa.append([])
        ang_out_cg_wat.append([])
        ang_out_aa_wat.append([])
    for tgt in dist_tgts:
        dist_out_cg.append([])
        dist_out_aa.append([])
        dist_out_cg_wat.append([])
        dist_out_aa_wat.append([])
    
    molecules_aa = u_aa.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))
    molecules_cg = u_cg.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))
    
    print('Calculating AA  distributions...')
    for resid in tqdm.tqdm(molecules_aa.resids):
        dist= []
        angs  = []
        
        for tgt in ang_tgts:
            angs.append(beadstoangle(u_aa, resid, resname, tgt[0],
                                           resid, resname, tgt[1],
                                           resid, resname, tgt[2]))
        
        for ts in u_aa.trajectory[::stride1]:
            for idx, ang in enumerate(angs):
                ang_out_aa[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out_aa[idx].append(beadstodistance(u_aa, resid,resname,dista[0],resid,resname,dista[1]))
                
    
    for angle in ang_out_aa:
        ang_hist_aa.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out_aa:
        dist_hist_aa.append(np.histogram(distance, bins=bins_dist, density=True)[0])
    np.save('AA_angles.npy', ang_out_aa)
    np.save('AA_angles_hist.npy', ang_hist_aa)
    np.save('AA_distances.npy', dist_out_aa)
    np.save('AA_distances_hist.npy', dist_hist_aa)
    
    
    print('Calculating CG  distributions...')
    for resid in tqdm.tqdm(molecules_cg.resids):
        angs  = []
        
        for tgt in ang_tgts:
            angs.append(beadstoangle(u_cg, resid, resname, tgt[0],
                                           resid, resname, tgt[1],
                                           resid, resname, tgt[2]))
        
        for ts in u_cg.trajectory[::stride2]:
            for idx, ang in enumerate(angs):
                ang_out_cg[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out_cg[idx].append(beadstodistance(u_cg, resid,resname,dista[0],resid,resname,dista[1]))
                
    
    for angle in ang_out_cg:
        ang_hist_cg.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out_cg:
        dist_hist_cg.append(np.histogram(distance, bins=bins_dist, density=True)[0])
    
    np.save('CG_angles.npy', ang_out_aa)
    np.save('CG_angles_hist.npy', ang_hist_aa)
    np.save('CG_distances.npy', dist_out_aa)
    np.save('CG_distances_hist.npy', dist_hist_aa)
    
    
    no_angs = len(D_ang) 
    nangs = int(no_angs/2)+1
    fig, axs = plt.subplots(nangs, 2, tight_layout=True, figsize=(25,35))
    fig.suptitle('Angles', fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins_angl[1:], ang_hist_cg[idx], color ='tab:red' ,alpha=0.75, label='CG')
            ax.plot( bins_angl[1:], ang_hist_aa[idx], color ='tab:blue', label='AA')
            ax.set_xlabel('Angle (°)', fontweight='bold', fontsize=16)
            ax.set_xlim(0,180)
            if idx in [3]:
                ax.set_title('-'.join(ang_tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(ang_tgts[idx]), fontweight='bold', fontsize=16)
            ax.set_ylim(0,0.1)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8) 
        axs.flatten()[0].set_ylabel('Probability density', fontweight='bold', fontsize=16)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=3, 
                    hspace=3)
    
    no_bonds = len(D_bond) 
    nbonds = int(no_bonds/2)+1
    fig, axs = plt.subplots(nbonds, 2, tight_layout=True, figsize=(25,35))
    fig.suptitle('Distances', fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins_dist[1:], dist_hist_cg[idx], color ='tab:red' ,alpha=0.5, label='CG')
                
            ax.plot( bins_dist[1:], dist_hist_aa[idx], color ='tab:blue', label='AA',zorder=10)
            
            if idx in [0,2]:
                ax.set_ylabel('Probability density', fontweight='bold', fontsize=16)
            if idx in [2,3]:
                ax.set_xlabel('Distance (nm)', fontweight='bold', fontsize=16)
            if idx in [0]:
                ax.set_title('-'.join(dist_tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(dist_tgts[idx]), fontweight='bold', fontsize=16)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8) 
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=3, 
                    hspace=3)
    return

def measure_ang_dist_dihed (dist_tgts, ang_tgts, dihed_tgts, u_aa, u_cg, bead_names, resname, stride1=100, stride2=1):
    '''Measures angles and distances - not dihedrals ! This can be added.
    dist_tgts and ang_tgts are the defined distances and angles, respectively.
    u_aa and u_cg is the mapped AA simulation and the simulated CG simulation.
    Stride1 and stride2 is for u_aa and u_cg, respectively.'''


    bins_dihed = np.arange(-181,181,2)
    bins_angl = np.arange(-1,181,2)
    bins_dist = np.arange(0,15,0.2)



    D_ang, D_bond = make_angle_bond_dicts(ang_tgts, dist_tgts, bead_names)

    ang_out_cg  = []
    ang_hist_cg = []
    ang_out_cg_wat  = []
    ang_hist_cg_wat = []
    ang_out_aa  = []
    ang_hist_aa = []
    ang_out_aa_wat  = []
    ang_hist_aa_wat = []
    
    
    
    dist_out_cg = []
    dist_hist_cg = []
    dist_out_cg_wat = []
    dist_hist_cg_wat = []
    dist_out_aa = []
    dist_hist_aa = []
    dist_out_aa_wat = []
    dist_hist_aa_wat = []
    
    dihed_out_cg  = []
    dihed_hist_cg = []
    dihed_out_cg_wat  = []
    dihed_hist_cg_wat = []
    dihed_out_aa  = []
    dihed_hist_aa = []
    dihed_out_aa_wat  = []
    dihed_hist_aa_wat = []



    for tgt in ang_tgts:
        ang_out_cg.append([])
        ang_out_aa.append([])
        ang_out_cg_wat.append([])
        ang_out_aa_wat.append([])
    for tgt in dist_tgts:
        dist_out_cg.append([])
        dist_out_aa.append([])
        dist_out_cg_wat.append([])
        dist_out_aa_wat.append([])
    for tgt in dihed_tgts:
        dihed_out_cg.append([])
        dihed_out_aa.append([])
        dihed_out_cg_wat.append([])
        dihed_out_aa_wat.append([])
    
    #molecules_aa = u_aa.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))
    #molecules_cg = u_cg.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))

    molecules_cg = u_cg.select_atoms('resname {} and name {}'.format(resname, dihed_tgts[0][0]))
    molecules_aa = u_aa.select_atoms('resname {} and name {}'.format(resname, dihed_tgts[0][0]))

    
    print('Calculating AA  distributions...')
    for resid in tqdm.tqdm(molecules_aa.resids):
        dist= []
        angs  = []
        diheds= []
        
        for tgt in ang_tgts:
            print (tgt)
            angs.append(beadstoangle(u_aa, resid, resname, tgt[0],
                                           resid, resname, tgt[1],
                                           resid, resname, tgt[2]))

        for tgt in dihed_tgts:
            diheds.append(beadstodihedral(u_aa, resid, resname, tgt[0],
                                                resid, resname, tgt[1],
                                                resid, resname, tgt[2],
                                                resid, resname, tgt[3]))
    
    

        for ts in u_aa.trajectory[::stride1]:
            for idx, ang in enumerate(angs):
                ang_out_aa[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out_aa[idx].append(beadstodistance(u_aa, resid,resname,dista[0],resid,resname,dista[1]))
            for idx, dihed in enumerate(diheds):
                dihed_out_aa[idx].append(dihed.value())

   #Making the distributions 
    for angle in ang_out_aa:
        ang_hist_aa.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out_aa:
        dist_hist_aa.append(np.histogram(distance, bins=bins_dist, density=True)[0])
    for dihed in dihed_out_aa:
        dihed_hist_aa.append(np.histogram(dihed, bins=bins_dihed, density=True)[0])


    np.save('AA_angles.npy', ang_out_aa)
    np.save('AA_angles_hist.npy', ang_hist_aa)
    np.save('AA_distances.npy', dist_out_aa)
    np.save('AA_distances_hist.npy', dist_hist_aa)
    np.save('AA_dihedrals.npy', dihed_out_aa)
    np.save('AA_dihedrals_hist.npy', dihed_hist_aa)
    
    
    print('Calculating CG  distributions...')
    for resid in tqdm.tqdm(molecules_cg.resids):
        diheds= []
        angs  = []
        
        for tgt in dihed_tgts:
            diheds.append(beadstodihedral(u_cg, resid, resname, tgt[0],
                                                resid, resname, tgt[1],
                                                resid, resname, tgt[2],
                                                resid, resname, tgt[3])) 
            for tgt in ang_tgts:
                angs.append(beadstoangle(u_cg, resid, resname, tgt[0],
                                               resid, resname, tgt[1],
                                               resid, resname, tgt[2]))
        
        for ts in u_cg.trajectory[::stride2]:
            for idx, ang in enumerate(angs):
                ang_out_cg[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out_cg[idx].append(beadstodistance(u_cg, resid,resname,dista[0],resid,resname,dista[1]))
            for idx, dihed in enumerate(diheds):
                dihed_out_cg[idx].append(dihed.value())
        
    for dihed in dihed_out_cg:
        dihed_hist_cg.append(np.histogram(dihed, bins=bins_dihed, density=True)[0])
    for angle in ang_out_cg:
        ang_hist_cg.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out_cg:
        dist_hist_cg.append(np.histogram(distance, bins=bins_dist, density=True)[0])
    
    np.save('CG_angles.npy', ang_out_aa)
    np.save('CG_angles_hist.npy', ang_hist_aa)
    np.save('CG_distances.npy', dist_out_aa)
    np.save('CG_distances_hist.npy', dist_hist_aa)
    np.save('CG_dihedrals.npy', dihed_out_aa)
    np.save('CG_dihedrals_hist.npy', dihed_hist_aa)
   

    fig, axs = plt.subplots(2, 2, tight_layout=True)
    fig.suptitle('Dihedral Angles', fontsize=25, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            if idx in [0,1]: ### Sharp distros
                ax.vlines(x=bins_dihed[1:][np.where(dihed_hist_cg[idx] == np.max(dihed_hist_cg[idx]))],
                           ymin=0,
                           ymax=0.1,
                           color='tab:red', alpha=0.75, label='CG')
                ax.scatter(bins_dihed[1:][np.where(dihed_hist_cg[idx] == np.max(dihed_hist_cg[idx]))], 0.1, marker='o',zorder=5,
                          color='tab:red', alpha=1)
                
                ax.vlines(x=bins_dihed[1:][np.where(dihed_hist_cg_wat[idx] == np.max(dihed_hist_cg_wat[idx]))],
                           ymin=0,
                           ymax=0.1,
                           color='orange', alpha=0.75, label='CG (Water)')
                ax.scatter(bins_dihed[1:][np.where(dihed_hist_cg_wat[idx] == np.max(dihed_hist_cg_wat[idx]))], 0.1, marker='o',zorder=5,
                          color='orange', alpha=1)
            else:
                ax.plot( bins_dihed[1:], dihed_hist_cg[idx], color ='tab:red' ,alpha=0.75, label='CG')
                ax.plot( bins_dihed[1:], dihed_hist_cg_wat[idx], color ='orange' ,alpha=0.55, linestyle='-', label='CG (Water)')
                
            ax.plot( bins_dihed[1:], dihed_hist_aa[idx], color ='tab:blue', label='AA',zorder=10)
            ax.plot( bins_dihed[1:], dihed_hist_aa_wat[idx], color ='cyan',alpha=0.55, linestyle='-', label='AA (Water)',zorder=10)
       
                
            if idx in [0,2]:
                ax.set_ylabel('Probability density', fontweight='bold', fontsize=16)
            if idx in [2,3]:
                ax.set_xlabel('Dihedral angle (°)', fontweight='bold', fontsize=16)
            ax.set_xlim(-180,180)
            if idx in [3]:
                ax.set_title('-'.join(dihed_tgts[idx]), fontweight='bold', fontsize=16, color='tab:red')
            else:
                ax.set_title('-'.join(dihed_tgts[idx]), fontweight='bold', fontsize=16)
                
    #         ax.set_ylim(0,0.2)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8) 


    
    no_angs = len(D_ang) 
    nangs = int(no_angs/2)+1
    fig, axs = plt.subplots(nangs, 2, tight_layout=True, figsize=(25,35))
    fig.suptitle('Angles', fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins_angl[1:], ang_hist_cg[idx], color ='tab:red' ,alpha=0.75, label='CG')
            ax.plot( bins_angl[1:], ang_hist_aa[idx], color ='tab:blue', label='AA')
            ax.set_xlabel('Angle (°)', fontweight='bold', fontsize=16)
            ax.set_xlim(0,180)
            if idx in [3]:
                ax.set_title('-'.join(ang_tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(ang_tgts[idx]), fontweight='bold', fontsize=16)
            ax.set_ylim(0,0.1)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8) 
        axs.flatten()[0].set_ylabel('Probability density', fontweight='bold', fontsize=16)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=3, 
                    hspace=3)
    
    no_bonds = len(D_bond) 
    nbonds = int(no_bonds/2)+1
    fig, axs = plt.subplots(nbonds, 2, tight_layout=True, figsize=(25,35))
    fig.suptitle('Distances', fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins_dist[1:], dist_hist_cg[idx], color ='tab:red' ,alpha=0.5, label='CG')
                
            ax.plot( bins_dist[1:], dist_hist_aa[idx], color ='tab:blue', label='AA',zorder=10)
            
            if idx in [0,2]:
                ax.set_ylabel('Probability density', fontweight='bold', fontsize=16)
            if idx in [2,3]:
                ax.set_xlabel('Distance (nm)', fontweight='bold', fontsize=16)
            if idx in [0]:
                ax.set_title('-'.join(dist_tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(dist_tgts[idx]), fontweight='bold', fontsize=16)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8) 
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=3, 
                    hspace=3)
    return



def re_run (resname, mdp_loc='/data1/lisbeth/Params/FRAGMENTS/AA_SIMS_WATER_IONS/MDPs', maxwarn=1):
    '''Re-runs min, and short production run. By default it uses the mdp files
    located here: /data1/lisbeth/Params/FRAGMENTS/AA_SIMS_WATER_IONS/MDPs.'''
    ### Minimize the system
    subprocess.call("gmx grompp -f {}/min.mdp -c memion.gro -p topol.top -o m.tpr -maxwarn {}".format(mdp_loc,maxwarn)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm m "
                    , shell = True)
    ### Relax the system
    subprocess.call("gmx grompp -f {}/eq.mdp -c m.gro -p topol.top -o r.tpr -maxwarn {}".format(mdp_loc,maxwarn)
                    , shell = True)
    subprocess.call("gmx mdrun -v -deffnm r  -nt 10"
                , shell = True)
    
    ### Prepare the simulation
    subprocess.call("gmx grompp -f {}/run.mdp -c r.gro -p topol.top -o p.tpr -maxwarn {}".format(mdp_loc,maxwarn)
                    , shell = True)
    ### Prepare the simulation
    subprocess.call("gmx mdrun -v -deffnm p -nt 10".format(mdp_loc)
                    , shell = True)

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.xtc -pbc mol'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()

    ###Correct for the PBC
    p = subprocess.Popen('gmx trjconv -f p.xtc -s p.tpr -o pbc.gro -pbc mol -b 0 -e 0'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate(resname)
    p.wait()
    return 


def TI_WAT (mdp_loc, resname):
    '''stand in CG_WAT directory and run this function'''
    solvent_wat    = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/solvent.gro'
    solvat_oco_inp = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/solvate_OCO.inp'
    oco            = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/OCO-vmd.pdb' 
    W              = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/W.pdb'
    
    os.mkdir('TI_WAT')
    os.chdir('TI_WAT')
    
    
    u = md.Universe('../p.gro')
    res = u.select_atoms('resname {}'.format(resname))
    sys = u.select_atoms('all')
    rest = u.select_atoms('all and not resname {}'.format(resname))
    
    #MAKE INDEX FILE
    with md.selections.gromacs.SelectionWriter('index.ndx', mode='w') as ndx:
        ndx.write(res, name='{}'.format(resname))
        ndx.write(sys, name='System')
        ndx.write(rest, name='Solvent')
    
    os.system('echo 0 > sel')
    os.system('echo 0 >> sel')
    os.system("cat sel | gmx editconf -f ../p.tpr -o molecule.gro -n index.ndx -c")
    os.system('rm sel')
    
    #solvate
    os.system("gmx solvate -cp molecule.gro -cs {} -o solvated.gro -box 5.5 5.5 5.5".format(solvent_wat))
    
    u = md.Universe('solvated.gro')
    nWat = u.select_atoms('resname W').resids.shape[0]
    
    #Write topol.top
    topint = open('topol.top', 'w')
    topint.write('#include "{}/martini_v3.0.0.itp"\n'.format(mdp_loc))
    topint.write('#include "{}/Martini_v1_tails.itp"\n'.format(mdp_loc))
    topint.write('#include "../initial_CG.itp"\n')
    topint.write('#include "{}/martini_v3.0.0_solvents_v1.itp"\n'.format(mdp_loc))
    topint.write('\n')
    topint.write('[ system ]\n')
    topint.write('one molecule\n')
    topint.write('[ molecules ]\n')
    topint.write('{}    1\n'.format(resname))
    topint.write('W     {}\n'.format(nWat))
    topint.close()
    
    #min
    os.system("gmx grompp -f {}/min.mdp -p topol.top -c solvated.gro -o min.tpr -maxwarn 10".format(mdp_loc))
    os.system('gmx mdrun -v -deffnm min -nt 2')
    os.system('rm -rf \#*')
    working_dir = os.getcwd()
    
    
    for s in range(0,21):
        os.mkdir('{}'.format(s))
        os.chdir('{}'.format(s))
        os.system('sed -e "s/sedstate/{}/" {}/run.mdp > run.mdp'.format(s, mdp_loc))
        os.system('gmx grompp -f run.mdp -p ../topol.top -c ../min.gro -maxwarn 10')
        os.system('gmx mdrun -v -nt 2 -rdd 1.3 -dlb yes -cpi state.cpt > mdrun.log')
        os.system('rm -rf \#*')
        os.chdir(working_dir)
        os.system('ln -s {}/dhdl.xvg dhdl.{}.xvg'.format(s,s))
    
    
    os.system('gmx bar -f dhdl.*.xvg -o -oi -oh | grep total > final.dat')
    os.chdir('../')
    return



def TI_OCO (resname, mdp_loc):
    '''stand in CG_WAT directory and run this function'''
    solvent_wat    = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/solvent.gro'
    solvat_oco_inp = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/solvate_OCO.inp'
    oco            = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/OCO-vmd.pdb' 
    W              = '/data1/lisbeth/Params/FRAGMENTS/SOLVENTS/W.pdb'

    os.mkdir('TI_OCT')
    os.chdir('TI_OCT')
    
    u = md.Universe('../p.gro')
    res = u.select_atoms('resname {}'.format(resname))
    sys = u.select_atoms('all')
    rest = u.select_atoms('all and not resname {}'.format(resname))
    
    #MAKE INDEX FILE
    with md.selections.gromacs.SelectionWriter('index.ndx', mode='w') as ndx:
        ndx.write(res, name='{}'.format(resname))
        ndx.write(sys, name='System')
        ndx.write(rest, name='Solvent')

    os.system('echo 0 > sel')
    os.system('echo 0 >> sel')
    os.system("cat sel | gmx editconf -f ../p.tpr -o molecule.gro -n index.ndx -c")
    os.system('rm sel')
    
    #solvate
    os.system('cp {} .'.format(solvat_oco_inp))
    os.system('cp {} .'.format(oco))
    os.system('cp {} .'.format(W))
    
    os.system('packmol < solvate_OCO.inp')
    os.system('gmx editconf -f solvated.pdb -box 5.5 5.5 5.5 -o solvated.gro')
    
    #Write topol.top
    u = md.Universe('solvated.gro')
    nWat = u.select_atoms('resname W').resids.shape[0]
    nOCO = u.select_atoms('resname OCO').resids.shape[0]
    
    #Write topol.top
    topint = open('topol.top', 'w')
    topint.write('#include "{}/martini_v3.0.0.itp"\n'.format(mdp_loc))
    topint.write('#include "{}/Martini_v1_tails.itp"\n'.format(mdp_loc))
    topint.write('#include "../initial_CG.itp"\n')
    topint.write('#include "{}/martini_v3.0.0_solvents_v1.itp"\n'.format(mdp_loc))
    topint.write('\n')
    topint.write('[ system ]\n')
    topint.write('one molecule\n')
    topint.write('[ molecules ]\n')
    topint.write('{}    1\n'.format(resname))
    topint.write('OCO     {}\n'.format(nOCO))
    topint.write('W     {}\n'.format(nWat))
    topint.close()
    
    #Min
    os.system('gmx grompp -f {}/min.mdp -p topol.top -c solvated.gro -o min.tpr -maxwarn 10'.format(mdp_loc))
    os.ssytem('gmx mdrun -v -deffnm min -nt 2')
    
    
    for s in range(0,21):
        os.mkdir('{}'.format(s))
        os.chdir('{}'.format(s))
        os.system('sed -e "s/sedstate/{}/" {}/run.mdp > run.mdp'.format(s, mdp_loc))
        os.system('gmx grompp -f run.mdp -p ../topol.top -c ../min.gro -maxwarn 10')
        os.system('gmx mdrun -v -nt 2 -rdd 1.3 -dlb yes -cpi state.cpt > mdrun.log')
        os.system('rm -rf \#*')
        os.chdir(working_dir)
        os.system('ln -s {}/dhdl.xvg dhdl.{}.xvg'.format(s,s))
    
    os.system('gmx bar -f dhdl.*.xvg -o -oi -oh | grep total > final.dat')
    return


def measure_bonded_terms_simple (dist_tgts, ang_tgts, u, bead_names, resname, outname, stride=100):
    '''Measure bonded parameters, bonds, angles and dihedrals. Run this for both AA and CG sims.
    Returns the normalized distributions and saves both the raw date and the distributions in the working directory under the outname provided.'''
    D_ang, D_bond = make_angle_bond_dicts(ang_tgts, dist_tgts, bead_names)

    bins_angl = np.arange(-1,181,2)
    bins_dist = np.arange(0,15,0.2)

    ang_out  = []
    ang_hist = []

    dist_out = []
    dist_hist = []

    for tgt in ang_tgts:
        ang_out.append([])
    for tgt in dist_tgts:
        dist_out.append([])

    molecules = u.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))
    print('Calculating distributions...')
    for resid in tqdm.tqdm(molecules.resids):
        dist= []
        angs  = []
        diheds = []

        for tgt in ang_tgts:
            angs.append(beadstoangle(u, resid, resname, tgt[0],
                                        resid, resname, tgt[1],
                                        resid, resname, tgt[2]))

        for ts in u.trajectory[::stride]:
            for idx, ang in enumerate(angs):
                ang_out[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out[idx].append(beadstodistance(u, resid,resname,dista[0],resid,resname,dista[1]))


    for angle in ang_out:
        ang_hist.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out:
        dist_hist.append(np.histogram(distance, bins=bins_dist, density=True)[0])

    np.save('{}_angles.npy'.format(outname), ang_out)
    np.save('{}_angles_hist.npy'.format(outname), ang_hist)
    np.save('{}_distances.npy'.format(outname), dist_out)
    np.save('{}_distances_hist.npy'.format(outname), dist_hist)
    return ang_hist, dist_hist, ang_out, dist_out


def measure_bonded_terms (dist_tgts, ang_tgts, dihed_tgts, u, bead_names, resname, outname, stride=100):
    '''Measure bonded parameters, bonds, angles and dihedrals. Run this for both AA and CG sims.
    Returns the normalized distributions and saves both the raw date and the distributions in the working directory under the outname provided.'''
    D_dihed, D_ang, D_bond = make_angle_bond_dihed_dicts(ang_tgts, dist_tgts, dihed_tgts, bead_names)

    bins_dihed = np.arange(-181,181,2)
    bins_angl = np.arange(-1,181,2)
    bins_dist = np.arange(0,15,0.2)

    dihed_out  = []
    dihed_hist = []
    
    ang_out  = []
    ang_hist = []

    dist_out = []
    dist_hist = []

    for tgt in ang_tgts:
        ang_out.append([])
    for tgt in dist_tgts:
        dist_out.append([])
    for tgt in dihed_tgts:
        dihed_out.append([])

    molecules = u.select_atoms('resname {} and name {}'.format(resname, dist_tgts[0][0]))
    print('Calculating distributions...')
    for resid in tqdm.tqdm(molecules.resids):
        dist= []
        angs  = []
        diheds = []

        for tgt in ang_tgts:
            angs.append(beadstoangle(u, resid, resname, tgt[0],
                                           resid, resname, tgt[1],
                                           resid, resname, tgt[2]))
        for tgt in dihed_tgts:
            diheds.append(beadstodihedral(u, resid, resname, tgt[0],
                                                resid, resname, tgt[1],
                                                resid, resname, tgt[2],
                                                resid, resname, tgt[3]))

        for ts in u.trajectory[::stride]:
            for idx, ang in enumerate(angs):
                ang_out[idx].append(ang.value())
            for idx, dista in enumerate(dist_tgts):
                dist_out[idx].append(beadstodistance(u, resid,resname,dista[0],resid,resname,dista[1]))
            for idx, dihed in enumerate(diheds):
                dihed_out[idx].append(dihed.value())


    for angle in ang_out:
        ang_hist.append(np.histogram(angle, bins=bins_angl, density=True)[0])
    for distance in dist_out:
        dist_hist.append(np.histogram(distance, bins=bins_dist, density=True)[0])
    for dihed in dihed_out:
        dihed_hist.append(np.histogram(dihed, bins=bins_dihed, density=True)[0])

    np.save('{}_angles.npy'.format(outname), ang_out)
    np.save('{}_angles_hist.npy'.format(outname), ang_hist)
    np.save('{}_distances.npy'.format(outname), dist_out)
    np.save('{}_distances_hist.npy'.format(outname), dist_hist)
    np.save('{}_dihedrals.npy'.format(outname), ang_out)
    np.save('{}_dihedrals_hist.npy'.format(outname), ang_hist)
    return ang_hist, dist_hist, dihed_hist, ang_out, dist_out, dihed_out

def plot_4dists (Dict, tgts, hist_aa, hist_aa_wat, hist_cg, hist_cg_wat, bead_names, resname, hist_type='angles'):
    '''Plot both AA dist in water and bilayer besides CG dist in water and bilayer.
    hist_type can be angles, bonds or dihedrals'''
    if hist_type == 'angles':
        bins = np.arange(-1,181,2)
        label = 'Angle (°)'
    elif hist_type == 'bonds':
        bins = np.arange(0,15,0.2)
        label = 'Bond (nm)'
    elif hist_type == 'dihedrals':
        bins = np.arange(-181,181,2)
        label = 'Dihedral (°)'

    #D_dihed, D_ang, D_bond = make_angle_bond_dihed_dicts(ang_tgts, dist_tgts, dihed_tgts, bead_names)

    no_type = len(Dict)
    ntypes = int(no_type/2)+1
    fig, axs = plt.subplots(ntypes, 2, tight_layout=True, figsize=(25,35), sharex=True)
    fig.suptitle(hist_type, fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins[1:], hist_cg[idx], color ='tab:red' ,alpha=0.75, label='CG')
            ax.plot( bins[1:], hist_aa[idx], color ='tab:blue', label='AA')
            ax.plot( bins[1:], hist_cg_wat[idx], color ='tab:red' ,alpha=0.75, label='CG_wat', linestyle='--')
            ax.plot( bins[1:], hist_aa_wat[idx], color ='tab:blue', label='AA_wat', linestyle='--')

            ax.set_xlabel(label, fontweight='bold', fontsize=16)
            #ax.set_xlim(0,180)
            if idx in [3]:
                ax.set_title('-'.join(tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(tgts[idx]), fontweight='bold', fontsize=16)
            #ax.set_ylim(0,0.1)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8)
        axs.flatten()[0].set_ylabel('Probability density', fontweight='bold', fontsize=16)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=3,
                    hspace=3)
    return

def plot_2dists (Dict, tgts, hist_aa, hist_cg, bead_names, resname, hist_type='angles'):
    '''Plot AA dist in either water or bilayer along with the corresponding CG dist provided.  hist_type can be angles, bonds or dihedrals'''
    if hist_type == 'angles':
        bins = np.arange(-1,181,2)
        label = 'Angle (°)'
    elif hist_type == 'bonds':
        bins = np.arange(0,15,0.2)
        label = 'Bond (nm)'
    elif hist_type == 'dihedrals':
        bins = np.arange(-181,181,2)
        label = 'Dihedral (°)'

    #D_dihed, D_ang, D_bond = make_angle_bond_dihed_dicts(ang_tgts, dist_tgts, dihed_tgts, bead_names)

    no_type = len(Dict)
    ntypes = int(no_type/2)+1
    fig, axs = plt.subplots(ntypes, 2, tight_layout=True, figsize=(25,35))
    fig.suptitle(hist_type, fontsize=15, y=1.01, fontweight='bold')
    for idx, ax in enumerate(axs.flatten()):
        try:
            ax.plot( bins[1:], hist_cg[idx], color ='tab:red' ,alpha=0.75, label='CG')
            ax.plot( bins[1:], hist_aa[idx], color ='tab:blue', label='AA')

            ax.set_xlabel(label, fontweight='bold', fontsize=16)
            #ax.set_xlim(0,180)
            if idx in [3]:
                ax.set_title('-'.join(tgts[idx]), fontweight='bold', fontsize=16)
            else:
                ax.set_title('-'.join(tgts[idx]), fontweight='bold', fontsize=16)
            #ax.set_ylim(0,0.1)
        except IndexError:
            pass
        axs.flatten()[0].legend(loc='upper center', ncol=4, fontsize=8)
        axs.flatten()[0].set_ylabel('Probability density', fontweight='bold', fontsize=16)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=3,
                    hspace=3)
    return

def make_angle_bond_dihed_dicts(ang_tgts, dist_tgts, dihed_tgts, bead_names):
    '''Returns D_dihed, D_ang, D_bond'''
    D_ang = {}
    for idx, a in enumerate(ang_tgts):
        loc_ang = []
        for i in a:
            for idj, j in enumerate(bead_names):
                if i==j:
                    loc_ang.append(idj+1)
        D_ang[idx+1] = loc_ang

    D_bond = {}
    for idx, a in enumerate(dist_tgts):
        loc = []
        for i in a:
            for idj, j in enumerate(bead_names):
                if i == j:
                    loc.append(idj+1)
        D_bond[idx+1] = loc

    D_dihed = {}
    for idx, a in enumerate(dihed_tgts):
        loc_dihed = []
        for i in a:
            for idj, j in enumerate(bead_names):
                if i == j:
                    loc_dihed.append(idj+1)
        D_dihed[idx+1] = loc_dihed
    return D_dihed, D_ang, D_bond


def write_initial_CGitp3 (resname, bead_names, bead_sizes, bead_types, bead_charges, D_bond, D_ang, dist_tgts, ang_tgts):
    '''Function to write the inital CG itp file, given the defined angles and bonds.
    It also makes a nice format of the itp. Hence why you need to give so many parameters.
    The fc is by default 500 and 25 for bonds and angles, respectively. '''
    mass_dir = {'R':72,
            'S':54,
            'T':36}
    initial = open('initial_CG.itp', 'w')
    initial.write('[ moleculetype ]\n')
    initial.write('{}  1'.format(resname))
    initial.write('\n')
    initial.write('[ atoms ]\n')
    initial.write('; nr type resnr residue atom cgnr charge mass\n')
    for idx, bead in enumerate(bead_names):
        initial.write("{:4}    {:4}    {:4}    {:4}    {:4}    {:4}    {:4}    {:4}\n"
                      .format(idx+1, bead_types[idx], 0, resname, bead, idx+1, bead_charges[idx], mass_dir[bead_sizes[idx]] ))
    initial.write('\n')
    initial.write('[ bonds ]\n')
    initial.write('; i  j  funct length\n')
    for idx, i in enumerate(D_bond):
        #print (D_bond[i])
        initial.write('{:4}    {:4}    {:4}    {:4}    {:4}  ;{}-{}\n'
                      .format(D_bond[i][0], D_bond[i][1], 1,  0.47, 5000, dist_tgts[idx][0], dist_tgts[idx][1]))
    initial.write('\n')
    initial.write('[ angles ]\n')
    initial.write(';  i  j  k   funct   angle   force.c.\n')
    for idx, i in enumerate(D_ang):
        #print (D_ang[i])
        initial.write('{:4}    {:4}    {:4}    {:4}    {:4}    {:4}  ;{}-{}-{}\n'
                      .format(D_ang[i][0], D_ang[i][1], D_ang[i][2], 2,  130, 25, ang_tgts[idx][0],ang_tgts[idx][1],ang_tgts[idx][2]))
    initial.write('\n')
    initial.close()
    return


def write_initial_CGitp2 (resname, bead_names, bead_sizes, bead_types, bead_charges, D_bond, D_ang, D_dihed,dist_tgts, ang_tgts, dihed_tgts):
    '''Function to write the inital CG itp file, given the defined angles, bonds and dihedrals.
    It also makes a nice format of the itp. Hence why you need to give so many parameters.
    The fc is by default 500 and 25 for bonds and angles, respectively. '''
    mass_dir = {'R':72,
            'S':54,
            'T':36}
    initial = open('initial_CG.itp', 'w')
    initial.write('[ moleculetype ]\n')
    initial.write('{}  1'.format(resname))
    initial.write('\n')
    initial.write('[ atoms ]\n')
    initial.write('; nr type resnr residue atom cgnr charge mass\n')
    for idx, bead in enumerate(bead_names):
        initial.write("{:4}    {:4}    {:4}    {:4}    {:4}    {:4}    {:4}    {:4}\n"
                      .format(idx+1, bead_types[idx], 0, resname, bead, idx+1, bead_charges[idx], mass_dir[bead_sizes[idx]] ))
    initial.write('\n')
    initial.write('[ bonds ]\n')
    initial.write('; i  j  funct length\n')
    for idx, i in enumerate(D_bond):
        #print (D_bond[i])
        initial.write('{:4}    {:4}    {:4}    {:4}    {:4}  ;{}-{}\n'
                      .format(D_bond[i][0], D_bond[i][1], 1,  0.47, 5000, dist_tgts[idx][0], dist_tgts[idx][1]))
    initial.write('\n')
    initial.write('[ angles ]\n')
    initial.write(';  i  j  k   funct   angle   force.c.\n')
    for idx, i in enumerate(D_ang):
        #print (D_ang[i])
        initial.write('{:4}    {:4}    {:4}    {:4}    {:4}    {:4}  ;{}-{}-{}\n'
                      .format(D_ang[i][0], D_ang[i][1], D_ang[i][2], 2,  130, 25, ang_tgts[idx][0],ang_tgts[idx][1],ang_tgts[idx][2]))
    initial.write('\n')
    initial.write('[ dihedrals ]\n')
    initial.write(';  i  j  k l  funct   angle   force.c.\n')
    for idx, i in enumerate(D_dihed):
        dihed = dihed_tgts[idx]
        initial.write('{:4}    {:4}    {:4}    {:4}   ;{}-{}-{}-{}\n'
                      .format(D_dihed[i][0], D_dihed[i][1], D_dihed[i][2], D_dihed[i][3], dihed[0],dihed[1],dihed[2],dihed[3]))
    initial.close()
    return
