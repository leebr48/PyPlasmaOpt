import numpy as np
import pathlib as pl

def Checkpoint(obj, iteration=0):
    '''
    Saves stellarator data necessary for postprocessing
    and restarting.

    Inputs:
    obj: Instance of NearAxisQuasiSymmetryObjective
    iteration: Iteration number of optimizer

    Outputs:
    [Files written in output directory specified by obj.outdir]
    '''

    for stellind,stellarator in enumerate(obj.stellarator_group):
        np.savetxt(str(pl.Path(obj.outdir).joinpath('currents_%d.txt'%stellind)), obj.stellarator_group[stellind]._base_currents)
        for coilind,coil in enumerate(obj.stellarator_group[stellind].coils):
            np.savetxt(str(pl.Path(obj.outdir).joinpath('coil-%d.txt'%coilind)),obj.stellarator_group[0].coils[coilind].gamma)
            np.savetxt(str(pl.Path(obj.outdir).joinpath('current-%d_%d.txt'%(coilind,stellind))),[obj.stellarator_group[stellind].currents[coilind]])

    with open(str(pl.Path(obj.outdir).joinpath('coilCoeffs.txt')),'w') as f:
        for ind,coil in enumerate(obj.stellarator_group[0]._base_coils):
            f.write('COIL %d\n' % ind)
            if isinstance(coil.coefficients[0],np.ndarray): # This handles the traditional coils
                for group in coil.coefficients:
                    save = PrepforSave(group)
                    f.write(save)
            else: # This handles the control coils
                save = PrepforSave(coil.coefficients)
                f.write(save)
    
    for i,ma in enumerate(obj.ma_group):
        save = []
        for item in obj.ma_group[i].coefficients:
            save.append(item.tolist())
        SaveMACoeffs(save,i,obj.outdir,'maCoeffs')

    for i,qsf in enumerate(obj.qsf_group):
        save1 = obj.qsf_group[i].eta_bar 
        np.savetxt(str(pl.Path(obj.outdir).joinpath('eta_bar_%d.txt'%i)), [save1],fmt='%.20f') 
        save2 = obj.calc_iotas[i]
        np.savetxt(str(pl.Path(obj.outdir).joinpath('iota_ma_%d.txt'%i)), [save2],fmt='%.20f')

    np.savetxt(obj.outdir + "Jvals.txt", obj.Jvals)
    np.savetxt(obj.outdir + "dJvals.txt", obj.dJvals)
    np.savetxt(obj.outdir + "xiterates.txt", obj.xiterates)
    np.savetxt(obj.outdir + "Jvals_individual.txt", obj.Jvals_individual)
    
    if obj.qfm_weight > obj.ignore_tol:
        [obj.qfm_group[i].SaveState() for i in obj.stellList]

    if obj.tanMap:
        for i in obj.stellList:
            R,Z = obj.tangentMap_group[i].ft_RZ(nfp=obj.nfp,Nt=obj.Nt_ma,adjoint=False)
            SaveMACoeffs([R,Z],i,obj.outdir,'tanMap_axis_coeffs')
            
    save = 'Stellarator parameters last saved after iteration %d.\n' % iteration
    with open(str(pl.Path(obj.outdir).joinpath('lastSave.txt')), "w") as f:
        f.write(save)

def PrepforSave(nonstring):
    return str(nonstring).replace('\n',' ').replace('[','').replace(']','') + '\n'

def SaveMACoeffs(saveList,i,outdir,name):
        with open(str(pl.Path(outdir).joinpath(name + '_%d.txt'%i)), "w") as f:
            for line in saveList:
                for ind,item in enumerate(line):
                    f.write(str(item))
                    if ind!=len(line)-1:
                        f.write(' ')
                f.write('\n')
