import numpy as np
import pathlib as pl

def Checkpoint(obj, iteration=0):
    '''Saves stellarator data necessary for postprocessing
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

    matlabcoils = [c.tomatlabformat() for c in obj.stellarator_group[0]._base_coils]
    np.savetxt(str(pl.Path(obj.outdir).joinpath('coilsmatlab.txt')), np.hstack(matlabcoils))

    save = obj.stellarator_group[0]._base_coils[0].coefficients
    for i in range(1,len(obj.stellarator_group[0]._base_coils)):
        save = np.append(save,obj.stellarator_group[0]._base_coils[i].coefficients,axis=0)
    np.savetxt(str(pl.Path(obj.outdir).joinpath('coilCoeffs.txt')), save,fmt='%.20f')

    for i,ma in enumerate(obj.ma_group):
        save = []
        for item in obj.ma_group[i].coefficients:
            save.append(item.tolist())
        with open(str(pl.Path(obj.outdir).joinpath('maCoeffs_%d.txt'%i)), "w") as f:
            for line in save:
                for ind,item in enumerate(line):
                    f.write(str(item))
                    if ind!=len(line)-1:
                        f.write(' ')
                f.write('\n')

    for i,qsf in enumerate(obj.qsf_group):
        save1 = obj.qsf_group[i].eta_bar 
        np.savetxt(str(pl.Path(obj.outdir).joinpath('eta_bar_%d.txt'%i)), [save1],fmt='%.20f') 
        save2 = obj.calc_iotas[i]
        np.savetxt(str(pl.Path(obj.outdir).joinpath('iota_ma_%d.txt'%i)), [save2],fmt='%.20f')

    np.savetxt(obj.outdir + "Jvals.txt", obj.Jvals)
    np.savetxt(obj.outdir + "dJvals.txt", obj.dJvals)
    np.savetxt(obj.outdir + "xiterates.txt", obj.xiterates)
    np.savetxt(obj.outdir + "Jvals_individual.txt", obj.Jvals_individual)

    save = 'Stellarator parameters last saved after iteration %d.' % iteration
    with open(str(pl.Path(obj.outdir).joinpath('lastSave.txt')), "w") as f:
        f.write(save)
