python3 simple.py --ppp 20 --out hsx2I5Coil --Nt_ma 6 --Nt_coils 6 --min_dist 0.1 --dist_wt 1000. --tors 0 --curv 1e-6 --arclen 0.00 --QS_wt 1 --iota_targ -0.395938929522566 -0.793989 --flat --num_coils 5 --nfp 4 --QFM 1 --maj_rad 1.2 --min_rad 0.3 --qfm_vol 0.44 --mmax 6 --nmax 6 --ntheta 40 --nphi 40 --Taylor --kick --z0factr 1 --mag 0.01 --image 10
python3 qfm.py --sourcedir output-hsx2I5Coil* --stellID 0
python3 qfm.py --sourcedir output-hsx2I5Coil* --stellID 1
