#python3 simple.py --ppp 20 --out ncsx2INoQFM --Nt_ma 6 --Nt_coils 6 --min_dist 0.2 --dist_wt 1000. --tors 0 --curv 1e-6 --arclen 0.00 --QS_wt 1 --iota_targ -0.395938929522566 -0.793989 --QFM 0 --qfm_vol 2.959 --mmax 6 --nmax 6 --ntheta 40 --nphi 40 --Taylor --image 10
python3 qfm.py --sourcedir output-ncsx2INoQFM* --stellID 0 --qfm_vol 2.959
python3 qfm.py --sourcedir output-ncsx2INoQFM* --stellID 1 --qfm_vol 2.959
