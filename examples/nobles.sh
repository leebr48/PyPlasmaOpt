mpirun -n 4 python3 example3_simple.py --ppp 20 --at-optimum --output ncsxPartC --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000. --torsion 0.0 --curvature 1e-6 --arclength 0.00
mpirun -n 4 python3 example3_simple.py --ppp 20 --at-optimum --output noble1 --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000. --torsion 0.0 --curvature 1e-6 --arclength 0.00 --reload *ncsxPartC* --freezeCoils --iota_target -0.381966
mpirun -n 4 python3 example3_simple.py --ppp 20 --at-optimum --output noble2 --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000. --torsion 0.0 --curvature 1e-6 --arclength 0.00 --reload *ncsxPartC* --freezeCoils --iota_target -0.206011
mpirun -n 4 python3 example3_simple.py --ppp 20 --at-optimum --output noble3 --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000. --torsion 0.0 --curvature 1e-6 --arclength 0.00 --reload *ncsxPartC* --freezeCoils --iota_target -0.618034
mpirun -n 4 python3 example3_simple.py --ppp 20 --at-optimum --output noble4 --Nt_ma 4 --Nt_coils 6 --min-dist 0.2 --dist-weight 1000. --torsion 0.0 --curvature 1e-6 --arclength 0.00 --reload *ncsxPartC* --freezeCoils --iota_target -0.793989
