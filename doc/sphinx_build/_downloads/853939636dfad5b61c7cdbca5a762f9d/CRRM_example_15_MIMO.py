# Keith Briggs 2025-09-25 basic test of MIMO spectral efficiency

from CRRM import Simulator, Parameters
import numpy as np
np.set_printoptions(precision=2,linewidth=200)

for MIMO in ((1,1),(2,2),(4,4),(8,2),(8,4),(8,8),(64,64),):
  crrm=Simulator(Parameters(n_ues=20,MIMO=MIMO))
  print(f"MIMO={MIMO} UE spectral efficiencies={crrm.get_spectral_efficiency(ues=(0,1,2,3,4,5,),subbands=0)} b/s/Hz")
