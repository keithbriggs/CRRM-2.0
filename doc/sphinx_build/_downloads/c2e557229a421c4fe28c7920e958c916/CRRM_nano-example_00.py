# Keith Briggs 2025-09-15

from CRRM import Simulator, Parameters

crrm = Simulator(
    Parameters(n_ues=20, layout_plot_fnbase="img/CRRM_nano-example_layout")
)
crrm.layout_plot()
crrm.update()
print(f"UE throughputs={crrm.get_UE_throughputs().astype(int)[:,0].flatten()} Mb/s")
