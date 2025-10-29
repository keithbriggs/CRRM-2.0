# Keith Briggs 2025-09-22

from CRRM import Simulator, Parameters

crrm = Simulator(
    Parameters(n_ues=20, layout_plot_fnbase="img/CRRM_example_01_quick-start_layout")
)
crrm.update()
crrm.layout_plot(show_attachment_type="attachment")
print(f"UE throughputs={crrm.get_UE_throughputs(subbands=0).astype(int)} Mb/s")
