# Keith Briggs 2025-09-29 callback demo

from CRRM import Simulator, Parameters

def show_attachments(a):
    print(f"attachment vector = {a}")

crrm = Simulator(Parameters(n_ues=20))
crrm.a.set_callback(show_attachments)
crrm.update()
