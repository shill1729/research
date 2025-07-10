import numpy as np
import matplotlib.pyplot as plt

from tda.ellipsoidal.complexes.complexes import EllipsoidalVR, EllipsoidalCech
from tda.ellipsoidal.filtrations.filtrations import build_filtration
from tda.toydata.toydata import generate_toy_data
from tda.ellipsoidal.plotting.plotting import plot_simplices, plot_point_cloud, plot_diagram_and_barcode


# ---------------------------------------------------------------------------
#  One-stop demo
# ---------------------------------------------------------------------------
def demo(surface_or_curve, n_pts=10, radii=np.linspace(0.01, 10.0, 25), max_dim=2):
    # toy data
    xs, A_list, pc = generate_toy_data(n=n_pts, surface_or_curve=surface_or_curve)
    if pc.target_dim == 2:
        curve, _ = pc.get_curve(200)
    elif pc.target_dim == 3:
        curve, _ = pc.get_surface(200)
    else:
        raise NotImplementedError("Only intrinsic dim 1 and 2 are implemented")
    # --- VR ---------------------------------------------------------------
    st_vr, diag_vr = build_filtration(xs, A_list, EllipsoidalVR, radii, max_dim)
    plot_diagram_and_barcode(diag_vr, 'Ellipsoidal VR')

    # --- Čech -------------------------------------------------------------
    st_c, diag_c = build_filtration(xs, A_list, EllipsoidalCech, radii, max_dim)
    plot_diagram_and_barcode(diag_c, 'Ellipsoidal Čech')

    # --- Optional: visualise complexes at two extreme radii --------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, eps, name in zip(axes, [radii[0], radii[-1]], ['ε min', 'ε max']):
        comp = EllipsoidalCech(xs, A_list, eps)
        comp.build_complex(max_dim=3)
        plot_point_cloud(ax, xs, A_list, curve, eps)
        plot_simplices(ax, xs, comp.edges, comp.triangles)
        # ax.set_title(f'VR complex, {name}={eps:.2f}')
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from ae.toydata.curves import Circle
    from ae.toydata.surfaces import Sphere, Paraboloid
    demo(Circle())
    demo(Sphere())

