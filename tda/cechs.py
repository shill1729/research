from matplotlib import pyplot as plt

from tda.ellipsoidal.complexes.complexes import EllipsoidalVR, EllipsoidalCech
from tda.ellipsoidal.plotting.plotting import plot_point_cloud, plot_simplices
from tda.toydata.toydata import generate_curve_point_cloud_and_ellipses


def build_and_plot_all(xs, A_list, curve, epsilons):
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    # Define order: VR @ eps[0], Čech @ eps[0], VR @ eps[1], Čech @ eps[1]
    configs = [
        (epsilons[0], EllipsoidalVR, 'VR'),
        (epsilons[0], EllipsoidalCech, 'Čech'),
        (epsilons[1], EllipsoidalVR, 'VR'),
        (epsilons[1], EllipsoidalCech, 'Čech'),
    ]

    for ax, (eps, ComplexClass, name) in zip(axes, configs):
        # Build complex up to triangles
        complex_obj = ComplexClass(xs, A_list, eps)
        complex_obj.build_complex(max_dim=3)

        # Plot base geometry
        plot_point_cloud(ax, xs, A_list, curve, eps)
        plot_simplices(ax, xs, complex_obj.edges, complex_obj.triangles)

        ax.set_title(f"{name} Complex (ε={eps})")
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    xs, A_list, point_cloud = generate_curve_point_cloud_and_ellipses(n=3)
    curve, _ = point_cloud.get_curve(50)
    # epsilons = [0.8, 1.25]
    epsilons = [0.1, 0.5]
    build_and_plot_all(xs, A_list, curve, epsilons)
