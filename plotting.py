import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def configure_matplotlib():
    config = {
        'figure.figsize':(14,4),
        'figure.dpi':100,
        'figure.facecolor':'white',
        'axes.grid':True,
        'grid.linestyle':'--',
        'grid.linewidth':0.5,
        'axes.spines.top':False,
        'axes.spines.bottom':False,
        'axes.spines.left':False,
        'axes.spines.right':False
    }

    plt.rcParams.update(config)
    
    
def quantecon_param_plot():

    """This function creates the graph on page 189 of
    Sargent Macroeconomic Theory, second edition, 1987.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')

    # Set axis
    xmin, ymin = -3, -2
    xmax, ymax = -xmin, -ymin
    plt.axis([xmin, xmax, ymin, ymax])

    # Set axis labels
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(r'$\phi_2$', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(r'$\phi_1$', rotation=0, fontsize=16)
    ax.yaxis.set_label_position('right')

    # Draw (t1, t2) points
    ρ1 = np.linspace(-2, 2, 100)
    ax.plot(ρ1, -abs(ρ1) + 1, c='black')
    ax.plot(ρ1, np.full_like(ρ1, -1), c='black')
    ax.plot(ρ1, -(ρ1**2 / 4), c='black')

    # Turn normal axes off
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(False)

    # Add arrows to represent axes
    axes_arrows = {'arrowstyle': '<|-|>', 'lw': 1.3}
    ax.annotate('', xy=(xmin, 0), xytext=(xmax, 0), arrowprops=axes_arrows)
    ax.annotate('', xy=(0, ymin), xytext=(0, ymax), arrowprops=axes_arrows)

    # Annotate the plot with equations
    plot_arrowsl = {'arrowstyle': '-|>', 'connectionstyle': "arc3, rad=-0.2"}
    plot_arrowsr = {'arrowstyle': '-|>', 'connectionstyle': "arc3, rad=0.2"}
    ax.annotate(r'$\phi_1 + \phi_2 < 1$', xy=(0.5, 0.3), xytext=(0.8, 0.6),
                arrowprops=plot_arrowsr, fontsize='12')
    ax.annotate(r'$\phi_1 + \phi_2 = 1$', xy=(0.38, 0.6), xytext=(0.6, 0.8),
                arrowprops=plot_arrowsr, fontsize='12')
    ax.annotate(r'$\phi_2 < 1 + \phi_1$', xy=(-0.5, 0.3), xytext=(-1.3, 0.6),
                arrowprops=plot_arrowsl, fontsize='12')
    ax.annotate(r'$\phi_2 = 1 + \phi_1$', xy=(-0.38, 0.6), xytext=(-1, 0.8),
                arrowprops=plot_arrowsl, fontsize='12')
    ax.annotate(r'$\phi_2 = -1$', xy=(1.5, -1), xytext=(1.8, -1.3),
                arrowprops=plot_arrowsl, fontsize='12')
    ax.annotate(r'${\phi_1}^2 + 4\phi_2 = 0$', xy=(1.15, -0.35),
                xytext=(1.5, -0.3), arrowprops=plot_arrowsr, fontsize='12')
    ax.annotate(r'${\phi_1}^2 + 4\phi_2 < 0$', xy=(1.4, -0.7),
                xytext=(1.8, -0.6), arrowprops=plot_arrowsr, fontsize='12')

    # Label categories of solutions
    ax.text(1.5, 1, 'Explosive\n growth', ha='center', fontsize=16)
    ax.text(-1.5, 1, 'Explosive\n oscillations', ha='center', fontsize=16)
    ax.text(0.05, -1.5, 'Explosive oscillations', ha='center', fontsize=16)
    ax.text(0.09, -0.5, 'Damped oscillations', ha='center', fontsize=16)

    # Add small marker to y-axis
    ax.axhline(y=1.005, xmin=0.495, xmax=0.505, c='black')
    ax.text(-0.12, -1.12, '-1', fontsize=10)
    ax.text(-0.12, 0.98, '1', fontsize=10)

    return fig, ax
    
    
def plot_eigenvalues(eigs, include_kde=False, cmap=None):
    if cmap is None:
        cmap = plt.cm.tab20
    
    n, k = eigs.shape

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect='equal'))
    circ = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(circ), np.sin(circ), c='k', ls='--', lw=0.5)
    
    real_part = eigs.real.ravel()
    imag_part = eigs.imag.ravel()
    if include_kde:
        sns.kdeplot(x=real_part, y=imag_part, ax=ax, shade=True)

    scatter = ax.scatter(real_part, 
                         imag_part, 
                         c=np.tile(np.arange(k), n), 
                         cmap=cmap,
                         alpha=0.35, s=10)
    fig.legend(*scatter.legend_elements(), loc='center right')
    plt.show()