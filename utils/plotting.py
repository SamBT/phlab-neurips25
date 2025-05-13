import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
import matplotlib 
import seaborn as sns

def make_corner(x,labels,label_names=None,axwidth=2,return_fig=False):
    N = x.shape[1]
    blo,bhi = x.min(axis=0), x.max(axis=0)
    fig,axes = plt.subplots(N,N,figsize=(N*axwidth,N*axwidth))
    for i in range(N):
        for j in range(N):
            plt.sca(axes[i,j])
            plt.axis('off')

    unique_labels = sorted(list(set(labels)))
    patches = []
    xlims = [[np.quantile(x[:,i],0.00),np.quantile(x[:,i],1.0)] for i in range(N)]
    bins = [np.linspace(xlims[i][0],xlims[i][1],20) for i in range(N)]
    for il,label in enumerate(unique_labels):
        mask = labels==label
        xlims = []
        for i in range(N):
            plt.sca(axes[i,i])
            plt.axis('on')
            h = plt.hist(x[mask,i],bins=bins[i],density=True,histtype='step',color=f"C{il}")
            #xlims.append(plt.gca().get_xlim())

        for i in range(1,N):
            for j in range(i):
                plt.sca(axes[i,j])
                plt.scatter(x[mask,j],x[mask,i],s=0.5,color=f"C{il}")
                plt.xlim(axes[j,j].get_xlim())
        
        patches.append(Patch(label=label_names[label] if label_names is not None else label,color=f"C{il}"))
    
    plt.sca(axes[0,-1])
    plt.legend(handles=patches,ncol=3)
    if return_fig:
        return fig
        

def cmap_alpha(base_color='red', n_colors=256):
    # Create a list of colors with increasing alpha
    colors = []
    for i in range(n_colors):
        # Calculate alpha from 0.1 (mostly transparent) to 1.0 (fully opaque)
        alpha = 0.1 + 0.7 * i / (n_colors + 1)
        # If base_color is a string, convert to RGB
        if isinstance(base_color, str):
            rgb = mcolors.to_rgb(base_color)
        else:
            rgb = base_color
        # Create a color with the specified alpha
        colors.append((*rgb, alpha))

    cmap = mcolors.ListedColormap(colors, name=f"{base_color}_alpha")
    return cmap


def make_corner2(x, labels, label_names=None, axwidth=3, return_fig=False, 
                kde=False, subsample=True, label_map=None):
    N = x.shape[1]

    name_map = dict(enumerate([r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\Delta$"]))
    if label_map is not None:
        # merge by labels 
        new_labels = np.zeros_like(labels)
        for i, label in enumerate(label_map):
            for j in label_map[label]:
                new_labels[labels == j] = i
        labels = new_labels

    fig,axes = plt.subplots(N, N, figsize=(N*axwidth,N*axwidth), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(N):
        for j in range(N):
            plt.sca(axes[i,j])
            plt.axis('off')

    unique_labels = sorted(list(set(labels)))
    color_palettes = {}
    import seaborn as sns
    for il, label in enumerate(unique_labels):
        # Create a sequential palette from the base color
        base_color = f"C{il}"
        color_palettes[label] = sns.light_palette(base_color, as_cmap=True, )
    
    patches = []
    xlims = [[np.quantile(x[:,i],0.00),np.quantile(x[:,i],1.0)] for i in range(N)]
    bins = [np.linspace(xlims[i][0], xlims[i][1],20) for i in range(N)]
    for il,label in enumerate(unique_labels):
        mask = labels==label
        xlims = []
        for i in range(N):
            ax = axes[i,i]
            ax.axis('on')
            if not kde:
                ax.hist(x[mask,i], bins=bins[i], density=True, histtype='step',color=f"C{il}")
            else:
                import seaborn as sns
                sns.kdeplot(x[mask,i], color=f"C{il}", ax=ax, fill=True, alpha=0.5)
                ax.set_ylabel("")
            ax.set_title(f"{name_map[i]}", fontsize='large')
            ax.set_yticks([])
            ax.set_xticks([])
        for i in range(N):
            ax = axes[i,0]
            ax.set_ylabel(f"{name_map[i]}", fontsize='large', ha='center')
        for i in range(N):
            ax = axes[-1,i]
            ax.set_xlabel(f"{name_map[i]}", fontsize='large', ha='center')

        for i in range(1,N):
            for j in range(i):
                ax = axes[i,j]
                _x, _y = x[mask,j], x[mask,i]
                if not kde:
                    ax.scatter(_x, _y, s=0.5, color=f"C{il}", alpha=0.5)
                else:
                    if subsample:
                        if len(_x) > 1000:
                            idx = np.random.choice(len(_x), 1000, replace=False)
                            _x, _y = _x[idx], _y[idx]
                    Nlevels = 20
                    linear_space = np.linspace(0, 1, Nlevels)[1:]
                    transformed_space = 1 - np.exp(-linear_space * 1.5)
                    transformed_space = np.r_[transformed_space, 1]
                    sns.kdeplot(x=_x, y=_y, ax=ax, cmap=cmap_alpha(f"C{il}", Nlevels), fill=False, linewidths=1,
                                levels=Nlevels, 
                                )
                ax.set_xlim(axes[j,j].get_xlim())
                ax.axis('on')
                ax.set_yticks([])
                ax.set_xticks([])

        for ax in axes.flatten():
            ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        for i in range(N):
            axes[i, 0].spines[['left']].set_visible(True)
            axes[-1, i].spines[['bottom']].set_visible(True)

        for ax in axes.flatten():
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')

        patches.append(Patch(label=label_names[label] if label_names is not None else label, 
                             edgecolor=f"C{il}", facecolor=matplotlib.colors.to_rgba(f"C{il}", 0.5)
                             ))
    
    leg = axes[1, 1].legend(handles=patches,ncol=3, frameon=False, title=f'Classes', 
                       loc='upper center', bbox_to_anchor=(1+(N-2)/2, 1),
                       fontsize='large')
    plt.setp(leg.get_title(), fontsize='x-large')
    axes[0, 0].text(1+(N-1)/2, 0.5, (f"Contrastive space of {N} dimensions - "+"{"+",".join(list(name_map.values())[:N])+"}"),
                    fontsize='xx-large', ha='center', transform=axes[0, 0].transAxes)
    if return_fig:
        return fig


if "__name__" == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    # Generate some sample data
    X, y = make_moons(n_samples=1000, noise=0.2 )
    X = np.concatenate([X, np.random.rand(1000, 2)], axis=1)  # Add two more dimensions
    # Create a dictionary to simulate the input from npz files
    test_arrays = {
        'test': {
            'data': X,
            'label': y
        }
    }

    fig = make_corner2(test_arrays['test']['data'], test_arrays['test']['label'], return_fig=True,
                label_map={0: [0], 1: [1]}, kde=False,
                label_names={0: "All bkg", 1: "Anomaly"}
    )
    fig.show()