import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def tjur_score(y_true, y_pred, average=False):

    k = y_true.shape[1]
    score_per_class = np.zeros(k)

    for i in range(k):
        score_0 = y_pred[y_true[:, i] == 0, i].mean()
        score_1 = y_pred[y_true[:, i] == 1, i].mean()
        score_per_class[i] = score_1 - score_0
    
    if average is None:
        return score_per_class
    else:
        return score_per_class.mean()


def plot_face(v, tri, overlay=None, cmap='RdBu', threshold=None, cmin=None, cmax=None, sym_cmap=True,
              width=None, height=None, col_titles=None, row_titles=None):
    """ Plot a face mesh in 3D.
    
    Parameters
    ----------
    v : np.ndarray
        A 2D or 3D numpy array, in which the first dimension represents different vertex meshes to be
        plotted (in rows), the second dimension represents the vertices, and the third dimension represents
        the coordinates (x, y, z)
    tri : np.ndarray
        A 2D array with the faces ("triangles")
    overlay : None or np.ndarray
        Overlay to be plotted on top of the vertex mesh. If it has the same shape as `v`, it will assume
        that each vertex mesh gets its own overlay. Each overlay in the last dimension is plotted in a different
        column
    cmap : str
        Name of Plotly colormap
    threshold : float
        Threshold to be applied to the overlay
    cmin : float/int
        Sets minimum of colorscale
    cmax : float/int
        Sets maximum of colorscale
    sym_cmap : bool
        Whether it is a symmetric colorscale
    width : int
        Width of figure
    height : int
        Height of figure
    
    Returns
    -------
    fig : plotly.Figure
        The resulting Figure
    """

    if overlay is None:
        # Plot darkgray face if no overlay
        color = 'darkgray'
        cmin, cmax = None, None
    else:
        # Determine cmax/cmin intelligently
        color = None
        if cmax is None:
            cmax = max([overlay.max(), np.abs(overlay.min())])
        
        if sym_cmap:
            # Equal distance from 0
            cmin = -cmax
        else:
            if cmin is None:
                cmin = overlay.min()

    # Define layout. I just tried random things until it looked nice
    layout = go.Layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=0, z=0))
    )

    # Determine number of rows/columns depending on v/overlay
    if overlay is not None:
        if overlay.ndim == 1:
            n_cols = 1
        else:
            n_cols = min(5, overlay.shape[-1])
    else:
        n_cols = 1
    
    if v.ndim == 3:
        n_rows = v.shape[0]
    else:
        n_rows = 1


    # Create figure!
    specs = [[{'type': 'surface'}] * n_cols] * n_rows
    fig = make_subplots(
        rows=n_rows, cols=n_cols, specs=specs,
        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0,
        horizontal_spacing=0, column_titles=col_titles, row_titles=row_titles
    )
    
    # Loop across rows (vertex meshes)
    for i in range(n_rows):

        if v.ndim == 3:
            this_v = v[i, :, :]
        else:
            this_v = v
        
        # Loop across columns (overlays)
        for ii in range(n_cols):

            # Note to self: copy() is necessary because otherwise
            # it mutates overlay
            if overlay is None:
                this_o = None
            elif overlay.ndim == 3:
                this_o = overlay[i, :, ii].copy()
            elif overlay.ndim == 2:
                this_o = overlay[:, ii].copy()
            else:
                this_o = overlay.copy()

            # Threshold strategy based on symmetric colorscale or not
            if threshold is not None and this_o is not None:
                if sym_cmap:
                    this_o[np.abs(this_o) < threshold] = np.nan
                else:
                    this_o[this_o < threshold] = np.nan

            # Finally, define mesh
            mesh = go.Mesh3d(
                y=this_v[:, 0], z=this_v[:, 1], x=this_v[:, 2],
                j=tri[:, 0], k=tri[:, 1], i=tri[:, 2],
                intensity=this_o, colorscale=cmap,
                cmax=cmax, cmin=cmin, reversescale=True,
                color=color, showscale=False
            )
            
            # Add mesh to figure
            fig.add_trace(mesh, row=i+1, col=ii+1)        

    # This is necessary to set the *same* scene for each subplot
    fig.update_scenes(layout['scene'])
    
    # The width/height below gives a nice aspect ratio
    if width is None:
        width = n_cols * 200
    
    if height is None:
        height = n_rows * 275
    
    fig.update_layout(width=width, height=height,
        margin=dict(l=0, b=0)
    )
    return fig


def get_parameters(sub, coef_df):
    """ Extracts parameters from TSV file. """
    if sub == 'average':
        # Average across participants
        if 'emotion' in coef_df.columns:
            Z = coef_df.groupby('emotion').mean().drop('sub', axis=1)
            alpha_hat, beta_hat = Z.loc[:, 'icept'].to_numpy(), Z.iloc[:, 1:].to_numpy()
            return alpha_hat, beta_hat, Z
        else:
            Z = coef_df.drop('sub', axis=1).mean(axis=0)
            sigma_hat, alpha_hat, beta_hat = Z.loc['sigma'], Z.loc['icept'], Z.iloc[2:].to_numpy()
            return alpha_hat, beta_hat, sigma_hat, Z
    else:
        Z = coef_df.query("sub == @sub").drop(['feature_set', 'sub'], axis=1)
        if 'emotion' in coef_df.columns:
            Z = Z.set_index('emotion')
            alpha_hat, beta_hat = Z.loc[:, 'icept'].to_numpy(), Z.iloc[:, 1:].to_numpy()
            return alpha_hat, beta_hat, Z
        else:
            Z = Z.iloc[0, :]
            sigma_hat, alpha_hat, beta_hat = Z.loc['sigma'], Z.loc['icept'], Z.iloc[2:].to_numpy()
            return alpha_hat, beta_hat, sigma_hat, Z
