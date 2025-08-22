# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %matplotlib inline

from rapid_code_load_T0 import load_COMPAS_data, load_BSE_data, load_COSMIC_data, load_SeBa_data, load_T0_data
# load_TO_data applies for Combine and SEVN
from sklearn.neighbors import KernelDensity
import formation_channels as fc
import arviz as az
az.style.use("arviz-doc")

# +

datadir = 'data/big_data/pilot_runs/'
def get_data(code):
    if code == 'COMPAS': 
        return load_COMPAS_data(datadir + 'COMPAS_pilot.h5')
    elif code == 'COSMIC': 
        return load_COSMIC_data(datadir + 'cosmic_pilot.h5', metallicity=0.02)
    elif code ==  'BSE':    
        return load_BSE_data(datadir + 'bse_pilot.dat', metallicity=0.02)
    #elif code ==  'SeBa':   
    #    return load_SeBa_data(datadir + 'Seba_BinCodex.h5', metallicity=0.02)
        
    else:
        print("Code unknown")
        return
        
#SEVN_mist = 'data/T0_format_pilot/MIST/setA/Z0.02/sevn_mist'
#sm, sm_header = load_T0_data(SEVN_mist, code='SEVN', metallicity=0.02)

data_names = ['COMPAS', 'COSMIC', 'BSE'] #, 'SeBa']  

print(data_names)

#test
get_data(data_names[3])

# +
# Setup single KDE

kde = KernelDensity(bandwidth=0.05, kernel='gaussian')

def get_grid(d, exp=0.1, num=50):
    d_range =  np.max(d) - np.min(d)
    d_lo = np.min(d) - exp*d_range
    d_hi = np.max(d) + exp*d_range
    return (d_lo, d_hi), np.linspace(d_lo, d_hi, num=num)

def plot_1D_kde(ax, data):
    # instantiate and fit the KDE model
    kde.fit(data[:, None])
    xlims, xgrid = get_grid(data)
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(xgrid[:, None])
    ax.fill_between(xgrid, np.exp(logprob), alpha=0.5)
    ax.set_xlim(xlims)
    
# Setup posterior for 2 params 

def plot_2D_kde(ax, xdata, ydata, skipby=50): 
    az.plot_kde(xdata, ydata, ax=ax, 
        #hdi_probs=[0.393, 0.865, 0.989],  # 1, 2 and 3 sigma contours
        #hdi_probs=[0.3, 0.6, 0.9],  # 30, 60, 90
        contour_kwargs={"levels":4},
        #contour_kwargs={'linewidths':2},
        contourf_kwargs={"alpha": 0, 
                         "levels":4,
                         "line_width":2,
                         "cmap": "Blues",
                        },
        #contourf_kwargs={"cmap": "Blues"},
    )
    ax.scatter(xdata[::skipby], ydata[::skipby], alpha=0.3, s=0.5, color='red', zorder=10)
    
    xlims, _ = get_grid(xdata)
    ylims, _ = get_grid(ydata)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

def make_corner_plots(data, figwidth=8, fs_label=18, ticklabelsize=10):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(figwidth, figwidth))
    
    ### Collect the data
    ZAMS, _, _ = fc.select_evolutionary_states(d=data)
    a = ZAMS.semiMajor 
    m1 = ZAMS.mass1
    m2 = ZAMS.mass2
    q = m2/m1
    
    params = [np.log10(m1), q, np.log10(a)]
    names = [ r"$\log_{10}(m_1)$", "$q$", r"$\log_{10}(a)$"]
    
    for ii in range(3):
        for jj in range(3):
            ax = axes[ii][jj]
            ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
            ax.spines.right.set_visible(True)
            ax.spines.top.set_visible(True)
    
            xdata = np.array(params[jj])
            ydata = np.array(params[ii])
            xlabel = names[jj]
            ylabel = names[ii]
            if (jj == 0) & (ii != 0):
                ax.set_ylabel(ylabel, fontsize=fs_label)
            if ii == 2: # not robust...
                ax.set_xlabel(xlabel, fontsize=fs_label)
            
            if ii == jj: # diagonal: 1D hists
                plot_1D_kde(ax, xdata)
                ax.set_yticklabels([])
                
            elif ii > jj: # lower triangle, 2D hists
                plot_2D_kde(ax, xdata, ydata, skipby=100) 
            else: # need to hide all elements of these
                ax.set_axis_off()
                continue




# +

make_corner_plots(data)
# -













# +

def plot_1D_kde(ax, data):
    # instantiate and fit the KDE model
    kde.fit(data[:, None])
    xlims, xgrid = get_grid(data)
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(xgrid[:, None])
    ax.fill_between(xgrid, np.exp(logprob), alpha=0.5)
    ax.set_xlim(xlims)
# -

















# +
np.unique(d.event)

# interacting events are 31, 32, 41, 42, 52 (no CEE?)
# -

#d.loc
interaction_events = [ 31, 32, 41, 42, 52 ]
ids_interactors = d.ID[np.in1d(d.event, interaction_events )].unique()
ids_noninteractors = d.ID[~np.in1d(d.ID, ids_interactors)].unique()
mask_init = (d.event == 13)
mask_t0_int = (d.event == 13) & np.in1d(d.ID, ids_interactors)
mask_t0_nonint = (d.event == 13) & np.in1d(d.ID, ids_noninteractors)
d.loc[mask_t0_int]
#d.loc[mask_t0_nonint]


# +

m1 = d.mass1
m2 = d.mass2
q = m2/m1
np.unique(q[mask_init], return_counts=True)

# +
fig, axes = plt.subplots()

# Plot 3 plots horizontally, for q=0.1, 0.5, and 0.9
# Plots should be M1 vs log(a), with an extra black dot for the non-interactors


# -


