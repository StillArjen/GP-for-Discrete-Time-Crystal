"""Plot exact-model numerical checks from saved NPZ data, with no interpolation.

Figure contract: quantitative grids compare the GP response with a conventional
observable on the identical grid. No phase labels are inferred. Each pixel is a
deterministic Lindblad calculation, not an experimental replicate. PNG previews
and editable SVG figures are exported; NPZ files retain all source values.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})


def save(fig, prefix):
    fig.savefig(str(prefix)+'.png',dpi=300,bbox_inches='tight')
    fig.savefig(str(prefix)+'.svg',bbox_inches='tight')
    fig.savefig(str(prefix)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def mesh(ax,x,y,z,label,cmap='viridis',vmin=None,vmax=None):
    artist=ax.pcolormesh(x,y,z,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
    plt.colorbar(artist,ax=ax,label=label,pad=.02)


def kerr(path):
    d=np.load(path)
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    panels=[(d['nbar'],'Mean photon number','viridis',None,None),
            (d['phase'],'Full-loop GP (rad)','twilight',-np.pi,np.pi),
            (abs(d['Z']),'Geometric visibility','viridis',0,1),
            (np.angle(d['Z_half']),'Half-loop GP (rad)','twilight',-np.pi,np.pi)]
    for idx,(ax,(z,title,cmap,lo,hi)) in enumerate(zip(axs.flat,panels)):
        mesh(ax,d['delta'],d['eta'],z,title,cmap,lo,hi)
        ax.set(xlabel='Detuning / |U|',ylabel='Squeezing strength / |U|',title=f'{chr(97+idx)}  {title}')
    fig.suptitle(f'Kerr steady-state check | Fock dimension {int(d["cutoff"])}',fontsize=10)
    save(fig,path.with_suffix(''))


def dicke(path, traces_data=None):
    d=np.load(path)
    metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    p=metadata['parameters']
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    panels=[('late_amplitude_sigma_x','Magnetization'),
            ('late_amplitude_toy_scaled_g_xy','Transverse-generator GP signal'),
            ('late_amplitude_S','Paired-generator contrast'),
            ('late_amplitude_Z_n','Number-generator control')]
    tail=np.array([r['max_tail_population_at_half_period_samples'] for r in metadata['points']])
    tail=tail.reshape(len(d['kappa_values']),len(d['epsilon_values']))
    warned=tail>1e-5
    xx,yy=np.meshgrid(d['epsilon_values'],d['kappa_values'])
    for idx,(ax,(key,title)) in enumerate(zip(axs.flat,panels)):
        mesh(ax,d['epsilon_values'],d['kappa_values'],d[key],'Alternating-mean amplitude',vmin=0)
        ax.scatter(xx[warned],yy[warned],marker='x',c='white',s=14,linewidths=.8)
        ax.set(xlabel='Frequency detuning',ylabel='Cavity loss rate',title=f'{chr(97+idx)}  {title}')
    fig.suptitle(f'Finite-time Dicke response | {p["atoms"]} atoms, {p["periods"]} periods\n'
                 f'Fock dimension {p["cutoff"]}; last {p["window"]} samples; x marks Fock-tail warnings',fontsize=9)
    save(fig,path.with_suffix(''))

    # Show all strobe samples; the two predeclared detunings are from the paper.
    if traces_data is not None:
        d=np.load(traces_data)
        metadata=json.loads(traces_data.with_suffix('.json').read_text(encoding='utf-8'))
        p=metadata['parameters']
    kap=int(np.argmin(abs(d['kappa_values']-.05)))
    picks=[int(np.argmin(abs(d['epsilon_values']-eps))) for eps in [-.056,.01]]
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
    colors=['#355F8D','#D17832','#628F70']
    for row,idx in enumerate(picks):
        for col,demodulate in enumerate([False,True]):
            ax=axs[row,col]
            sign=(-1.)**d['m'] if demodulate else np.ones(len(d['m']))
            for key,label,color in zip(['sigma_x','toy_scaled_g_xy','S'],
                                      ['Magnetization','Transverse GP signal','Paired GP contrast'],colors):
                ax.plot(d['m'],sign*d[key][kap,idx],color=color,label=label,lw=1)
            ax.axhline(0,color='0.7',lw=.5)
            ax.set(xlabel='Drive periods',ylabel='Demodulated signal' if demodulate else 'Stroboscopic signal',
                   title=f'{chr(97+row*2+col)}  detuning={d["epsilon_values"][idx]:g}')
            if row==0 and col==1:
                ax.legend(fontsize=6.5,loc='best')
    fig.suptitle(f'Exact finite-system trajectories | {p["atoms"]} atoms, Fock dimension {p["cutoff"]}, loss={d["kappa_values"][kap]:g}\n'
                 'Right column removes the alternating sign',fontsize=9)
    save(fig,path.with_name(path.stem+'_traces'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['kerr','dicke'])
    parser.add_argument('data',type=Path)
    parser.add_argument('--traces-data',type=Path,help='Optional larger-cutoff data for Dicke traces')
    args=parser.parse_args()
    if args.kind=='kerr':
        kerr(args.data)
    else:
        dicke(args.data,args.traces_data)


if __name__=='__main__':
    main()
