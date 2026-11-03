"""Presentation-only font and print-box settings for retained flywheel plots."""
from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT=Path(__file__).resolve().parents[1]
TARGETS=ROOT/'data/flywheel_typography.json'
_rendered={}

class _OffsetFormatter(FixedFormatter):
    def __init__(self,labels,offset):
        super().__init__(labels)
        self.offset=offset
    def get_offset(self):
        return self.offset

def style_figure(fig,output):
    name=Path(output).name
    spec=json.loads(TARGETS.read_text())['figures'].get(name)
    if spec is None or len(fig.axes)!=len(spec['axes']):
        return
    plt.rcParams.update({'font.family':'Liberation Serif','mathtext.fontset':'custom',
                         'mathtext.rm':'Liberation Serif','mathtext.it':'Liberation Serif:italic',
                         'mathtext.bf':'Liberation Serif:bold','mathtext.fallback':None,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig.set_size_inches(*spec['canvas_inches'],forward=True)
    for ax,old in zip(fig.axes,spec['axes']):
        assert list(ax.get_xlim())==old['xlim'] and list(ax.get_ylim())==old['ylim']
        assert ax.get_xscale()==old['xscale'] and ax.get_yscale()==old['yscale']
        for axis,obj in [('x',ax.xaxis),('y',ax.yaxis)]:
            ticks=old['ticks'][axis]
            reflow=lambda labels:[spec.get('tick_reflow',{}).get(label,label) for label in labels]
            obj.set_major_locator(FixedLocator(ticks['major']))
            obj.set_minor_locator(FixedLocator(ticks['minor']))
            if len(ticks['major_text'])==len(ticks['major']):
                obj.set_major_formatter(_OffsetFormatter(reflow(ticks['major_text']),ticks['offset']))
            if len(ticks['minor_text'])==len(ticks['minor']):
                obj.set_minor_formatter(FixedFormatter(reflow(ticks['minor_text'])))
    for _ in range(2):
        for text in fig.findobj(Text):
            if text.get_text() in spec.get('text_reflow',{}):
                text.set_text(spec['text_reflow'][text.get_text()])
            text.set_fontfamily('Liberation Serif')
            text.set_math_fontfamily('custom')
            minimum=10.5 if '$' in text.get_text() and any(c in text.get_text() for c in '^_') else 8.5
            text.set_fontsize(max(minimum,text.get_fontsize()))
        fig.canvas.draw()
    for ax in fig.axes:
        if 'x_tick_fontsize' in spec:
            ax.tick_params(axis='x',which='both',labelsize=spec['x_tick_fontsize'])
        if 'legend_fontsize' in spec and ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(spec['legend_fontsize'])
    fig.tight_layout(pad=.5,w_pad=1.0,h_pad=.8)
    if spec.get('first_legend_below'):
        ax=fig.axes[0]
        box=ax.get_position()
        ax.set_position([box.x0,box.y0+.10,box.width,box.height-.10])
        ax.legend(loc='lower center',bbox_to_anchor=(.22,.02),bbox_transform=fig.transFigure,
                  ncols=3,fontsize=8.5,handlelength=.8,handletextpad=.25,columnspacing=.7,
                  borderpad=.2,frameon=True,facecolor='white',framealpha=.88,edgecolor='none')
    _rendered[name]=spec

def finalize_typography_receipts(generator):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    font=Path(font_manager.findfont('Liberation Serif',fallback_to_default=False))
    for name,spec in _rendered.items():
        output=ROOT/name;receipt=ROOT/'data'/spec['provenance']
        data=json.loads(receipt.read_text()) if receipt.exists() else {
            'schema':'whitepaper.telemetry-figure/v1',
            'input_csv_sha256':{p:sha(ROOT/'data'/p) for p in ['rounds.csv','admissions.csv']}}
        data['figure_sha256']=sha(output)
        if name=='fig_fw_target_history.pdf':
            data['reproduce']='python3 figures/scripts/make_promotions_figure.py --committed-target-history'
        if 'generator_sha256' in data:
            data['generator_sha256']=sha(generator)
        if 'figure_inches' in data:
            data['figure_inches']=spec['canvas_inches']
        if 'minimum_text_size_pt' in data:
            data['minimum_text_size_pt']=7.0
        data['rendering']={'font_family':'Liberation Serif','font_file_sha256':sha(font),
                           'style_helper_sha256':sha(__file__),'style_manifest_sha256':sha(TARGETS),
                           'canvas_inches':spec['canvas_inches'],'minimum_print_target_pt':7.0,
                           'tick_values_and_strings':'Preserved from the pre-refresh committed generator rendering.',
                           'scientific_inputs_unchanged':True}
        receipt.write_text(json.dumps(data,indent=2)+'\n')
