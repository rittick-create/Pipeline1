"""Render the ECGFounder architecture diagram as PNG and editable SVG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(22, 16))
fig.patch.set_facecolor('#f8fafc')
ax.set(xlim=(0,22), ylim=(0,16))
ax.axis('off')
INK='#152b45'; BLUE='#2875b8'; ORANGE='#ba6424'; GREEN='#287b66'

def box(x,y,w,h,title,detail='',face='#ffffff',edge='#c6d4e2',size=13):
    patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.025,rounding_size=0.14',
                         facecolor=face,edgecolor=edge,linewidth=1.5,zorder=4)
    ax.add_patch(patch)
    ax.text(x,y+(0.18 if detail else 0),title,ha='center',va='center',fontsize=size,
            fontweight='bold',color=INK,zorder=5)
    if detail:
        ax.text(x,y-0.29,detail,ha='center',va='center',fontsize=10.5,color='#40566f',zorder=5,linespacing=1.4)

def arrow(points,color=INK,width=1.8):
    # White outlines distinguish paths where lines cross without joining.
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=width,zorder=2,
                path_effects=[pe.Stroke(linewidth=width+4,foreground='#f8fafc'),pe.Normal()])
    a,b=points[-2:]
    line=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,lw=width,zorder=3)
    line.set_path_effects([pe.Stroke(linewidth=width+3,foreground='#f8fafc'),pe.Normal()])
    ax.add_patch(line)

ax.text(11,15.55,'Pipeline2 · ECGFounder architecture',ha='center',fontsize=25,fontweight='bold',color=INK)
ax.text(11,15.08,'B = batch size    H = 256 features    C = 14 findings',ha='center',fontsize=13,color='#52657a')

centers=[3.5,11,18.5]
# Light panels group each independent modality encoder.
for x in centers:
    p=FancyBboxPatch((x-3.15,8.45),6.3,6.1,boxstyle='round,pad=0.04,rounding_size=0.2',
                    facecolor='#eef3f8',edgecolor='#dce5ee',zorder=0)
    ax.add_patch(p)

inputs=[('ECG','Stored SYMILE: [B, 1, 5000, 12]'),
        ('Laboratory tests','50 percentile values + 50 observed indicators'),
        ('Chest X-ray','SYMILE image tensor: [B, 3, 320, 320]')]
prep=[('ECGFounder preprocessing','Lead reorder · notch + bandpass\nBaseline removal · global z-score'),
      ('Concatenate values + indicators','Input: [B, 100]'),
      ('Use existing SYMILE preprocessing','No additional image preprocessing in the loader')]
enc=[('Pretrained ECGFounder · Net1D','1D residual convolution encoder\nGlobal average pooling → [B, 1024]'),
     ('Lab feature MLP','Linear(100 → 256) · ReLU · dropout\nOutput: [B, 256]'),
     ('ResNet50 stem','Conv1 · batch norm · ReLU · max pool\nFeeds two independent ResNet branches')]
for x,(title,detail),(pt,pd),(et,ed) in zip(centers,inputs,prep,enc):
    box(x,13.85,5.9,0.85,title,detail)
    box(x,12.48,5.9,1.15,pt,pd,size=12)
    box(x,10.9,5.9,1.35,et,ed,size=13)
    arrow([(x,13.4),(x,13.08)])
    arrow([(x,11.88),(x,11.6)])
    arrow([(x,10.2),(x,9.96),(x-1.52,9.96),(x-1.52,9.65)],BLUE)
    arrow([(x,10.2),(x,9.96),(x+1.52,9.96),(x+1.52,9.65)],ORANGE)

for x,name in zip(centers,['ECG','Labs','CXR']):
    shared_detail='Linear 1024 → H' if name=='ECG' else ('Linear H → H' if name=='Labs' else 'ResNet layers 1–4\nPool · Linear 2048 → H')
    box(x-1.52,9.1,2.85,1.05,'Shared',shared_detail,face='#e5f0fc',edge=BLUE,size=13)
    box(x+1.52,9.1,2.85,1.05,'Distinct',shared_detail,face='#fff0e4',edge=ORANGE,size=13)
    # The same projection MLP is applied separately to every shared vector.
    arrow([(x-1.52,8.55),(x-1.52,7.65),(11,7.65),(11,7.14)],BLUE)

ax.text(11,8.04,'Shared projection MLP (same weights for all three modalities): H → 2H → H → H',
        ha='center',fontsize=11.5,color=BLUE,bbox=dict(facecolor='#f8fafc',edgecolor='none',pad=3),zorder=6)
box(11,6.53,5.7,1.16,'Shared fusion · logit pooling',
    'logit(mean(sigmoid(projected shared vectors)))\nFused representation: [B, H]',face='#e5f0fc',edge=BLUE,size=14)

# Four feature tokens for the complete-input condition.
tokens=[(3.5,'ECG distinct'),(8.5,'Labs distinct'),(13.5,'Fused shared'),(18.5,'CXR distinct')]
for x,label in tokens:
    color=BLUE if label=='Fused shared' else ORANGE
    box(x,4.45,4.45,0.9,label,'[B, H]',face='#e5f0fc' if color==BLUE else '#fff0e4',edge=color,size=13)
arrow([(5.02,8.55),(5.02,5.7),(3.5,5.7),(3.5,4.93)],ORANGE)
arrow([(12.52,8.55),(14.25,8.55),(14.25,5.47),(8.5,5.47),(8.5,4.93)],ORANGE)
arrow([(20.02,8.55),(20.02,5.7),(18.5,5.7),(18.5,4.93)],ORANGE)
arrow([(11,5.93),(11,5.22),(13.5,5.22),(13.5,4.93)],BLUE)

ax.text(1,5.04,'EMBEDDINGS',fontsize=10,fontweight='bold',color='#52657a')
for x,_ in tokens:
    arrow([(x,3.98),(x,3.63),(11,3.63),(11,3.36)],GREEN)
box(11,2.68,17.4,1.32,'Disease-aware attention · 4 input tokens',
    'Stack [B, 4, 256]  →  Q + V + 14 disease keys  →  attention weights [B, 14, 4]\n'
    'Weighted sum of values → one embedding per finding: [B, 14, 256]',face='#e6f3ed',edge=GREEN,size=15)
arrow([(11,1.99),(11,1.56)],GREEN)
box(11,0.94,13.6,1.12,'Final prediction head',
    'Linear 256 → 14: [B, 14, 14]  →  diagonal  →  sigmoid  →  [B, 14]',face='#edf0fc',edge='#6865a8',size=15)

# Protocol and training heads are explicit notes, separate from the inference path.
fig.text(.025,.015,'Training: all modalities · auxiliary heads on each distinct vector and fused shared vector · masked labels (0/1 only)',
         fontsize=11,color=INK)
fig.text(.025,.001,'Testing: same best checkpoint → all modalities / CXR only / ECG + labs. Excluded encoders and feature tokens are omitted; no zero padding.',
         fontsize=11,color=INK)
fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.055)
for suffix in ('png','svg'):
    fig.savefig(OUT/f'model_architecture.{suffix}',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.2)
print(OUT/'model_architecture.png')
plt.close(fig)
