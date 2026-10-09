import numpy as np, sys, os
out = sys.argv[1]
INK="#222222"; GREY="#9a9a9a"
def esc(t): return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def text(x,y,t,size=14,anchor="middle",weight="normal",color=INK):
    return f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{esc(t)}</text>'
def svg(w,h,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Arial, sans-serif"><rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>'

# ---------- 1. weight matrices
pal = ["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#8c564b","#e377c2","#17becf","#bcbd22"]
def matrix(x0,y0,M,c,labels_rows=None):
    b=[]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M[i,j]
            fill = "#ffffff" if v<0 else (pal[v] if v<len(pal) else "#999")
            b.append(f'<rect x="{x0+j*c}" y="{y0+i*c}" width="{c}" height="{c}" fill="{fill}" fill-opacity="{0.85 if v>=0 else 1}" stroke="#bbbbbb" stroke-width="1"/>')
    b.append(f'<rect x="{x0}" y="{y0}" width="{M.shape[1]*c}" height="{M.shape[0]*c}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    return "".join(b)
c=22
# FC: 36 distinct -> use hatch of random colours from a big palette
rng=np.random.default_rng(0)
body=[text(450,26,"Weight matrices: fully connected vs convolution (6 inputs)",17,weight="bold")]
# FC panel
x0,y0=40,70
cols=["#%02x%02x%02x"%tuple(rng.integers(90,230,3)) for _ in range(36)]
for i in range(6):
    for j in range(6):
        body.append(f'<rect x="{x0+j*c}" y="{y0+i*c}" width="{c}" height="{c}" fill="{cols[i*6+j]}" stroke="#bbbbbb"/>')
body.append(f'<rect x="{x0}" y="{y0}" width="{6*c}" height="{6*c}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
body.append(text(x0+3*c,y0-12,"fully connected",14,weight="bold"))
body.append(text(x0+3*c,y0+6*c+22,"36 weights, all different",13))
# conv k3 pad
def band(n,k=3,offset=0):
    M=-np.ones((n,n),int)
    for i in range(n):
        for t in range(k):
            j=i+t-1
            if 0<=j<n: M[i,j]=t+offset
    return M
x1=230
body.append(matrix(x1,y0,band(6),c))
body.append(text(x1+3*c,y0-12,"conv, kernel 3, zero-pad",14,weight="bold"))
body.append(text(x1+3*c,y0+6*c+22,"3 weights, repeated on a band",13))
body.append(text(x1+3*c,y0+6*c+40,"white = 0 (no connection)",12,color=GREY))
# stride 2
M=-np.ones((3,6),int)
for i in range(3):
    for t in range(3):
        j=2*i+t-1
        if 0<=j<6: M[i,j]=t
x2=420
body.append(matrix(x2,y0,M,c))
body.append(text(x2+3*c,y0-12,"stride 2",14,weight="bold"))
body.append(text(x2+3*c,y0+3*c+22,"3×6: every other row",13))
# Problem 10.6
x3=640; y3=50
M2=np.vstack([band(6,offset=0),band(6,offset=3)])
body.append(matrix(x3,y3+20,M2,c*0.85))
cc=c*0.85
body.append(text(x3+3*cc,y3+8,"Problem 10.6: 2 channels",14,weight="bold"))
body.append(text(x3+6*cc+8,y3+20+3*cc+4,"h1–h6 (ω1,ω2,ω3)",12,anchor="start"))
body.append(text(x3+6*cc+8,y3+20+9*cc+4,"h7–h12 (ω4,ω5,ω6)",12,anchor="start"))
body.append(f'<line x1="{x3}" y1="{y3+20+6*cc}" x2="{x3+6*cc}" y2="{y3+20+6*cc}" stroke="{INK}" stroke-width="2"/>')
body.append(text(x3+3*cc,y3+20+12*cc+22,"12×6, 6 weights + 2 biases",13))
# legend rows/cols
body.append(text(20,y0+3*c,"outputs h",12,color=GREY).replace('x="20"','x="20" transform="rotate(-90 20 '+str(int(y0+3*c))+')"'))
body.append(text(x0+3*c,y0-30,"inputs x1 … x6 →",12,color=GREY))
open(os.path.join(out,"weight-matrices.svg"),"w").write(svg(900,330,"".join(body)))

# ---------- 2. AlexNet params
L=[("conv1  11×11×3→96",11*11*3*96+96),("conv2  5×5×96→256",5*5*96*256+256),("conv3  3×3×256→384",3*3*256*384+384),
   ("conv4  3×3×384→384",3*3*384*384+384),("conv5  3×3×384→256",3*3*384*256+256),("fc6  9216→4096",9216*4096+4096),
   ("fc7  4096→4096",4096*4096+4096),("fc8  4096→1000",4096*1000+1000)]
tot=sum(v for _,v in L)
W,H=820,40+len(L)*34+60
b=[text(W/2,26,f"AlexNet parameters per layer (total {tot:,})",17,weight="bold")]
x0=210; maxw=480; mx=max(v for _,v in L)
for k,(n,v) in enumerate(L):
    y=50+k*34
    col="#1f77b4" if n.startswith("conv") else "#d62728"
    b.append(text(x0-10,y+17,n,13,anchor="end"))
    w=max(2,v/mx*maxw)
    b.append(f'<rect x="{x0}" y="{y+4}" width="{w:.0f}" height="22" fill="{col}" fill-opacity="0.8"/>')
    b.append(text(x0+w+8,y+20,f"{v:,}  ({100*v/tot:.1f}%)",13,anchor="start"))
yb=50+len(L)*34+18
conv=sum(v for n,v in L if n.startswith("conv"))
b.append(text(W/2,yb,f"Conv layers: {conv:,} ({100*conv/tot:.0f}%)  ·  Fully connected: {tot-conv:,} ({100*(tot-conv)/tot:.0f}%)",14,weight="bold"))
b.append(text(W/2,yb+20,"weights = k·k·C_in·C_out (conv) or D_in·D_out (fc), plus one bias per output channel/unit",12,color=GREY))
open(os.path.join(out,"alexnet-params.svg"),"w").write(svg(W,H,"".join(b)))

# ---------- 3. down / upsampling
X=np.array([[1,3,2,1],[4,2,0,1],[0,1,5,2],[2,3,1,0]])
sub=X[::2,::2]; mxp=np.array([[X[2*i:2*i+2,2*j:2*j+2].max() for j in range(2)] for i in range(2)])
mean=np.array([[X[2*i:2*i+2,2*j:2*j+2].mean() for j in range(2)] for i in range(2)])
blk=["#cfe3f5","#f8d0d0","#d3efd3","#e6dbf2"]
def g(x0,y0,M,c,block=False,fmtf=lambda v:str(v),hi=None):
    s=[]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            f = blk[(i//2)*2+(j//2)] if block else blk[i*2+j] if M.shape[0]==2 else "#ffffff"
            if hi is not None: f = blk[(i//2)*2+(j//2)] if hi[i,j] else "#ffffff"
            s.append(f'<rect x="{x0+j*c}" y="{y0+i*c}" width="{c}" height="{c}" fill="{f}" stroke="{INK}" stroke-width="1"/>')
            s.append(text(x0+j*c+c/2,y0+i*c+c/2+5,fmtf(M[i,j]),14))
    return "".join(s)
c=30
b=[text(450,24,"Downsampling 4×4 → 2×2 and upsampling 2×2 → 4×4",17,weight="bold")]
b.append(text(85,55,"input",13,weight="bold")); b.append(g(25,65,X,c,block=True))
for k,(name,M,fm) in enumerate([("sub-sample (stride 2)",sub,str),("max pool",mxp,str),("mean pool",mean,lambda v:f"{v:g}")]):
    x=230+k*190
    b.append(text(x+c,55,name,13,weight="bold")); b.append(g(x,80,M,c,fmtf=fm))
b.append(f'<line x1="160" y1="125" x2="215" y2="125" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
y2=240
U=np.array([[5,2],[4,6]])
dup=np.kron(U,np.ones((2,2),int))
# max unpool: positions of maxima from X-based max pool (use argmax locations of X blocks)
un=np.zeros((4,4),int); mask=np.zeros((4,4),bool)
for i in range(2):
    for j in range(2):
        blkX=X[2*i:2*i+2,2*j:2*j+2]; a,bb=np.unravel_index(blkX.argmax(),(2,2))
        un[2*i+a,2*j+bb]=U[i,j]; mask[2*i+a,2*j+bb]=True
b.append(text(55,y2-10,"input",13,weight="bold")); b.append(g(25,y2,U,c))
b.append(text(260,y2-10,"duplicate (nearest)",13,weight="bold")); b.append(g(200,y2,dup,c,block=True))
b.append(text(470,y2-10,"max unpool",13,weight="bold")); b.append(g(410,y2,un,c,hi=mask))
b.append(text(560,y2+50,"values go back to where",12,anchor="start",color=GREY))
b.append(text(560,y2+66,"the maxima were in the",12,anchor="start",color=GREY))
b.append(text(560,y2+82,"max pool above",12,anchor="start",color=GREY))
b.append(text(450,y2+150,"Other upsamplers: bilinear interpolation, transposed convolution (learned; weight matrix = transpose of a strided conv)",12,color=GREY))
defs='<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#222"/></marker></defs>'
open(os.path.join(out,"down-up-sampling.svg"),"w").write(svg(900,410,defs+"".join(b)))
print(sub,mxp,mean,un,sep="\n")
