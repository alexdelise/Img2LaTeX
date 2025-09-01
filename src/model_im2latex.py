import math, torch, torch.nn as nn

def build_2d_sincos(H,W,d):
    assert d%4==0; d4=d//4
    y=torch.arange(H).float().unsqueeze(1).repeat(1,W)
    x=torch.arange(W).float().unsqueeze(0).repeat(H,1)
    omg=torch.exp(torch.arange(d4).float()*(-math.log(10000.0)/d4))
    posx=torch.cat([torch.sin(x.unsqueeze(-1)*omg),torch.cos(x.unsqueeze(-1)*omg)],dim=-1)
    posy=torch.cat([torch.sin(y.unsqueeze(-1)*omg),torch.cos(y.unsqueeze(-1)*omg)],dim=-1)
    return torch.cat([posy,posx],dim=-1)

class CNNEncoder(nn.Module):
    def __init__(self,d_model=512):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,64,3,2,1),nn.ReLU(),
            nn.Conv2d(64,128,3,2,1),nn.ReLU(),
            nn.Conv2d(128,256,3,2,1),nn.ReLU(),
            nn.Conv2d(256,d_model,3,1,1),nn.ReLU())
    def forward(self,x):
        f=self.net(x); B,C,H,W=f.shape
        pe=build_2d_sincos(H,W,C).to(f.device).permute(2,0,1).unsqueeze(0)
        f=f+pe; tok=f.flatten(2).transpose(1,2)
        return tok.transpose(0,1)

class TransformerDecoder(nn.Module):
    def __init__(self,vocab,d_model=512,nhead=8,nlayers=6):
        super().__init__()
        self.emb=nn.Embedding(vocab,d_model); self.pos=nn.Embedding(2048,d_model)
        layer=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward=4*d_model,batch_first=False)
        self.dec=nn.TransformerDecoder(layer,nlayers); self.proj=nn.Linear(d_model,vocab)
    def forward(self,y_in,mem):
        B,T=y_in.shape; pos=torch.arange(T,device=y_in.device)
        tgt=self.emb(y_in).transpose(0,1)+self.pos(pos).unsqueeze(1)
        mask=nn.Transformer.generate_square_subsequent_mask(T).to(y_in.device)
        h=self.dec(tgt,mem,tgt_mask=mask)
        return self.proj(h).transpose(0,1)

class Im2Latex(nn.Module):
    def __init__(self,vocab,d_model=512,nhead=8,nlayers=6):
        super().__init__(); self.enc=CNNEncoder(d_model); self.dec=TransformerDecoder(vocab,d_model,nhead,nlayers)
    def forward(self,x,y_in):
        mem=self.enc(x); return self.dec(y_in,mem)

def cross_entropy_smoothed(logits,y_out,pad_id=0,eps=0.1):
    V=logits.size(-1); logp=logits.log_softmax(-1)
    y=y_out.reshape(-1); lp=logp.reshape(-1,V)
    mask=(y!=pad_id).float()
    nll=-lp[torch.arange(y.numel()),y]*mask
    u=-lp.mean(-1)*mask
    return ((1-eps)*nll+eps*u).sum()/mask.sum().clamp_min(1.)
