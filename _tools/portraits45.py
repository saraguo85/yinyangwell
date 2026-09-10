"""Site portraits, all 4:5 to match .doc .av / .media / .about-art."""
import numpy as np, sys, os
from PIL import Image, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import person_mask

SRC  = "/Users/saraguo/content/TCM_Content/website/抱朴堂素材_已拆分/医生照片"
REPO = os.path.expanduser("~/content/TCM_Content/yinyangwell-website")
AR   = 0.8                                   # 4:5
BG   = np.array([0xE9,0xE6,0xDF], float)/255  # shared studio backdrop

def smoothstep(x,e0,e1):
    t=np.clip((x-e0)/(e1-e0),0,1); return t*t*(3-2*t)

def crop45(im, cx, eye, head_h, head_frac, eye_frac, mk=None, pad=True):
    H = head_h/head_frac; W = H*AR
    left, up = cx - W/2, eye - eye_frac*H
    b = (int(left), int(up), int(left+W), int(up+H))
    if not pad:   # keep the real background: clamp instead of padding
        iw, ih = im.size
        W = min(W, iw); H = W/AR
        left = max(0, min(cx-W/2, iw-W)); up = max(0, min(eye-eye_frac*H, ih-H))
        b = (int(left), int(up), int(left+W), int(up+H))
        return im.crop(b), None
    iw, ih = im.size
    px0, py0 = max(0,-b[0]), max(0,-b[1]); px1, py1 = max(0,b[2]-iw), max(0,b[3]-ih)
    if px0 or py0 or px1 or py1:
        im = ImageOps.expand(im,(px0,py0,px1,py1),fill=(200,200,200))
        if mk is not None: mk = ImageOps.expand(mk,(px0,py0,px1,py1),fill=0)
        b = (b[0]+px0,b[1]+py0,b[2]+px0,b[3]+py0)
    return im.crop(b), (mk.crop(b) if mk is not None else None)

def doctor(key, name, cx, head_top, chin, eye, out):
    im = Image.open(f'{SRC}/{name}.jpeg').convert('RGB')
    mk = Image.fromarray(person_mask(f'{SRC}/{name}.jpeg')[0]).resize(im.size, Image.LANCZOS)
    im, mk = crop45(im, cx, eye, chin-head_top, 0.40, 0.34, mk)
    W,H = 800,1000
    im = im.resize((W*2,H*2), Image.LANCZOS)
    mk = mk.resize((W*2,H*2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(2.2))
    m = np.clip((np.asarray(mk).astype(float)[...,None]/255 - .30)/.70, 0, 1)
    a = np.asarray(im).astype(float)/255
    subj = m[...,0]>.7; lum=a.mean(2)
    coat = subj & (lum >= np.percentile(lum[subj],95))
    cm = a[coat].mean(0)
    a = np.clip(a*(cm.mean()/cm)[None,None,:],0,1)
    a = np.clip(a*(0.92/a[coat].mean()),0,1)
    g = (a*np.array([.2126,.7152,.0722])).sum(2,keepdims=True)
    a = np.clip(g+(a-g)*0.74,0,1)
    a = a*m + BG[None,None,:]*(1-m)
    Image.fromarray((a*255).round().astype(np.uint8)).resize((W,H),Image.LANCZOS).save(out,quality=93,subsampling=0)
    print('  ', os.path.basename(out))

def drawing(src, out, ground, target, sat, size):
    im = Image.open(src).convert('RGB')
    w,h = im.size
    tw = h*AR
    im = im.crop((int((w-tw)/2),0,int((w-tw)/2+tw),h)) if tw<=w else im.crop((0,0,w,int(w/AR)))
    a = np.asarray(im).astype(float)/255
    G = np.array([int(ground[i:i+2],16) for i in (1,3,5)],float)/255
    T = np.array([int(target[i:i+2],16) for i in (1,3,5)],float)/255
    lum = a.mean(2,keepdims=True)
    a = np.clip(a*(1+(T/G-1)*smoothstep(lum,.06,.52)),0,1)
    g = (a*np.array([.2126,.7152,.0722])).sum(2,keepdims=True)
    a = np.clip(g+(a-g)*sat,0,1)
    a = np.clip((a-.055)/.945,0,1)
    Image.fromarray((a*255).round().astype(np.uint8)).resize(size,Image.LANCZOS).save(out,quality=93,subsampling=0)
    print('  ', os.path.basename(out))

def sara(src, out):
    im = Image.open(src).convert('RGB')
    im,_ = crop45(im, 634, 922, 1152, 0.40, 0.36, pad=False)
    a = np.asarray(im.resize((1600,2000),Image.LANCZOS)).astype(float)/255
    hi = np.percentile(a.reshape(-1,3), 97, axis=0)          # the white linen top is the neutral
    a = np.clip(a*(hi.mean()/hi)[None,None,:],0,1)
    g = (a*np.array([.2126,.7152,.0722])).sum(2,keepdims=True)
    a = np.clip(g+(a-g)*0.76,0,1)
    a = np.clip((a-.02)/.98,0,1)
    Image.fromarray((a*255).round().astype(np.uint8)).resize((800,1000),Image.LANCZOS).save(out,quality=93,subsampling=0)
    print('  ', os.path.basename(out))

print('doctors:')
doctor('youyang','尤洋_2',1380,420,2100,1260, f'{REPO}/images/team/youyang.jpg')
doctor('zhuqianru','朱茜如',628,285,1105,670, f'{REPO}/images/team/zhuqianru.jpg')
print('likeness:')
drawing('/Users/saraguo/Downloads/李师姐_素描版.jpg', f'{REPO}/images/practitioner/likeness-drawn.jpg', '#E1CEA2','#F4F3EF',0.50,(1000,1250))
print('sara:')
sara('/Users/saraguo/Downloads/郭希昱头像.jpg', f'{REPO}/images/sara.jpg')
