import numpy as np, sys
from PIL import Image, ImageFilter
sys.path.insert(0, '/private/tmp/claude-501/-Users-saraguo-content/8f5c22e9-9b35-4054-95f0-422e53eb155d/scratchpad/team')
from seg import person_mask

SRC = "/Users/saraguo/content/TCM_Content/website/抱朴堂素材_已拆分/医生照片"
AR, BG = 0.8, np.array([0xE9,0xE6,0xDF], float)/255
HEAD_FRAC, EYE_FRAC = 0.542, 0.34
STRENGTH = 0.70          # how far her skin moves toward his; 1.0 would erase her own colouring

def skin_mask(a, m):
    """Skin, not coat and not hair: warm chroma, mid luminance, inside the subject."""
    R,G,B = a[...,0], a[...,1], a[...,2]
    lum = a.mean(2)
    s = (m[...,0]>.8) & (R>G) & (G>B) & ((R-B)>0.07) & (lum>0.40) & (lum<0.96)
    sm = np.asarray(Image.fromarray((s*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(float)/255
    return sm[...,None], s

def render(path, cx, head_top, chin, eye, skin_target=None):
    im = Image.open(path).convert('RGB')
    mk = Image.fromarray(person_mask(path)[0]).resize(im.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.4))
    iw, ih = im.size
    H = (chin-head_top)/HEAD_FRAC; W = H*AR
    l, u = int(cx-W/2), int(eye-EYE_FRAC*H)
    assert l>=0 and u>=0 and l+W<=iw and u+H<=ih, 'crop escapes the source'
    box=(l,u,l+int(W),u+int(H)); im, mk = im.crop(box), mk.crop(box)

    OW,OH = 800,1000
    im = im.resize((OW*2,OH*2), Image.LANCZOS); mk = mk.resize((OW*2,OH*2), Image.LANCZOS)
    m = np.clip((np.asarray(mk).astype(float)[...,None]/255 - .30)/.70, 0, 1)
    a = np.asarray(im).astype(float)/255

    subj = m[...,0]>.7; lum = a.mean(2)
    coat = subj & (lum >= np.percentile(lum[subj],95))
    cm = a[coat].mean(0)
    a = np.clip(a*(cm.mean()/cm)[None,None,:],0,1)
    a = np.clip(a*(0.92/a[coat].mean()),0,1)
    g = (a*np.array([.2126,.7152,.0722])).sum(2,keepdims=True)
    a = np.clip(g+(a-g)*0.74,0,1)

    sm, s = skin_mask(a, m)
    skin_now = a[s].mean(0)
    if skin_target is not None:
        target = skin_now + STRENGTH*(skin_target - skin_now)
        gain = target/np.maximum(skin_now, 1e-6)
        a = np.clip(a*(1 + (gain[None,None,:]-1)*sm), 0, 1)      # only where skin is
        print('   skin %s -> %s (target %s)' % ((skin_now*255).round(0), (a[s].mean(0)*255).round(0), (skin_target*255).round(0)))
    else:
        print('   skin %s  (reference)' % (skin_now*255).round(0))

    out = a*m + BG[None,None,:]*(1-m)
    q = np.clip(out*255,0,255).round().astype(np.uint8)
    return Image.fromarray(q).resize((OW,OH), Image.LANCZOS), skin_now

print('you yang (reference):')
img_y, skin_y = render('youyang_clean.jpg', 1380, 420, 2100, 1260)
img_y.save('H_youyang.jpg', quality=93, subsampling=0)
print('zhu qianru (matched to his skin):')
img_z, _ = render(f'{SRC}/朱茜如.jpeg', 628, 285, 1105, 670, skin_target=skin_y)
img_z.save('H_zhuqianru.jpg', quality=93, subsampling=0)
