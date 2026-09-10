import numpy as np, sys
from PIL import Image, ImageFilter
sys.path.insert(0, '/private/tmp/claude-501/-Users-saraguo-content/8f5c22e9-9b35-4054-95f0-422e53eb155d/scratchpad/team')
from seg import person_mask

SRC = "/Users/saraguo/content/TCM_Content/website/抱朴堂素材_已拆分/医生照片"
AR, BG = 0.8, np.array([0xE9,0xE6,0xDF], float)/255
HEAD_FRAC, EYE_FRAC = 0.542, 0.34      # the largest framing both sources hold with no invented pixels

def render(path, cx, head_top, chin, eye, out):
    im = Image.open(path).convert('RGB')
    mk = Image.fromarray(person_mask(path)[0]).resize(im.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.4))
    iw, ih = im.size
    H = (chin-head_top)/HEAD_FRAC; W = H*AR
    l, u = int(cx-W/2), int(eye-EYE_FRAC*H)
    assert l >= 0 and u >= 0 and l+W <= iw and u+H <= ih, f'{out}: crop escapes the source'
    box = (l, u, l+int(W), u+int(H))
    im, mk = im.crop(box), mk.crop(box)

    OW, OH = 800, 1000
    im = im.resize((OW*2,OH*2), Image.LANCZOS); mk = mk.resize((OW*2,OH*2), Image.LANCZOS)
    m = np.clip((np.asarray(mk).astype(float)[...,None]/255 - .30)/.70, 0, 1)
    a = np.asarray(im).astype(float)/255

    subj = m[...,0] > .7
    lum  = a.mean(2)
    coat = subj & (lum >= np.percentile(lum[subj], 95))
    cm   = a[coat].mean(0)
    a = np.clip(a*(cm.mean()/cm)[None,None,:], 0, 1)          # neutralise on the lab coat
    a = np.clip(a*(0.92/a[coat].mean()), 0, 1)                # same exposure for both
    g = (a*np.array([.2126,.7152,.0722])).sum(2, keepdims=True)
    a = np.clip(g+(a-g)*0.74, 0, 1)
    a = a*m + BG[None,None,:]*(1-m)

    q = np.clip(a*255, 0, 255).round().astype(np.uint8)
    Image.fromarray(q).resize((OW,OH), Image.LANCZOS).save(out, quality=93, subsampling=0)
    print('   %-16s crop %s  mean %s' % (out, box, q.reshape(-1,3).mean(0).round(1)))

render('youyang_clean.jpg', 1380, 420, 2100, 1260, 'G_youyang.jpg')
render(f'{SRC}/朱茜如.jpeg',  628, 285, 1105,  670, 'G_zhuqianru.jpg')
