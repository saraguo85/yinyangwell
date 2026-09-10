import numpy as np, Quartz, Vision
from Foundation import NSURL
from PIL import Image

def person_mask(path):
    url = NSURL.fileURLWithPath_(path)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    cg  = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
    req.setQualityLevel_(Vision.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
    req.setOutputPixelFormat_(Quartz.kCVPixelFormatType_OneComponent8)
    h = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})
    ok, err = h.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(err)
    obs = req.results()[0]
    pb  = obs.pixelBuffer()
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    w = Quartz.CVPixelBufferGetWidth(pb); hgt = Quartz.CVPixelBufferGetHeight(pb)
    stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
    base = Quartz.CVPixelBufferGetBaseAddress(pb)
    buf = base.as_buffer(stride * hgt)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(hgt, stride)[:, :w].copy()
    Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    return arr, (Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg))

if __name__ == '__main__':
    S = "/Users/saraguo/content/TCM_Content/website/抱朴堂素材_已拆分/医生照片"
    for n in ['尤洋_2', '朱茜如']:
        m, size = person_mask(f'{S}/{n}.jpeg')
        print(n, 'img', size, 'mask', m.shape, 'coverage %.1f%%' % (100*(m>127).mean()))
        Image.fromarray(m).save(f'/private/tmp/claude-501/-Users-saraguo-content/8f5c22e9-9b35-4054-95f0-422e53eb155d/scratchpad/team/mask_{n}.png')
