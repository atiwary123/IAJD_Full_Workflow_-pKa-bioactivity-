#!/usr/bin/env python3
"""OCR the image-only SI PDFs -> text files for formula/pKa corroboration."""
import fitz, pytesseract, io, sys
from PIL import Image
PAIRS=[('ja1c09585','ja1c09585_si_001.pdf'),('ja3c07337','ja3c07337_si_003.pdf'),
('ja3c13569','ja3c13569_si_001.pdf'),('ja5c07232','ja5c07232_si_001.pdf'),
('bm4c01599','bm4c01599_si_001.pdf'),('bm4c01107','bm4c01107_si_001.pdf')]
for key,pdf in PAIRS:
    d=fitz.open('IAJD_master/source_papers/'+pdf)
    out=[]
    for i,p in enumerate(d):
        pix=p.get_pixmap(dpi=300)
        img=Image.open(io.BytesIO(pix.tobytes('png'))).convert('L')
        try: out.append(pytesseract.image_to_string(img))
        except Exception as e: out.append('')
        if (i+1)%10==0: print(f'{key}: {i+1}/{d.page_count}',flush=True)
    open(f'audit_work/paper_text/{key}_ocr.txt','w').write('\n'.join(out))
    print(f'DONE {key}: {d.page_count}pp -> {sum(len(x) for x in out)} chars',flush=True)
print('ALL OCR DONE')
