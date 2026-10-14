ffmpeg -i birds_ori.mp4 -i birds_proteus.mp4 -i birds_dinov3.mp4 -i birds_distill.mp4 \
-filter_complex \
"[0:v]pad=iw+5:ih+5:0:0:white[pad0]; \
 [1:v]pad=iw+5:ih+5:0:0:white[pad1]; \
 [2:v]pad=iw+5:ih+5:0:0:white[pad2]; \
 [3:v]pad=iw+5:ih+5:0:0:white[pad3]; \
 [pad0][pad1]hstack=inputs=2[top]; \
 [pad2][pad3]hstack=inputs=2[bottom]; \
 [top][bottom]vstack=inputs=2[grid]; \
 [grid]drawtext=text='Input video':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=blue@0.5, \
 drawtext=text='Proteus (ViT-B)':x=w/2+15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=blue@0.5, \
 drawtext=text='DINO-v3 (ViT-B)':x=15:y=h/2+15:fontsize=24:fontcolor=white:box=1:boxcolor=blue@0.5, \
 drawtext=text='ACTok (ViT-B) (ours)':x=w/2+15:y=h/2+15:fontsize=24:fontcolor=yellow:box=1:boxcolor=blue@0.5[v]" \
-map "[v]" -map 0:a? \
-c:v libx264 -crf 18 -preset slow -c:a aac \
birds_comp.mp4