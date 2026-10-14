ffmpeg -i aerial_original_reconstructed.mp4 -i aerial_mae_reconstructed.mp4 -i aerial_dinov3_reconstructed.mp4 \
 -i aerial_proteus_reconstructed.mp4 -i aerial_radio_reconstructed.mp4 -i aerial_actok_reconstructed.mp4 \
-filter_complex \
"[0:v]pad=iw+2:ih+2:0:0:white[p0]; \
[1:v]pad=iw+2:ih+2:0:0:white[p1]; \
[2:v]pad=iw+2:ih+2:0:0:white[p2]; \
[3:v]pad=iw+2:ih+2:0:0:white[p3]; \
[4:v]pad=iw+2:ih+2:0:0:white[p4]; \
[5:v]pad=iw+2:ih+2:0:0:white[p5]; \
[p0][p1][p2]hstack=inputs=3[top]; \
[p3][p4][p5]hstack=inputs=3[bottom]; \
[top][bottom]vstack=inputs=2[grid]; \
[grid]drawtext=text='Input':x=5:y=5:fontsize=12:fontcolor=white:box=1:boxcolor=gray@0.5:boxborderw=2, \
drawtext=text='MAE':x=w/3+5:y=5:fontsize=12:fontcolor=white:box=1:boxcolor=gray@0.5:boxborderw=2, \
drawtext=text='DINOv3':x=2*w/3+5:y=5:fontsize=12:fontcolor=white:box=1:boxcolor=gray@0.5:boxborderw=2, \
drawtext=text='Proteus':x=5:y=h/2+5:fontsize=12:fontcolor=white:box=1:boxcolor=gray@0.5:boxborderw=2, \
drawtext=text='Radiov2.5':x=w/3+5:y=h/2+5:fontsize=12:fontcolor=white:box=1:boxcolor=gray@0.5:boxborderw=2, \
drawtext=text='ACTok (Ours)':x=2*w/3+5:y=h/2+5:fontsize=12:fontcolor=yellow:box=1:boxcolor=gray@0.5:boxborderw=2[v]" \
-map "[v]" -map 0:a? \
-c:v libx264 -crf 18 -preset slow -c:a aac \
aerial_comp.mp4