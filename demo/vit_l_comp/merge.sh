ffmpeg -ss 2 -i dogs_original_reconstructed.mp4 -ss 2 -i dogs_mae_reconstructed.mp4 -ss 2 -i dogs_dinov3_reconstructed.mp4 \
-ss 2 -i dogs_proteus_reconstructed.mp4 -ss 2 -i dogs_radio_reconstructed.mp4 -ss 2 -i dogs_actok_reconstructed.mp4 \
-filter_complex \
"[0:v]pad=iw+5:ih+5:0:0:white[p0]; \
[1:v]pad=iw+5:ih+5:0:0:white[p1]; \
[2:v]pad=iw+5:ih+5:0:0:white[p2]; \
[3:v]pad=iw+5:ih+5:0:0:white[p3]; \
[4:v]pad=iw+5:ih+5:0:0:white[p4]; \
[5:v]pad=iw+5:ih+5:0:0:white[p5]; \
[p0][p1][p2]hstack=inputs=3[top]; \
[p3][p4][p5]hstack=inputs=3[bottom]; \
[top][bottom]vstack=inputs=2[grid]; \
[grid]drawtext=text='Input video':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=gray@0.5, \
drawtext=text='MAE':x=w/3+15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=gray@0.5, \
drawtext=text='DINOv3':x=2*w/3+15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=gray@0.5, \
drawtext=text='Proteus':x=15:y=h/2+15:fontsize=24:fontcolor=white:box=1:boxcolor=gray@0.5, \
drawtext=text='Radiov2.5':x=w/3+15:y=h/2+15:fontsize=24:fontcolor=white:box=1:boxcolor=gray@0.5, \
drawtext=text='ACTok (Ours)':x=2*w/3+15:y=h/2+15:fontsize=24:fontcolor=yellow:box=1:boxcolor=gray@0.5[v]" \
-map "[v]" -map 0:a? \
-c:v libx264 -crf 18 -preset slow -c:a aac \
dogs_comp.mp4