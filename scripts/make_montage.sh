find . -maxdepth 2 -name "sample*.png" | sort -V > pnglist.txt

montage -density 900 -title "Diffusion Samples" -tile 5x0 -geometry +5+50 -border 10 @pnglist.txt all_samples.png

#&& ffmpeg -i sample.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" sample.mp4 && vlc sample.mp4

