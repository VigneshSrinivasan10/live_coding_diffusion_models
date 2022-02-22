for i in {0..9}
    do
	ffmpeg -framerate 12.5 -i ddim/img$i/%04d.png -pattern_type glob -r 30 -loop -1 ddim_img$i.gif
    done

for i in {0..9}
    do
	ffmpeg -framerate 250 -i ddpm/img$i/%04d.png -pattern_type glob -r 30 -loop -1 ddpm_img$i.gif
    done
