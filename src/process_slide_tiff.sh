function slidepath () {
    local path=$(realpath /mnt/*/OncoScope/tubeID_${slide:0:5}/*/slideID_${slide}/bzScanner/proc)
    echo $path
}

slides=$@
for slide in $slides
do
python $program_path/detect_LEVs.py -i $(slidepath) -o ./${slide}.txt -L 99.7 -H 2 &&
python $program_path/extract_event_images.py -i $(slidepath) -d ./${slide}.txt -o ./${slide}.hdf5 -w 75 &&
python $program_path/create_montages.py -i ./${slide}.hdf5 -o ./${slide}.tif -w 75 -R TRITC FITC -G CY5 FITC -B DAPI FITC --sort TRITC_mean D &&
tiff2pdf -j -q 75 -o ${slide}.pdf ./${slide}.tif &&
rm -f ./${slide}.tif &&
echo $slide
done
