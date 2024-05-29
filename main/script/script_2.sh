# PLLS for an entire slide
for i in {1..2304}; do img=$(printf 'Tile%06d.jpg' $i) && qval=$(python $program_path/../script/qc.py /mnt/T/OncoScope/tubeID_0AB66/exptID_15070/slideID_0AB6602/bzScanner/proc/${img}) && echo -e "$i\t$qval" >> 0AB6602_plls.txt; done

# Plot slide wide metric
./plotSlideHeatmap.R -i 0AB6602_plls.txt -t PLLS -o 0AB6602_plls.png
