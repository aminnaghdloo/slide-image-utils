args = commandArgs(trailingOnly=TRUE)
if(length(args) != 4){
    print('Rscript append_frame_coords.R <events.txt> <frame_coords.txt> <output.txt> <slide_id>')
    quit('no')
}

df1 = read.table(args[1], header=TRUE, sep='\t')
df2 = read.table(args[2], header=FALSE, sep='\t')
names(df2) = c('frame_id', 'frame_x', 'frame_y')
if(!('slide_id' %in% names(df1))){
    df1$slide_id = args[4]
}
df_final = merge(df1, df2, 'frame_id', all.x=TRUE)
df_final = df_final[, c('slide_id', 'frame_id', 'cell_id', 'x', 'y', 'frame_x', 'frame_y')]
names(df_final) = c('slide_id', 'frame_id', 'cell_id', 'cell_x', 'cell_y', 'frame_x', 'frame_y')
write.table(df_final, args[3], row.names=FALSE, quote=FALSE, sep='\t')
