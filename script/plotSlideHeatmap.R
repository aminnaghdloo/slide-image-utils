#!/usr/bin/Rscript
library("ggplot2")
library("ggfittext")
library("optparse", quietly=TRUE)

args = commandArgs(trailingOnly=TRUE)

addFrameCoords = function(df){
  
  # Create frame grid
  xy = data.frame(frame_id=1:2304)
  xy['frame_x'] = ceiling(xy$frame_id / 24)
  xy['frame_y'] = 1:24

  for(i in seq(2, 96, 2)){
    xy$frame_y[xy$frame_x == i] = 24:1
  }

  xy$frame_x = as.factor(xy$frame_x)
  xy$frame_y = as.factor(xy$frame_y)
  
  # Merge the input data with the frame grid using frame_id
  df_plt = merge(xy, df, 'frame_id', all.x=TRUE)
  return(df_plt)
}

main = function(){
  parser = OptionParser("Plot slide heatmap of a function")
  parser = add_option(parser, c('-i', '--input'),
                      help="Input tab-delimited file with frame_id [Required]")
  parser = add_option(parser, c('-v', '--variable'),
                      help="Target variable to visualize [Required]")
  parser = add_option(parser, c('-o', '--output'),
                      help="Output image name", default='out.png')
  parser = add_option(parser, c('-s', '--slide'),
                      help="slide_id if the file has multiple slides",
                      default=NA)
  parser = add_option(parser, c('-t', '--title'),
                      help="title of the figure",
                      default="")
  parser = add_option(parser, c('-l', '--label'), action="store_true",
                      help="show numeric values on tiles",
                      default=FALSE)

  opt = parse_args(parser)
  if(is.null(opt$input) | is.null(opt$variable)){
    print_help(parser)
    quit(status=-1)
  }

  df = read.table(opt$input, header=TRUE, sep='\t')
  column_id = which(names(df) == opt$variable)

  # Filter selected slide
  if(!is.na(opt$slide)){
    df = df[df$slide_id == opt$slide,]
  }
  
  # Check the existence of target variable
  if(length(column_id) == 0){
    print('Variable does not exist in column names of input file.')
    quit(status=-1)
  }else{
    df = addFrameCoords(df)
    if(opt$label){
    p1 = ggplot(df, aes_string(x='frame_x', y='frame_y', fill=opt$variable,
                               label=opt$variable)) + 
        geom_tile(color='black') +
        geom_text(size=1) +
        xlab('frame x') + ylab('frame y') + ggtitle(opt$title) +
        scale_fill_gradientn(colors=hcl.colors(20, 'Temps'), na.value='white') +
        scale_x_discrete(breaks=c(1, seq(8, 96, 8))) +
        scale_y_discrete(breaks=c(1, seq(8, 24, 8))) + theme_bw() +
        guides(fill=guide_colourbar(barwidth=1, barheight=12))# +
        theme(text=element_text(size=18))
    }else{
    p1 = ggplot(df, aes_string(x='frame_x', y='frame_y', fill=opt$variable)) + 
        geom_tile(color='black') + xlab('frame x') + ylab('frame y') +
        ggtitle(opt$title) +
        scale_fill_gradientn(colors=hcl.colors(20, 'Temps'), na.value='white') +
        scale_x_discrete(breaks=c(1, seq(8, 96, 8))) +
        scale_y_discrete(breaks=c(1, seq(8, 24, 8))) + theme_bw() +
        guides(fill=guide_colourbar(barwidth=1, barheight=12)) +
        theme(text=element_text(size=18))
    }
    ggsave(plot=p1, filename=opt$output, height=3.6, width=12)
  }
}

main()
