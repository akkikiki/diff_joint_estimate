library('ggplot2')
library('GGally')
library(reshape2)
library(RColorBrewer)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

df <- read.csv('data/processed/difficulties_doc.tsv', sep='\t')
df_melted <- melt(df)
names(df_melted) <- c('doc.diff', 'word.diff.max', 'count')

p1 <- ggplot(df_melted, aes(doc.diff, word.diff.max)) +
  geom_tile(aes(fill = count), colour = "white") +
  scale_fill_distiller(palette = "Greys", direction = 1) + 
  theme_bw()

df <- read.csv('data/processed/difficulties_word.tsv', sep='\t')
df_melted <- melt(df)
names(df_melted) <- c('word.diff', 'doc.diff.min', 'count')

p2 <- ggplot(df_melted, aes(word.diff, doc.diff.min)) +
  geom_tile(aes(fill = count), colour = "white") +
  scale_fill_distiller(palette = "Greys", direction = 1) + 
  theme_bw()

multiplot(p1, p2, cols=2)
# output as a 8 in x 3.2 in PDF
