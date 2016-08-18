library(ggplot2)
library(scales)
library(Hmisc)

excluding.method <- 'FIC+BP'
bar.width <- 0.5

df <- read.table("summary")
colnames(df) <- c('data', 'N', 'trueK', 'seed', 'method', 'estK', 'runtime', 'TLL', 'PLL')
df <- subset(df, method != 'EM')
df$N <- factor(df$N)
levels(df$method)[levels(df$method) == 'FVAB'] <- 'F2AB'
levels(df$method)[levels(df$method) == 'VAB'] <- 'FIC+BP'
levels(df$method)[levels(df$method) == 'FABVB'] <- 'FAB'
levels(df$method)[levels(df$method) == 'ICL'] <- 'BICEM'
levels(df$method)[levels(df$method) == 'ICLO'] <- 'ICL'

df <- subset(df, data=='balanced' & method != excluding.method)

method.order <- rev(c(4,3,6,1,5,2))
df$method <- factor(df$method, levels = levels(df$method)[method.order])

dodge <- position_dodge(width=0.8)
gg1 <- ggplot(df, aes(x=N, y=estK, group=method, fill=method)) + 
  stat_summary(fun.y = mean, geom = "bar", position=dodge, width=bar.width) + 
  stat_summary(fun.data = mean_sdl, geom = "errorbar", position=dodge, width=0.3) +
  geom_hline(yintercept=df$trueK[1], linetype=3) +
  ylab("Selected K") 

#gg1 <- ggplot(df, aes(x=N, y=iscorrect, group=method, fill=method)) + 
#  stat_summary(fun.y = mean, geom = "bar", position=dodge) + 
#  stat_summary(fun.data = mean_sdl, mult=1, geom = "errorbar", position=dodge, width=0.3) +
#  geom_hline(yintercept=df$trueK[1], linetype=3) +
#  ylab("Selected K") + 
#  facet_wrap(~data, ncol=2)

gg2 <- ggplot(df, aes(x=N, y=runtime, group=method, fill=method)) +
  stat_summary(fun.y = mean, geom = "bar", position=dodge, width=bar.width) + 
#  stat_summary(fun.data = mean_sdl, mult=1, geom = "errorbar", position=dodge, width=0.3) +
#  scale_y_log10() +
  scale_y_continuous(trans=log_trans(), breaks=c(1,10,100,1000,10000)) + 
  ylab("Runtime (second)")



gg.out <- gg1 + theme_bw() + coord_flip() + theme(legend.position="none")
#print(gg.out)
ggsave('ex-toy_K.eps', gg.out , height=3, width=6)

gg.out <- gg2 + theme_bw() + coord_flip() + theme(legend.title=element_blank()) + guides(fill=guide_legend(reverse = TRUE))
 
#print(gg.out)
ggsave('ex-toy_time.eps', gg.out , height=3, width=6)


#gg.out <- gg2 + theme_bw() + theme(legend.position=c(0.35,1), legend.title=element_blank(), legend.direction='horizontal')
