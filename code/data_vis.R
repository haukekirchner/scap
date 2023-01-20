library(ggplot2)
library(dplyr)

#####################
export_ggplot <- function(g, target_size = "full", name, ratio = 1, width = NA){
  
  if(is.na(width)){
    if(target_size == "full"){
      width = 15
    } else if(target_size == "half"){
      #width = 10
      width = 7
    } else if(target_size == "third"){
      width = 5
    } else {
      print("Illegal target_size.")
      return(FALSE)
    }
  }
  
  
  
  height = width * ratio
  
  
  g <- g +
    theme(text = element_text(size=10)) +
    #theme(legend.position="bottom", legend.direction="horizontal")+
    theme(panel.background = element_blank())+ theme(legend.margin=ggplot2::margin(t = 0, unit='cm'))+
    theme(legend.key=element_blank())
  
  ggsave(name,
         width = width, height = height, dpi = 300, units = "cm")
}
#####################

data <- read.csv2(file = '~/Downloads/scap_results/202301_1150_scap_results.csv')
View(data)

data <- na.omit(data)

ggplot(data, aes(factor(tool), elapsed_time, fill = node)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")


g <-ggplot(data, aes(factor(node), elapsed_time, fill = tool)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")

g

export_ggplot(g, target_size = "full", name = "~/projects/scap/data/sacct_barplot_by_nodes.png", ratio = 1)
