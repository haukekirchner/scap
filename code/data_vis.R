library(ggplot2)
library(dplyr)
library(lubridate)

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

data_jobs <- read.csv2(file = '~/projects/scap/data/202301_1150_scap_results.csv')
data_jobs

data_sacct <- read.csv(file = '~/projects/scap/data/sacct.out', sep="|")
names(data_sacct)[names(data_sacct) == 'JobID'] <- 'job_id'
data_sacct

data <- merge(data_jobs,data_sacct,by="job_id")
data

data$time <- parse_date_time(data$Elapsed,
                c('%H:%M:%S'), exact = TRUE)

boundary <- ymd_hms("0000-01-01 00:00:00")
span <- interval(boundary, data$time) # interval
data$duration <- as.duration(span)


ggplot(data, aes(factor(tool), duration, fill = node)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")



# inital problem
g <-ggplot(filter(data, experiment =="sample-points"), aes(factor(node), duration)) + 
  geom_bar(stat="identity", position = "dodge")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")

g


g <-ggplot(filter(data, experiment ==""), aes(factor(node), duration, fill = tool, alpha = is_valid)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_alpha_discrete("Is a valid run.")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")
g
export_ggplot(g, target_size = "full", name = "~/projects/scap/data/sacct_barplot_by_nodes_no-experiment.png", ratio = 1)


g <-ggplot(filter(data, tool=="profiler-torch" & experiment != "batch-size-64" & (node =="scc_gtx1080" | node == "scc_cpu")), aes(factor(node), duration)) + 
  geom_bar(aes(fill = experiment),stat="identity", position = "dodge") + 
  scale_alpha_discrete("Is a valid run.")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")
g
export_ggplot(g, target_size = "half", name = "~/projects/scap/data/sacct_barplot_by_nodes_sample-points-effect.png", ratio = 1)


g <-ggplot(data, aes(factor(node), log(duration), fill = tool, alpha = is_valid)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_alpha_discrete("Is a valid run.")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [log(sec)]")

g

g <-ggplot(filter(data,experiment==""), aes(factor(node), duration, fill = tool, alpha = is_valid)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_alpha_discrete("Is a valid run.")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")

g

export_ggplot(g, target_size = "full", name = "~/projects/scap/data/sacct_barplot_by_nodes_no-experiment.png", ratio = 0.5)


g <-ggplot(filter(data, node != "scc_cpu" & experiment == ""), aes(factor(node), duration, fill = tool)) + 
  geom_bar(stat="identity", position = "dodge")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")

g

export_ggplot(g, target_size = "full", name = "~/projects/scap/data/sacct_barplot_by_nodes_no-experiment_gpu.png", ratio = 0.5)

g <-ggplot(filter(data, tool=="profiler-torch" & experiment != "sample-points" & node =="scc_gtx1080"), aes(factor(node), duration)) + 
  geom_bar(aes(fill = experiment),stat="identity", position = "dodge") + 
  scale_alpha_discrete("Is a valid run.")+
  scale_fill_brewer(palette = "Set1")+
  xlab("Compute Nodes")+
  ylab("Elapsed time [sec]")
g
export_ggplot(g, target_size = "half", name = "~/projects/scap/data/sacct_barplot_by_nodes_batch-size-effect.png", ratio = 1)
