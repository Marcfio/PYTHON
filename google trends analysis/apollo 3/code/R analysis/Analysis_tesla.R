####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_tesla.csv", header = TRUE, stringsAsFactors = FALSE)


tesla_trend= ts(data[11])
musk_trend= ts(data[12])
elon_trend= ts(data[13])
gen_tesla_trend=tesla_trend+musk_trend+elon_trend




date=(data[2])
tesla_price= ts(data[6])



data1 = cbind(gen_tesla_trend,tesla_price)
data2 = cbind(musk_trend,tesla_price)



garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1),model = "sGARCH"), distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvnorm")
dcc.fit =dccfit(dcc.garch11.spec,  data=data1)
#dcc.fit2 = dccfit(dcc.garch11.spec,  data=data1_ret)
#dcc.fit3 = dccfit(dcc.garch11.spec,  data=data2)

#plot(dcc.fit)                
#plot(dcc.fit2)                 
#plot(dcc.fit3)

pluto = rcor(dcc.fit)
gino = pluto[1,2,1]


dyncorr2 = integer(nrow(data1)-252)

for (i in (252 : nrow(data1))){
  
  dcc.fit =dccfit(dcc.garch11.spec,  data=data1[1:i,1:2])
  dmat = rcor(dcc.fit)
  dyncorr2[i-252] = dmat[1,2,i]
  
  
  
}


#dyncorr2 = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dynamic_corr2.csv")


dyncorr_musk = integer(nrow(data2)-252)

for (i in (252 : nrow(data2))){
  
  dcc.fit =dccfit(dcc.garch11.spec,  data=data2[1:i,1:2])
  dmat = rcor(dcc.fit)
  dyncorr_musk[i-252] = dmat[1,2,i]
  
  
  
}




####################################
write.csv(gino, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dynamic_corr.csv")
write.csv(dyncorr2, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dynamic_corr2.csv")
