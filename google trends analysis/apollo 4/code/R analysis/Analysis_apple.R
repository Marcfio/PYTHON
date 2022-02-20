####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/dataset_apple.csv", header = TRUE, stringsAsFactors = FALSE)


apple_trend= ts(data[12])
iphone_trend= ts(data[13])
apple_watch_trend= ts(data[14])
airpods_trend= ts(data[15])


gen_apple_trend=apple_trend + iphone_trend + apple_watch_trend + airpods_trend
weighted_apple_trend = 0.65*apple_trend + 0.20*iphone_trend + 0.075*apple_watch_trend +0.075*airpods_trend
weighted_apple_trend = cbind( weighted_apple_trend , date)

date=(data[2])
apple_price= ts(data[6])



data1 = cbind(gen_apple_trend,apple_price)
data2 = cbind(weighted_apple_trend,apple_price)



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


#dyncorr2 = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/dynamic_corr2.csv")


dyncorr_weighted = integer(nrow(data2)-252)

for (i in (252 : nrow(data2))){
  
  dcc.fit =dccfit(dcc.garch11.spec,  data=data2[1:i,1:2])
  dmat = rcor(dcc.fit)
  dyncorr_weighted[i-252] = dmat[1,2,i]
  
  
  
}




####################################
write.csv(dyncorr_weighted, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/dynamic_corr_wei_apple.csv")
write.csv(weighted_apple_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/wei_apple.csv")

write.csv(dyncorr2, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/dynamic_corr_apple.csv")
write.csv(gen_apple_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 4/dataset/gen_apple_trend.csv")
