####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2);



### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset/dataset_oil.csv", header = TRUE, stringsAsFactors = FALSE)


oil_trend= ts(data[11])
wti_trend= ts(data[13])
brent_trend= ts(data[14])


barrel_trend= ts(data[16])
date=(data[2])
oil_price= ts(data[6])

oil_trend_gen=oil_trend*0.65 + barrel_trend * 0.25 + wti_trend*0.05  +  brent_trend*0.05

data1 = cbind(oil_trend_gen,oil_price)
# data1 = cbind(wti_trend,oil_price)
# data1 = cbind(brent_trend,oil_price)
# data1 = cbind(barrel_trend,oil_price)



garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1),model = "sGARCH"), distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvnorm")
dcc.fit =dccfit(dcc.garch11.spec,  data=data1)
#dcc.fit2 = dccfit(dcc.garch11.spec,  data=data1_ret)
#dcc.fit3 = dccfit(dcc.garch11.spec,  data=data2)

plot(dcc.fit)                
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


#dyncorr2 = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset/dynamic_corr2.csv")




####################################
write.csv(gino, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset/dynamic_corr.csv")
write.csv(dyncorr2, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset/dynamic_corr2_gen.csv")













###################random test#############
random=runif(833,min=0,max=100)
data_rand=cbind(random,oil_price)
garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1),model = "sGARCH"), distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvnorm")
dccrand.fit =dccfit(dcc.garch11.spec,  data=data_rand)

plot(dccrand.fit)
