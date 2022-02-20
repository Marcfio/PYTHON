####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 1/dataset/dataset.csv", header = TRUE, stringsAsFactors = FALSE)
return = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 1/dataset/dataset_ret.csv", header = TRUE, stringsAsFactors = FALSE)

Trump= ts(data[14])
#Trump_ret = ts(return[10])
SeP500= ts(data[6])
#SeP500_ret= ts(return[5])

data1 = cbind(Trump,SeP500)
#data1_ret = cbind(Trump_ret,SeP500_ret)
#data2 = cbind(Trump,SeP500_ret)




garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1),model = "sGARCH"), distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvnorm")
dcc.fit =dccfit(dcc.garch11.spec,  data=data1)
#dcc.fit2 = dccfit(dcc.garch11.spec,  data=data1_ret)
#dcc.fit3 = dccfit(dcc.garch11.spec,  data=data2)

#plot(dcc.fit)                
#plot(dcc.fit2)                 
#plot(dcc.fit3)

#pluto = rcor(dcc.fit)
#gino = pluto[1,2,1]


dyncorr2 = integer(nrow(data1)-252)

for (i in (252 : nrow(data1))){
  
  dcc.fit =dccfit(dcc.garch11.spec,  data=data1[1:i,1:2])
  dmat = rcor(dcc.fit)
  dyncorr2[i-252] = dmat[1,2,i]
  
  
  
}

#plot(Trump)
#plot(SeP500)


write.csv(gino, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 1/dataset/dynamic_corr.csv")
write.csv(dyncorr2, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 1/dataset/dynamic_corr2.csv")
