####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/dataset_dollar_eur.csv", header = TRUE, stringsAsFactors = FALSE)


crisis_trend= ts(data[8])
job_trend= ts(data[9])
loan_trend= ts(data[10])
unemploy_trend= ts(data[11])
default_trend= ts(data[12])
suicide_trend= ts(data[13])
killing_trend= ts(data[14])
crime_trend= ts(data[15])
war_trend= ts(data[18])
uprising_trend=ts(data[19])
hiring_trend=ts(data[20])
work_trend=ts(data[21])
usajob_trend=ts(data[22])
apply_trend=ts(data[23])




gen_dollar_eur_trend=crisis_trend + job_trend + loan_trend + unemploy_trend + default_trend + suicide_trend + killing_trend + crime_trend + war_trend
weighted_dollar_eur_trend = 0.3*crisis_trend + 0.3*job_trend + 0.05*loan_trend + 0.1*default_trend + 0.3*usajob_trend
weighted_data= cbind( weighted_dollar_eur_trend , date)

date=ts(data[2])
dollar_eur= ts(data[6])



data1 = cbind(gen_dollar_eur_trend,dollar_eur)
data2 = cbind(weighted_dollar_eur_trend,dollar_eur)



garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1),model = "sGARCH"), distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvnorm")
dcc.fit =dccfit(dcc.garch11.spec,  data=data2)
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


#dyncorr2 = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/dynamic_corr2.csv")


dyncorr_weighted = integer(nrow(data2)-252)

for (i in (252 : nrow(data2))){
  
  dcc.fit =dccfit(dcc.garch11.spec,  data=data2[1:i,1:2])
  dmat = rcor(dcc.fit)
  dyncorr_weighted[i-252] = dmat[1,2,i]
  
  
  
}




####################################
write.csv(dyncorr_weighted, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/dynamic_corr_wei_dollar_eur.csv")
write.csv(weighted_dollar_eur_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/wei_dollar_eur.csv")

write.csv(dyncorr2, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/dynamic_corr_dollar_eur.csv")
write.csv(gen_dollar_eur_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 5/dataset/gen_dollar_eur_trend.csv")
