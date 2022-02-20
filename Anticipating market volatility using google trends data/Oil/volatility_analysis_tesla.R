####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2); library(xlsx)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset2_tesla.csv", header = TRUE, stringsAsFactors = FALSE)
tesla = ts(data[6])    
trend_data = ts(data[15])
dates = data[2:nrow(data), 2  ]
tesla = 100*diff(log(tesla))
for(i in ( 2 :length(tesla))){
  if ((tesla[i]>50) || (tesla[i]<(-50))){
    tesla[i]= tesla[i-1]}
  
}
  



trend_data = 100*diff(log(trend_data))
dataf = data.frame(dates,tesla,trend_data)

####plot and summary####################

plot(dataf$tsla_price,type = 'l' )
plot(dataf$tesla_gen_trend, type = 'l' )
summary(trend_data)
summary(tesla)
plot(dataf$tsla_price,dataf$tsla_price )

tesla_2 = ts (tesla)^2 ##############squared returns
trend_data_2 = ts (trend_data)^2 ##############squared returns
#############################Leverage, autocorrelation and conditional correlation
plot(tesla[1:(length(tesla[,1])-1), 1], tesla[2:length(tesla[,1]), 1]^2, ylab = 'tesla_sq.ret', xlab = 'tesla_lag')
lines(ksmooth(tesla[1:(length(tesla[,1])-1), 1], tesla[2:length(tesla[,1]), 1]^2, 'normal', bandwidth = 6), col = 'red')

plot(trend_data[1:(length(tesla[,1])-1), 1], tesla[2:length(tesla[,1]), 1]^2, ylab = 'tesla_sq.ret', xlab = 'tesla_lag')
lines(ksmooth(trend_data[1:(length(tesla[,1])-1), 1], tesla[2:length(tesla[,1]), 1]^2, 'normal', bandwidth = 6), col = 'red')






acf_tesla = acf(tesla)
summary(acf_tesla)

lag_corr = ccf(dataf$tsla_price , dataf$tesla_gen_trend)
lag_corr_2 = ccf(tesla_2[ , 1],trend_data_2[ , 1])

##########################regression#################################
reg_tesla_trend = lm(formula = dataf$tsla_price  ~ dataf$tesla_gen_trend, x=TRUE, y=TRUE)
reg_tesla_trend_2 = lm(formula = (dataf$tsla_price )^2 ~ (dataf$tesla_gen_trend)^2 , x=TRUE, y=TRUE)
 
summary(reg_tesla_trend)
summary(reg_tesla_trend_2)



plot(reg_tesla_trend)
plot(reg_tesla_trend_2)



###############################################################test
###########Jarque Bera--> Gaussian error
###########LM test-------> omoskedasticity
###########Akaike--------> n. of parameters



# ###akaike information for Arma orders 
# info_garch = matrix(0,4,4)
# for (i in 1:4){
#   for(j in 1: 4){
#   t_garch.spec.trend = ugarchspec(mean.model = list(armaOrder = c(i,j)), 
#                                     variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
#                                     distribution.model = "std" )    
#   t_garch.fit.trend= ugarchfit(spec=t_garch.spec.trend, data=data$tesla, solver.control=list(trace = 1))
# 
#   info_garch[i,j] = infocriteria(t_garch.fit.trend)[1]
#   }
# 
# }
# ############################arma(4,3) is the best
######################################t- GARCH (1,1)
t_garch.spec.trend = ugarchspec(mean.model = list(armaOrder = c(1,1)), 
                                  variance.model = list(garchOrder = c(2,2), model = "sGARCH"),
                                  distribution.model = "std" )
# Model Fitting
t_garch.fit.trend = ugarchfit(spec=t_garch.spec.trend, data=dataf$tsla_price, solver.control=list(trace = 1))
t_garch.fit.trend@fit$hessian
windows(); par(mfcol = c(2,2))

# Conditional Volatility
plot.ts(sigma(t_garch.fit.trend),
        main="trend - Conditional Volatility t-GARCH(2,2)",
        xlab = "Number of Observations", ylab="sqrt(h_t)", col="green4")
# Standardized Residuals
std.res_tgarch_trend = ts(as.numeric(residuals(t_garch.fit.trend, standardize=TRUE))) 
plot(std.res_tgarch_trend, main='Standardized Residuals', xlab='Number of Observations', ylab='Standardized Residuals')
acf((std.res_tgarch_trend)^2, main='ACF - Standardized Squared Residuals')
plot(t_garch.fit.trend, which=9)

summary(std.res_tgarch_trend); skewness(std.res_tgarch_trend); kurtosis(std.res_tgarch_trend)
summary(sigma(t_garch.fit.trend))
####################VAR ###########################
#alpha=(0.8)
# t_ht_port = sigma (t_garch.fit.trend)
# #v_std = t_garch.fit.trend@fit[["coef"]][["shape"]]#storing the degrees of freedom
# t_VaR = -(t_ht_port) * qt(alpha,v_std)/sqrt(v_std/(v_std-2))
# 
# 
# plot(t_VaR)



########################################VAR: no external regressor ############


t_garch.spec.trend = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                                variance.model = list(garchOrder = c(2,2), model = "sGARCH"),
                                distribution.model = "std" )



t_garch.fit.trend = ugarchfit(spec=t_garch.spec.trend, data=tesla, solver.control=list(trace = 0))

stat(t_garch.fit.trend)


t_garch.roll.trend= ugarchroll(t_garch.spec.trend,data=tesla, n.head = 1, refit.every = 20)

roll_var = report(t_garch.roll.trend, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)
########################################VAR: external regressor ############


t_garch.spec.exog = ugarchspec(mean.model = list(armaOrder = c(0,0), external.regressors = NULL), 
                                  variance.model = list(garchOrder = c(0,1), model = "sGARCH",  external.regressors =trend_data_2 ),
                                  distribution.model = "std")



t_garch.fit.exog = ugarchfit(spec=t_garch.spec.exog, data=tesla, solver.control=list(trace = 0))

stat(t_garch.fit.exog)


t_garch.roll.exog= ugarchroll(t_garch.spec.exog,data=tesla, n.head = 1, refit.every = 20)

roll_var_exg = report(t_garch.roll.exog, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)

########################################VAR: external regressor 2 ############


t_garch.spec.exog = ugarchspec(mean.model = list(armaOrder = c(0,0), external.regressors = NULL), 
                               variance.model = list(garchOrder = c(0,1), model = "sGARCH",  external.regressors =trend_data_2 ),
                               distribution.model = "std")



t_garch.fit.exog = ugarchfit(spec=t_garch.spec.exog, data=tesla, solver.control=list(trace = 0))

stat(t_garch.fit.exog)


t_garch.roll.exog= ugarchroll(t_garch.spec.exog,data=tesla, n.head = 1, refit.every = 20)

roll_var_exg = report(t_garch.roll.exog, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)

#####plot vari
plot(tesla_2,col= "red")
lines(trend_data_2)
cor(tesla_2,trend_data_2)



#########################################graph export ########################à
write.xlsx(dataf, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/graph.xlsx")
write.xlsx(data, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/graph2.xlsx")
sigma_trend = t_garch.fit.trend@fit$sigma
sigma_exog = t_garch.fit.exog@fit$sigma
Var_exog = t_garch.roll.exog@forecast[["VaR"]][["alpha(5%)"]]
Var_trend=t_garch.roll.trend@forecast[["VaR"]][["alpha(5%)"]]

sigma=cbind(sigma_exog,sigma_trend)
VaR= cbind(Var_exog, Var_trend)
residuals_exog = (t_garch.fit.exog@fit[["residuals"]])
residuals_trend = (t_garch.fit.trend@fit[["residuals"]])
redisuals = cbind(residuals_exog, residuals_trend)
residuals_2 = residuals

write.xlsx(sigma, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/sigma2.xlsx")
write.xlsx(VaR, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/VaR2.xlsx")
write.xlsx(residuals_exog, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/residuals.xlsx")
write.xlsx(residuals_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 3/dataset/dataset_graph/residuals_2.xlsx")





