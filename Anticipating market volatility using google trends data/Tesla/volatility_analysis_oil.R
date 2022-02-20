####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2); library(xlsx)
library(strucchange)

### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset/dataset2_oil.csv", header = TRUE, stringsAsFactors = FALSE)
oil = ts(data[7])
trend_data = ts(data[18])
dates = data[2:nrow(data), 2  ]
oil = 100*diff(log(oil))
for(i in ( 2 :length(oil))){
  if ((oil[i]>50) || (oil[i]<(-50))){
    oil[i]= oil[i-1]}
  
}
  



trend_data = 100*diff(log(trend_data))
dataf = data.frame(dates,oil,trend_data)

####plot and summary####################

plot(dataf$gen_trend,type = 'l' )
plot(dataf$oil,type = 'l' )
summary(trend_data)
summary(oil)
plot(dataf$gen_trend,dataf$oil)

oil_2 = ts (oil)^2 ##############squared returns
trend_data_2 = ts (trend_data)^2 ##############squared returns
#############################Leverage, autocorrelation and conditional correlation
plot(oil[1:(length(oil[,1])-1), 1], oil[2:length(oil[,1]), 1]^2, ylab = 'oil_sq.ret', xlab = 'oil_lag')
lines(ksmooth(oil[1:(length(oil[,1])-1), 1], oil[2:length(oil[,1]), 1]^2, 'normal', bandwidth = 6), col = 'red')

plot(trend_data[1:(length(oil[,1])-1), 1], oil[2:length(oil[,1]), 1]^2, ylab = 'oil_sq.ret', xlab = 'oil_lag')
lines(ksmooth(trend_data[1:(length(oil[,1])-1), 1], oil[2:length(oil[,1]), 1]^2, 'normal', bandwidth = 6), col = 'red')






acf_oil = acf(oil)
summary(acf_oil)

lag_corr = ccf(dataf$oil , dataf$gen_trend)
lag_corr_2 = ccf(oil_2[ , 1],trend_data_2[ , 1])

##########################regression#################################
reg_oil_trend = lm(formula = dataf$oil ~ dataf$gen_trend , x=TRUE, y=TRUE)
reg_oil_trend_2 = lm(formula = (dataf$oil)^2 ~ (dataf$gen_trend)^2 , x=TRUE, y=TRUE)
 
summary(reg_oil_trend)
summary(reg_oil_trend_2)



plot(reg_oil_trend)




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
#   t_garch.fit.trend= ugarchfit(spec=t_garch.spec.trend, data=data$oil, solver.control=list(trace = 1))
# 
#   info_garch[i,j] = infocriteria(t_garch.fit.trend)[1]
#   }
# 
# }
# ############################arma(4,3) is the best
######################################t- GARCH (1,1)
t_garch.spec.trend = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                                  variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
                                  distribution.model = "std" )
# Model Fitting
t_garch.fit.trend = ugarchfit(spec=t_garch.spec.trend, data=data$oil, solver.control=list(trace = 1))
t_garch.fit.trend@fit$hessian
windows(); par(mfcol = c(2,2))

# Conditional Volatility
plot.ts(sigma(t_garch.fit.trend),
        main="trend - Conditional Volatility t-GARCH(1,1)",
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


t_garch.spec.trend = ugarchspec(mean.model = list(armaOrder = c(1,1)), 
                                variance.model = list(garchOrder = c(2,2), model = "sGARCH"),
                                distribution.model = "std" )



t_garch.fit.trend = ugarchfit(spec=t_garch.spec.trend, data=data$oil, solver.control=list(trace = 00))
stat(t_garch.fit.trend)

t_garch.roll.trend= ugarchroll(t_garch.spec.trend,data=oil, n.head = 1, refit.every = 3)

roll_var = report(t_garch.roll.trend, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)
########################################VAR: external regressor ############


t_garch.spec.exog = ugarchspec(mean.model = list(armaOrder = c(1,1), external.regressors = NULL), 
                                  variance.model = list(garchOrder = c(2,2), model = "sGARCH",  external.regressors =trend_data_2 ),
                                  distribution.model = "std")



t_garch.fit.exog = ugarchfit(spec=t_garch.spec.exog, data=oil, solver.control=list(trace = 00))
stat(t_garch.fit.exog)

t_garch.roll.exog= ugarchroll(t_garch.spec.exog,data=oil, n.head = 1, refit.every = 3)

roll_var_ext = report(t_garch.roll.exog, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)


#####plot vari
plot(oil_2,col= "red")
lines(trend_data_2)
cor(oil_2,trend_data_2)
###########structural break#########################################
bp.oil <- breakpoints(oil ~ 1)
summary(bp.oil)
plot(bp.oil)

breakdates(bp.oil)

ci.oil<- confint(bp.oil)
breakdates(ci.oil)
ci.oil

plot(oil)
lines(ci.oil)






##################################MULTIVARIATE MODEL##################################
dcc.garch_ex.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), external.regressors = data=oil,
dccOrder = c(1,1),distribution="mvt")


dcc.garch.spec = dccspec(uspec = multispec (replicate(2, garch11.spec)), dccOrder = c(1,1),distribution="mvt")



#########################################graph export ########################à
write.xlsx(dataf, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/graph.xlsx")
write.xlsx(data, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/graph2.xlsx")
sigma_trend = t_garch.fit.trend@fit$sigma
sigma_exog = t_garch.fit.exog@fit$sigma
Var_exog = t_garch.roll.exog@forecast[["VaR"]][["alpha(5%)"]]
Var_trend=t_garch.roll.trend@forecast[["VaR"]][["alpha(5%)"]]

sigma=cbind(sigma_exog,sigma_trend[2:581])
VaR= cbind(Var_exog, Var_trend)
residuals_exog = (t_garch.fit.exog@fit[["residuals"]])
residuals_trend = (t_garch.fit.trend@fit[["residuals"]])
redisuals = cbind(residuals_exog, residuals_trend[2:581])
residuals_2 = residuals

write.xlsx(sigma, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/sigma2.xlsx")
write.xlsx(VaR, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/VaR.xlsx")
write.xlsx(residuals_exog, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/residuals.xlsx")
write.xlsx(residuals_trend, "C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND BANKING/thesis_2/apollo 2/dataset_graph/residuals_2.xlsx")


