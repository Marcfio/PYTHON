####cleaning#########
rm(list=ls()) #clear all variables
graphics.off()  # clean up graphic window
cat("\014") #clear the workspace
# Loading Libraries
library(rugarch); library(fGarch); library(rmgarch); library(zoo); library(ggplot2)


### import data##
data = read.csv("C:/Users/MARCOFIORAVANTIPC/Google Drive/MSC FINANCE AND trendNKING/thesis_2/apollo 5/dataset/dataset2_dollar_eur.csv", header = TRUE, stringsAsFactors = FALSE)
dollar_eur = ts(data[6])
trend_data = ts(data[25])
dates = data[2:nrow(data), 2  ]
dollar_eur = 100*diff(log(dollar_eur))
trend_data = diff(log(trend_data))
dataf = data.frame(dates,dollar_eur,trend_data)

####plot and summary####################

plot(dataf$wei_dollar_eur_gen_trend,type = 'l' )
plot(dataf$dollar_eur,type = 'l' )
summary(trend_data)
summary(dollar_eur)
plot(dataf$wei_dollar_eur_gen_trend,dataf$dollar_eur)

dollar_eur_2 = (dollar_eur)^2 ##############squared returns
trend_data_2 = (trend_data)^2 ##############squared returns


reg_dol_trend = lm(formula = dataf$dollar_eur ~ dataf$wei_dollar_eur_gen_trend , x=TRUE, y=TRUE)
reg_dol_trend_2 = lm(formula = (dataf$dollar_eur)^2 ~ (dataf$wei_dollar_eur_gen_trend)^2 , x=TRUE, y=TRUE)
 

plot(reg_dol_trend)

n_lag = 10
lag_corr = integer(n_lag+1)
for (i in (0:n_lag)){
  lag = i
  lag_corr[i+1]= cor(dataf$dollar_eur[(1+lag): length(dataf$dollar_eur)], dataf$wei_dollar_eur_gen_trend[1:(length(dataf$wei_dollar_eur_gen_trend)-lag)])

}

n_lag = 10
lag_corr_2 = integer(n_lag+1)
for (i in (0:n_lag)){
  lag = i
  lag_corr_2[i]= cor(dollar_eur_2[(1+lag): length(dollar_eur_2)], trend_data_2[1:(length(trend_data_2)-lag)])
  
}




######################################t- GARCH (1,1)
t_garch11.spec.trend = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                               variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
                               distribution.model = "norm" )
# Model Fitting
t_garch11.fit.trend = ugarchfit(spec=t_garch11.spec.trend, data=data$dollar_eur, solver.control=list(trace = 1))
t_garch11.fit.trend
t_garch11.fit.trend@fit$hessian
windows(); par(mfcol = c(2,2))

# Conditional Volatility
plot.ts(sigma(t_garch11.fit.trend),
        main="trend - Conditional Volatility t-GARCH(1,1)",
        xlab = "Number of Observations", ylab="sqrt(h_t)", col="green4")
# Standardized Residuals
std.res_tgarch_trend = ts(as.numeric(residuals(t_garch11.fit.trend, standardize=TRUE))) 
plot(std.res_tgarch_trend, main='Standardized Residuals', xlab='Number of Observations', ylab='Standardized Residuals')
acf((std.res_tgarch_trend)^2, main='ACF - Standardized Squared Residuals')
plot(t_garch11.fit.trend, which=9)

summary(std.res_tgarch_trend); skewness(std.res_tgarch_trend); kurtosis(std.res_tgarch_trend)
summary(sigma(t_garch11.fit.trend))
