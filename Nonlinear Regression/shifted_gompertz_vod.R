
library(tidyverse)
library(lubridate)
library (nlstools)
library(minpack.lm)
library(ggplot2)
library(car)
library(nlshelper)

curve_model_data <- read_csv("data/modelvoddata.csv",col_names = TRUE)

# Curve Fitting Using Shifted-Gompertz Distribution
# estimation method of nonlinear ordianry least squares
# with iterative minimization method is Marquardt-Levenberg

# data cleaning for $ and commas for numeric variables
curve_model_data$VOD_POS_8_R <- str_remove_all(curve_model_data$VOD_POS_8_R,"\\$|,")
curve_model_data$Box_2D <- str_remove_all(curve_model_data$Box_2D,"\\$|,")
curve_model_data$BI_VOD_8 <- str_remove_all(curve_model_data$BI_VOD_8,"\\$|,")
curve_model_data$VOD_POS_26_R <- str_remove_all(curve_model_data$VOD_POS_26_R,"\\$|,")
curve_model_data$BI_VOD_26 <- str_remove_all(curve_model_data$BI_VOD_26,"\\$|,")
curve_model_data$MED_VOD_BI_8 <- str_remove_all(curve_model_data$MED_VOD_BI_8,"\\$|,")
curve_model_data$MED_VOD_BI_26 <- str_remove_all(curve_model_data$MED_VOD_BI_26,"\\$|,")
curve_model_data$index_act <- str_remove_all(curve_model_data$index_act,"\\$|,")

curve_model_data$VOD_POS_8_R <- as.numeric(curve_model_data$VOD_POS_8_R)
curve_model_data$Box_2D <- as.numeric(curve_model_data$Box_2D)
curve_model_data$BI_VOD_8 <- as.numeric(curve_model_data$BI_VOD_8)
curve_model_data$VOD_POS_26_R <- as.numeric(curve_model_data$VOD_POS_26_R)
curve_model_data$BI_VOD_26 <- as.numeric(curve_model_data$BI_VOD_26)
curve_model_data$MED_VOD_BI_8 <- as.numeric(curve_model_data$MED_VOD_BI_8)
curve_model_data$MED_VOD_BI_26 <- as.numeric(curve_model_data$MED_VOD_BI_26)
curve_model_data$index_act <- as.numeric(curve_model_data$index_act)

curve_model_data$Mo <- as.character(curve_model_data$Mo)
curve_model_data$YR <- as.character(curve_model_data$YR)
curve_model_data$Month_start <- dmy(curve_model_data$Month_start)

# hold-out for train data and test data
curve_model_train_data <- curve_model_data[1:100,]
curve_model_test_data <- curve_model_data[101:181,]

# maximum number of iternations with nlslm
maxiter <- 1024

# convergence criterion
converge <- .000001
# impose lower boundary constraints on both parameters
bounds <- c(.000000001,.000000001,.000000001)

# initialize starting values for model parameters to be estimated
m <- 10000
eta = 10
b = .4

# cdf of shifted gompertz with M as a parameter
cdf_obj_fn <- function(m,b,eta,t){
  m*((1-exp(-(b)*(t)))*(exp(-(eta)*(exp(-(b)*(t))))))
}

model_fn <- function(m,b,eta,t){
  cdf_obj_fn(m,b,eta,t) - lag(cdf_obj_fn(m,b,eta,t),default = 0)
}

model_sgompertz <- nlsLM(MED_VOD_BI_8~model_fn(m,b,eta,t),data=curve_model_train_data,start=list(m=m,b=b,eta=eta),
                           algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, gtol=converge))

# model summary
overview(model_sgompertz)

# Model diagnostics
# Check the residuals
qqPlot(residuals(model_sgompertz))

# In the residuals vs. fitted vales visualization there is no
# decernable pattern such as a megaphone, also the same with autocorrelation
plot(fitted(model_sgompertz), residuals(model_sgompertz), pch=16, col="dimgrey")
abline(h=0)

nls_residuals <- nlsResiduals(model_sgompertz)
plot(nls_residuals)

shapiro.test(residuals(model_sgompertz))

predictions <- predict(model_sgompertz,newdata=curve_model_data)

# curve plot, merge predictions with data
viz <- data.frame("BI_VOD_8_PRED"=predictions,t=seq(1:length(predictions)))
viz_2 <- left_join(curve_model_data,viz,by="t")

value_col <- c("red","blue","blue","black")
value_col_2 <- c("Forecast", "MED_VOD_BI_8_Line", "MED_VOD_BI_8_Points", "Forecast New Data")

gg <- ggplot(viz_2) +
  labs(title="Shifted-Gompertz Fit Median",xlab("t"),ylab("BI_VOD_8")) +
  geom_line(aes(x = t, y = BI_VOD_8_PRED,color="1")) +
  geom_point(aes(x = t, y = MED_VOD_BI_8,color="2")) +
  geom_line(aes(x = t, y = MED_VOD_BI_8,color="3")) +
  geom_vline(aes(xintercept=100,color="4")) +
  theme(legend.position="bottom") +
  scale_color_manual(labels = value_col_2, values = value_col) +
  theme(legend.title = element_blank())

gg


# cdf of shifted gompertz without M as a parameter
bounds <- c(.000000001,.000000001)

cdf_obj_fn <- function(b,eta,t){
  ((1-exp(-(b)*(t)))*(exp(-(eta)*(exp(-(b)*(t))))))
}

model_fn <- function(b,eta,t){
  cdf_obj_fn(b,eta,t) - lag(cdf_obj_fn(b,eta,t),default = 0)
}

model_sgompertz_2 <- nlsLM(MED_VOD_BI_8~model_fn(b,eta,t),data=curve_model_train_data,start=list(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, gtol=converge))

overview(model_sgompertz_2)

# F-test shows the model with M is a better fit
anova(model_sgompertz,model_sgompertz_2)















#scrape code

# fit the model using Gaussian-Newton optimization
model_sgompertz_gn <- nls(MED_VOD_BI_8 ~ cdf_obj_fn,data=curve_model_data,start=c(b=b,eta=eta),
                          lower=bounds, trace=TRUE, control = list(maxiter = maxiter, tol=converge))
model_sgompertz_gn

# fit the model using nlsML for LM optimization for other types use nls
model_sgompertz <- nlsLM(MED_VOD_BI_8~cdf_obj_fn(b,eta),data=curve_model_data,start=list(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, tol=converge))


cdf_obj_fn_2 <- function(b,eta){
  ((1-exp(-(b)*(curve_model_train_data$t)))*(exp(-(eta)*(exp(-(b)*(curve_model_train_data$t))))))
}
cdf_obj_fn_2(0.00743378,0.458808)
cdf_obj_fn(34253.3,0.00743378,0.458808)

# ORIGNAL COMMENTED OUT CODE USING DIFFUSION PACKAGE
# fitVMED <- diffusion(curve_model_data[, 11], type = "gsgompertz", optim = ("hj")) 
# Needed to use HJ because default rm looked bad
# optim}{optimization method to use. This can be "nm" for Nelder-Meade or "hj" for Hooke-Jeeves.} from below
# plot(fitVMED)
# 
# wb2<-predict(fitVMED,200) # Contains both fit past & predict - future
# plot(wb2)
# 
# test_VMED <-as.data.frame(wb2$fit) 
# fCast_VMED <-as.data.frame(wb2$frc)
# Both <- rbind(test_VMED, fCast_VMED)
# 
# write.csv(Both , "C:/____S3/EST Growth 07 2019/Data Sets/VOD_111119.csv")
# write.foreign(Both,"C:/____S3/EST Growth 07 2019/Data Sets/VOD_111119.sas7bdat", "C:/____S3/EST Growth 07 2019/Data Sets/VODcode.sas",package = c("SAS"))
# End R
# 
# class(c(curve_model_data[, 11]))
