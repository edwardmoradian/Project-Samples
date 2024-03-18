
library(tidyverse)
library(lubridate)
library (nlstools)
library(minpack.lm)
library(ggplot2)
library(TSA)

curve_model_data <- read_csv("data/modelvoddata.csv",col_names = TRUE)

# Curve Fitting Using Shifted-Gompertz Distribution
# Includes: covariance matrix of the parameter estimates (covb)
# missing values being tracked on an equation-by-equation basis (missing=pairwise)
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

# maximum number of iternations
maxiter <- 32000

# convergence criterion
converge <- .000001
# impose lower boundary constraints on both parameters
bounds <- c(.000000001,.000000001,.000000001)

# initialize starting values for model parameters to be estimated
m <- 10000
eta = 10
b = .4


# cdf of shifted gompertz
cdf_obj_fn <- function(m,b,eta){
  m*((1-exp(-(b)*(curve_model_train_data$t)))*(exp(-(eta)*(exp(-(b)*(curve_model_train_data$t))))))
}


# fit the model using nlsML for LM optimization for other types use nls
model_sgompertz <- nlsLM(MED_VOD_BI_8~cdf_obj_fn(m,b,eta),data=curve_model_train_data,start=list(m=m,b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, gtol=converge))

model_sgompertz
overview(model_sgompertz)

warnings()


model_sgompertz <- nlsLM(MED_VOD_BI_8~10000*cdf_obj_fn(b,eta),data=curve_model_data,start=list(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, tol=converge))

# plot of the dataset
model_sgompertz_viz <- ggplot(curve_model_data, aes(x=YR,y=MED_VOD_BI_8)) +
  xlab("Year") + ylab("MED VOD BI B") +
  geom_point() 

model_sgompertz_viz



# scrap code

# cdf of shifted gompertz
cdf_obj_fn <- (1-exp(-(b)*(curve_model_data$t-gstart+1)))*(exp(-(eta)*(exp(-(b)*(curve_model_data$t-gstart+1)))))
formula <- as.formula(MED_VOD_BI_8 ~ cdf_obj_fn)
# zlag(cdf_obj_fn)

# fit the model using nlsML for LM optimization for other types use nls
model_sgompertz <- nlsLM(formula,data=curve_model_data,start=c(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, tol=converge))
model_sgompertz

# fit the model using Gaussian-Newton optimization
model_sgompertz_gn <- nls(MED_VOD_BI_8 ~ cdf_obj_fn,data=curve_model_data,start=c(b=b,eta=eta),
                          lower=bounds, trace=TRUE, control = list(maxiter = maxiter, tol=converge))
model_sgompertz_gn





bounds <- c(.000000001,.000000001)

cdf_obj_fn <- function(b,eta){10000*(
  (1-exp(-(b)*(curve_model_data$t)))*(exp(-(eta)*(exp(-(b)*(curve_model_data$t))))))
}

# fit the model using nlsML for LM optimization for other types use nls
model_sgompertz <- nlsLM(MED_VOD_BI_8~cdf_obj_fn(b,eta),data=curve_model_data,start=list(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, tol=converge))



bounds <- c(.000000001,.000000001)

cdf_obj_fn <- function(b,eta){10000*(
  (1-exp(-(b)*(curve_model_data$t)))*(exp(-(eta)*(exp(-(b)*(curve_model_data$t))))))
}

# fit the model using nlsML for LM optimization for other types use nls
model_sgompertz <- nlsLM(MED_VOD_BI_8~cdf_obj_fn(b,eta),data=curve_model_data,start=list(b=b,eta=eta),
                         algorithm = "LM",lower=bounds, trace=TRUE,control = list(maxiter = maxiter, tol=converge))



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
