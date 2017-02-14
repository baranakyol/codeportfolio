install.packages('locfit')
install.packages('bootstrap')
install.packages('bisoreg')

library('locfit')
library('bootstrap')
library('bisoreg')

# Make example reproducible
set.seed(19)

# Create a curve with noise
x <- 1:120

#signal function
g_signal <- function(x,period){ return(sin(2*pi*x/period)) }

period <- 120

ynonoise <- g_signal(x,period)
y <- g_signal(x,period) + runif(length(x),-1,1)


# Plot points on noisy curve
plot(x,y, main="Signal Curve + 'Uniform' Noise")
mtext("showing loess smoothing (local regression smoothing)")

lines(x,ynonoise,col='blueviolet',lwd=5)

#5-fold Cross Validation to select optimal bandwidth for loess
yloess5 <-loess.wrapper(x, y, span.vals = seq(0.25, 1, by = 0.05), folds = 5)




spanlist <- c(yloess5$s, 0.1, 0.5, 1, 1.5, 2)

for (i in 1:length(spanlist))
{
  y.loess <- loess(y ~ x, data.frame(x=x, y=y), span=spanlist[i])
  y.predict <- predict(y.loess, data.frame(x=x))
  
  # Plot the loess smoothed curve
  lines(x,y.predict,col=i)
  
  # Find peak point on smoothed curve
  peak <- optimize(function(x, model)
    predict(model, data.frame(x=x)),
    c(min(x),max(x)),
    maximum=TRUE,
    model=y.loess)
  
  # Show position of smoothed curve maximum
  points(peak$maximum,peak$objective, pch=FILLED.CIRCLE<-19, col=i)
}

legend (0,-0.8,
        c(paste("span=", formatC(spanlist, digits=2, format="f"))),
        lty=SOLID<-1, col=1:length(spanlist), bty="n")

# Make example reproducible
set.seed(19)

FullList <- 1:120
x <- FullList

ynonoise <- g_signal(x,period)

# "randomly" make 15 of the points "missing"
MissingList <- sample(x,15)
x[MissingList] <- NA

# Create sine curve with noise
y <- g_signal(x,period) + runif(length(x),-1,1)


# Plot points on noisy curve
plot(x,y, main="Sine Curve + 'Uniform' Noise")
lines(FullList,ynonoise,col='blueviolet',lwd=5)
mtext("Using loess smoothed fit to impute missing values")

#5-fold Cross Validation to select optimal bandwidth for loess
yloess5 <-loess.wrapper(x, y, span.vals = seq(0.25, 1, by = 0.05), folds = 5)

spanlist <- c(yloess5$s, 0.1, 0.5, 1, 1.5, 2)

for (i in 1:length(spanlist))
{
  y.loess <- loess(y ~ x, span=spanlist[i], data.frame(x=x, y=y))
  y.predict <- predict(y.loess, data.frame(x=FullList))
  
  # Plot the loess smoothed curve showing gaps for missing data
  lines(x,y.predict,col=i)
  
  # Show imputed points to fill in gaps
  y.Missing <-  predict(y.loess, data.frame(x=MissingList))
  points(MissingList, y.Missing, pch=FILLED.CIRCLE<-19, col=i)
}


legend (0,-0.8, c(paste("span=", formatC(spanlist, digits=2, format="f"))),
        lty=SOLID<-1, col=1:length(spanlist), bty="n")

