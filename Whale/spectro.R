library(seewave);
library(tuneR);
library(ggplot2);
library(wavelets);

train.right.whale <- c(6, 7, 9, 12, 28);
train.no.right.whale <- c(1, 2, 3, 4, 5);

train12 <- readWave("small_data_sample/right_whale/train12.wav");
train28 <- readWave("small_data_sample/right_whale/train28.wav");
train6 <- readWave("small_data_sample/right_whale/train6.wav");
train7 <- readWave("small_data_sample/right_whale/train7.wav");
train9 <- readWave("small_data_sample/right_whale/train9.wav");

train1 <- readWave("small_data_sample/no_right_whale/train1.wav");
train2 <- readWave("small_data_sample/no_right_whale/train2.wav");
train3 <- readWave("small_data_sample/no_right_whale/train3.wav");
train4 <- readWave("small_data_sample/no_right_whale/train4.wav");
train5 <- readWave("small_data_sample/no_right_whale/train5.wav");


train12.mra <- mra(as.numeric(train12@left), n.levels=8);

spectro(train12,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train28,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train6,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train7,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train9,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))

spectro(train1,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train2,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train3,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train4,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
spectro(train5,f=22050,wl=512,ovlp=50,zp=16,collevels=seq(-40,0,0.5))
