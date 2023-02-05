library(tidyverse)
dataset = list()
loc = "/home/chanyu/Desktop/school/DataMining/dataset/"
citys = c("tp", "tc", "tn", "ks", "nt", "ty")
for(y in 1:11) {
  if (y == 1) {
    for (c in city) {
      file = paste0("101", "-4 ", c)
      dataset[[file]] = read.csv(paste0(loc, file, ".csv"))
    }
  } else if (y == 11) {
    for (s in 1:3) {
      for (c in city) {
        file = paste0("1", y, "-", s, " ", c)
        dataset[[file]] = read.csv(paste0(loc, file, ".csv"))
      }
    }
  } else {
    for (s in  1:4) {
      for (c in city) {
        file = ifelse(y < 10,
                      paste0("10", y, "-", s, " ", c),
                      paste0("1", y, "-", s, " ", c)
        )
        dataset[[file]] = read.csv(paste0(loc, file, ".csv"))
      }
    }
  }
}

for (i in 1:length((dataset))) {
  data <- dataset[[i]] %>%
    select(-c(
      非都市土地使用分區, 
      非都市土地使用編定, 
      車位移轉總面積.平方公尺., 
      主要建材, 備註, 編號, 
      移轉編號)) %>%
    filter(主要用途 == "住家用") %>%
    mutate(電梯 = ifelse(is.na(電梯), 0, 電梯))
  loc = paste0("/home/chanyu/Desktop/school/DataMining/project/dataset/dataset/", 
               names(dataset[i]), ".csv")
  write.csv(data, loc)
}

# merge data
last_year <- dataset[grep("^110-4|111-[1-3]", names(dataset))]
city_df = list()
for (i in citys) {
  city_df[i] = data_frame()
}

for (i in 1:length(last_year)) {
  city_name = gsub("[0-9]+-[0-9]+\\s+", "", names(last_year[i]))
  year = gsub("[a-z]+", "", names(last_year[i]))
  df = last_year[[i]] %>%
    mutate(
      period = year, 
      電梯 = ifelse(電梯 == "無", 0, 1),
      建物現況格局.隔間 = ifelse(建物現況格局.隔間 == "無", 0, 1)
  )
  if (city_name == "tp") {
    cat(nrow(df), "\n")
  }
  city_df[[city_name]] = rbind(city_df[[city_name]], df)
}

for (i in 1:length((city_df))) {
  loc = paste0(
    "/home/chanyu/Desktop/school/DataMining/project/dataset/dataset/last_year/", 
    "LastYear_", names(city_df[i]), ".csv"
  )
  write.csv(city_df[[i]], loc)
}
