![ZHVI Growth Predictions](images/0_Zillow_Time_Series_Banner.png)
# Predicting the Popularity of Songs on Spotify
**Author:** Shawn Samuel

## Overview

Using ARIMA time series modeling, predictions were made and validated for median home prices for a desired set of zip codes. The predictions and historical analysis for each zip code were compared to select the top 5 zip codes to invest in based on potential for growth. The data is from the Zillow Home Value Index from 1996 to 2021. The hypothecial use case is to help first time home buyers to think like investors as the weigh the many factors in purchasing a home.

## Business Problem

For this project, I am answering the question "What are the 5 best zip codes to invest in?" for a fictional real estate agency: ACME Realtors. To further clarify, I will be designing a tool for first-time home buyers (FTHB) to help them think like investors. Generally, first-time home buyers are overwhelmed by many factors when considering what home to buy. The investment potential of particualr areas may not be top of mind. Also, FTHB may not have access to high-performance data analytics and machine learning tools that are usually geared towards real estate investors. The use case for this tool will be for FTHB who have an idea (either due to preference or restriction) of a group of zipcodes they are interested in. Once this machine learning tool is built, it can be used as an added-value product ACME Realtors can provide their FTHB clients. For the FTHB, it will be a resource to enable them to think about their home purchase as an investment and maximize returns.

## Data Understanding

The data contains monthly ZHVI median pricing for 30,297 zip codes across America. This can be used to extract time series for individual zip codes. We are then able to use ARIMA modeling on the individual zip code time series to make predictions.

The ZHVI was obtained from Zillow's [Research Website](https://www.zillow.com/research/data). According to Zillow, "Zillow Home Value Index is a smoothed, seasonally adjusted measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 35th to 65th percentile range. Check out this [overview of ZHVI](https://www.zillow.com/research/zhvi-methodology-2019-highlights-26221) and a [deep-dive into its methodology](https://www.zillow.com/research/zhvi-methodology-2019-deep-26226/)." This [video](https://www.youtube.com/embed/rousqnB-G2c) also provides a greater understanding of the ZHVI.

For our purposes, index appreciation can be interpreted as the market’s total appreciation. In other words, the ZHVI appreciation can be viewed as the theoretical financial return that could be gained from buying all homes in a given subset (by geography and/or home type) in one period and selling them in the next period. Although this index includes homes of all sizes and types, including those that are not of interest to the client, it gives a general and powerful perspective on the price trend in a given area. 

## Methods

Our methodoolgy had four main components. 

1. Preprocessing - this included pulling in the latest data available from Zillow, melting the data so that it reflected time series for individual zip codes and then filtering those columns to contain information for desired zipcoddes. One important thing to note is that we made the decision to build our model on the trend from the lowest point (in terms of price) after the housing market crash of 2008.
![Post Market Crash Timeseries Cutoff](images/1_time_cutoff.png)
2. Build a Model - this included
   
## Results

## Conclusion

## Future Work

I suggest further optomizing the model. Some potential areas include:

1. 

## For More Information

Please review the full analysis in the [Jupyter Notebook](project-notebook.ipynb) or the [presentation](project-presentation.pdf).

For additional info, contact Shawn Samuel at [shawnjohnsamuel@gmail.com](mailto:shawnjohnsamuel@gmail.com)

## Repository Structure

```
├── data
├── images
├── README.md
├── project-notebook.ipynb
├── project-presentation.pdf
└── sjs_utilities.py
```
