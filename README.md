![ZHVI Growth Predictions](images/0_Zillow_Time_Series_Banner.png)
# Predicting Growth in Zillow Home Value Index
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

1. **Preprocessing** - this included pulling in the latest data available from Zillow, melting the data so that it reflected time series for individual zip codes and then filtering those columns to contain information for desired zipcoddes. One important thing to note is that we made the decision to build our model on the trend from the lowest point (in terms of price) after the housing market crash of 2008.
![Post Market Crash Timeseries Cutoff](images/1_time_cutoff.png)
2. **Build a Model** - for modeling we focused on the [pmd auto ARIMA](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html) model. This included conducting stationairy tests, testing various train-test-splits, writing [custom functions](sjs_utilities.py) for ease of replication, buidling model for each of the desired zip codes.
3. **Make Predicitons** - the model was then utilized to make and plot predictions for the next 12 months for each zip code as well as the confidence interval. 
4. **Recommend Top Zip Codes** - using the predictions and historical information for each zip code, various filters were applied to select the top five recommended zip codes
   
## Results

Below is an example of a good model and a bad model: ![Good model and bad model](images/2_results_models.png)

We created a separate ARIMA model for each zip code and added metrics and information for each to a dataframe for comparison. Using the mean as a baseline we filtered the zipcodes based on a few factors:
- Historical performance is higher than average
- The predicted growth is higher than average
- The AIC score is less than (better than) the average AIC score
- The MAE of the validation set is less than the average MAE
- The predicted growth is greater than 10%

We then sorted the dataframe by historical growth. The reason for this is because this is an established value that we know to be true. Based on all of these factors, we can now select the top 5 cities as our recommendation for zip codes to invest in.

Based on this, for this particular set of zip codes, we would select the following cities with their projected growth :
1. East Rutherford - 16.35%
2. Garfield - 18.99%
3. Ridgefield Park - 20.02%
4. Hackensack - 15.41%
5. Bogota - 22.06%

## Conclusion

As a First Time Homebuyer myself, I believe there is great value in a tool such as what we have created the backbone for. Price is an interesting and simple aggregate of many factors that influence the profitablity of homes in a given zip code. A time series model is a powerful way to utilize this important predictor to empower First Time Home Buyers who often get lost in many details. There are many elements of homebuying, if viewed as investment, would be extremely financially beneficial for FTHs. This applies to the home search process as well as the financing of the purchase. Fine-tuning and providing a tool like the above would be an important enabling and equipping value that a Real Estate Agency (such ACME) could provide it's clients. 

This ARIMA model has been automatically tuned for best parameters. As described below, we think exploring more complex models and datasets could inform this early iteration of the prediction tool. Also the year 2020 and 2021 have been vastly different than the previous decade leading up to it, to the point of some claiming we are in a bubble. There are many factors contributing to the high prices which make modeling a time series somewhat restrictive. It may be beneficial to validate on other similar historical time frames. Also, any predicitons must be taken with a grain of salt as prices simply cannot go up forever, as any model that weighs the recent past may predict. We think further work must be done before this model can be ready for regular use. 

## Future Work

There are multiple areas of potential future work:

1. Design a proprietary scoring model that takes the various relevant factors into account
2. Explore other house pricing datasets and modeling packages such as Facebook Prophet 
3. Build a GUI where users can select desired zipcodes from a map to feed into the model and return results

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
