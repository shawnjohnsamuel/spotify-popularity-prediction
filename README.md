![first-time home buyer offer price tool by shawn samuel](images/house-price-prediction-banner.png)
# Recommending "Offer Price" to First-Time Home Buyers

**Authors**: [Shawn Samuel](mailto:shawnjohnsamuel@gmail.com)  

## Overview

This project has tasked us to formulate a business project around a given data set. The data set describes house sales over a 1 year period in King County, USA. I have decided to develop a price prediction tool for a real estate agency that focuses on the needs of home buyers. I will be cleaning the data, testing various transformations like log transformation for continuous variables, one hot encoding for categorical variables, and scaling to prepare the data for multiple linear regression models. We were able to build statistically significant model that could explain 53.2% of the variation in prices from our prediciton and strongly feel that further development will lead to a more accurate tool that can be extremely useful for buyer real estate agents assisting first-time home buyers.

## Business Problem

The housing market is booming! With decreased mortgage rates, increased demand and decreased supply - housing prices are also sky high. According to the New York Times, typically 55% to 70% of American home buyers are selling one home and buying another, with the remainder buying a home for the first time. However in recent months, the number of first-time buyers has sharply risen. This leads to increased demand without an equivalent increase in supply. 

I have been tasked with helping first-time home buyers with one of the crucial elements of home buying - determining an 'offer price'. There can be many factors that contribute to this. And being first time home buyers, it can be daunting to figure out what this magic price should be. 

This important price point is often guided buy Buyer's Real Estate Agent and this data driven prediction tool is being built for one such Agency. I will use the historical data of houses sold in King County, Washington to predict the best asking price based on relevant factors.  

## Data

The data set [('kc_house_data.csv')](data/kc_house_data.csv) describes house sales in King County, Washington, USA (which includes the city of Seattle) between 2014 and 2015. There are 21,597 rows of entries across 21 columns, including the target variable 'price'. The dependant variables include information about individual homes that have been sold such as square footage, year built, and many more details. 

We will prepare and explore the data to see which ones have the most impact for asking price determination for first-time home buyers. We will focus on the middle and low end of the market in terms of price, as first time home buyers will likely not be in the luxury home market.

## Methods

I built an initial baseline model using recommended features and no transformations. After this I followed the following plan:

☐ **Create a pyfile function for subsequent analysis** - this was done to make modeling more effecient  
☐ **Identify and tackle outliers** - I noticed a bedroom outlier in particular  
☐ **Log continous variables** - to removeness skewness of original data  
☐ **One Hot Encode categorical variables** - to encode categories for modeling  
☐ **Create an age columns based on year built or renovation year** - categorize based on relative age  
☐ **Create a category of within Seattle vs. outside Seattle** - see what impact a generalized location will have on model  
☐ **Set a price ceiling** - our tool will be utilized for first-time home buyers so will narrow our house prices  

## Results

I created 6 models beyond our base model and found that there were varying R2 values ranging from .532 to .609. I found that the last model minimized the MAE and RMSE to the smallest amount. The last model was slightly underfit based on the train vs test. This model was then run on the entire data set and returned a R2 of 0.532. This means that our final model can explain 53.2% of variance in prices. Based on the train-test split, this model is generalizable and can be used for data not yet seen.  Below you can see some of the visualizations generated from data exploration:  

### Example Of Continuous Variable (Sqft vs. Price)  
![example of continuous variabe square feet versus price](images/cont_variable_sqft_vs_price.png)

### Example Of Categorical Variable (Grade vs. Price)  
![example of categorical variabe grade versus price](images/cat_variable_grade_vs_price.png)

### Interesting Finding:
![older homes in seattle limit tend to be more expensive than newer homes](images/age_vs_price_in_or_out_seattle.png)  
One interesting finding was that older homes within the Seattle city limits tend to be more expensive than newer homes. This trend is reversed outside of Seattle.   

## Recommendations

In it's current form, this model is a better predictor of price than the simple mean price of \$540,296.60 or even median of \$450,000. I believe with further modeling, it can be very useful as a tool for buyer's real estate agents making offer price recommendations for time home-buyers. They can even make recommendations to first-time home buyers to reduce their price cost. For example, our model shows that square footage, whether or not the house is in Seattle as well as age greatly affect price. So to reduce cost, first-time home buyers can opt for smaller, slightly older homes outside of Seattle.

## Future Work

I believe that having an offer price that is data-driven is a very strong tool for a Buyer's Real Estate Agent and so we would recommend this Real Estate Agency continue to support the development of this tool. This could empower the already intuitive decision making that experienced Real Estate Agents engage in. The following are some potential areas of future work:

1) Creating separate prediction tools for different types of buyers - such as those looking to flip houses, first-time home buyers and luxury home buyers    
2) Experiment with inclusion of all available parameters.  
3) Build a graphical user interface where all available information can be input for easy use of prediction tool

## For More Information

Please review the full analysis in the [Jupyter Notebook](project-notebook.ipynb) or the [presentation](project-presentation.pdf).

For additional info, contact Shawn Samuel at [shawnjohnsamuel@gmail.com](mailto:shawnjohnsamuel@gmail.com)

## Repository Structure

```
├── data
├── images
├── project-notebook.ipynb
├── project-presendation.pdf
├── README.md
└── sjs_utilities.py
```
