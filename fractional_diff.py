import pandas as pd
import fml_lib
import sys
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as stats
# Chapter 5


def exercise1():
    mu,sigma = 0, 0.1
    time_series = np.random.normal(mu,sigma,1000)

    #Computing ADF statistic on the series and finding the p value
    time_adf = stats.adfuller(time_series)
    print("P value is :" + str(time_adf[1]))
    #Cumulative sum of observations
    time_adf_cum = np.cumsum(time_series)

    cum_adf = stats.adfuller(time_adf_cum)
    print(cum_adf)

    # This seems off, doesn't seem right that the order of integration is 0
    print("Order of integration is " + str(cum_adf[2]))
    print("P value is " + str(cum_adf[1]))

    #Differentiating the series twice and getting p value
    time_first_diff = np.diff(time_series)
    time_second_diff = np.diff(time_first_diff)
    second_diff_adf = stats.adfuller(time_second_diff)
    print("P value of twice differentiated series is :" + str(second_diff_adf[1]))

def exercise2():
    cycles = 2
    resolution = 10000
    length = np.pi*2*cycles
    my_wave = np.sin(np.arange(0, length, length / resolution))
    sine_adf = stats.adfuller(my_wave)
    print("P value for sine wave is: " + str(sine_adf[1]))
    shifted_wave = my_wave + 5
    shifted_wave_cum = np.cumsum(shifted_wave)
    cum_shifted_adf = stats.adfuller(shifted_wave_cum)
    print("P value for cumulative shifted wave is: " + str(cum_shifted_adf[1]))
    shifted_wave_cum = pd.DataFrame(shifted_wave_cum)
    frac_diff = fml_lib.fracDiff(shifted_wave_cum,0.0000000001)
    frac_diff_adf = stats.adfuller(frac_diff)
    print("P value for expanding window fracdiff is: " + str(frac_diff_adf[1]))

    ffd_diff = fml_lib.fracDiff_FFD(shifted_wave_cum, 1)
    ffd_diff_adf = stats.adfuller(ffd_diff)
    print("P value for FFD fracdiff is: " + str(ffd_diff_adf[1]))

if __name__ == "__main__":
    # chapter 5 exercise 1
    #exercise1()

    # Exercise 2
    exercise2()
    


