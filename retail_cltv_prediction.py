##############################################################
# CLTV Prediction with BG-NBD & Gamma-Gamma
##############################################################

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
# 5. Creating Segments by CLTV



##############################################################
# 1. Data Preparation
##############################################################
# Data
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The dataset contains sales data of an online store between 01/12/2009 - 09/12/2011

# Variables
# InvoiceNo: Invoice number. The unique number of each transaction. Aborted operation if it starts with C.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in GBP)
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.


#pip install lifetimes

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = pd.read_excel("dataset/online_retail_II.xlsx", sheet_name="Year 2009-2010")

df.describe().T

df.head()

df.isnull().sum()

#########################
# Data Preprocessing
#########################

df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0]

df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")

replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

#########################
# Preparation of Lifetime Data Structure
#########################


cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head()


cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]


cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

#to express in weekly terms
cltv_df["recency"] = cltv_df["recency"] / 7

#to express in weekly terms
cltv_df["T"] = cltv_df["T"] / 7

##############################################################
# 2. BG-NBD Modeling
##############################################################
#penalizer_coef: penalty coefficient
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

##############################################################
# 3. GAMMA-GAMMA Modeling
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##############################################################
# 4. Calculate CLTV Prediction with BG-NBD & GG
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 mon
                                   freq="W",  # Frequency information of T.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(5)


##############################################################
# 5. Creating Segments Based on CLTV Prediction Values
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})
















