import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns


class Eda:

    def eda(self, train: pd.DataFrame):

        # self.plotPriceCategory(train)

        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', 5000)
        # print(list(train.columns))
        # train.columns = train.columns.str.strip()

        # self.plotAgainstMeanOfPrices(train, 'BldgType')
        # self.plotAgainstMeanOfPrices(train, "MSZoning")
        # self.plotNeighborhoodAgainstMeanOfPrices(train)
        #self.lotAreaAndPriceCategoryHeatMap(train)
        self.lotAreaAndPriceCategoryGraph(train)

        # self.processOverallQualandOverallCond(train)

        return train


    def plotPriceCategory(self, train: pd.DataFrame):

        # grab the counts associated with each Price Category 
        priceCategory_counts = train['PriceCategory'].value_counts().sort_index()

        plt.figure(figsize=(10,6))
        priceCategory_counts.plot(kind='bar', rot=0)
        plt.xlabel('Price Category (price range increases per label)')
        plt.ylabel('Number of Properties')
        plt.title('Distribution of Properties by Price Label')

        # save the figure into the graphs folder 
        plt.savefig('eda_price_category_distribution.png')

        return

    def plotAgainstMeanOfPrices(self, train: pd.DataFrame, featureName:str):

        # take the mean of the SalePrice feature for each label in featureName column 
        mean_sale_price = train.groupby([featureName])['SalePrice'].mean().sort_index()

        # make the plot 
        plt.figure(figsize=(10,6))
        mean_sale_price.plot(kind = 'line', marker='o', linestyle='-',color='b')

        plt.grid()
        #plt.xticks(ticks=range(0, len(mean_sale_price), 2), labels=mean_sale_price.index[::2], rotation=45)
        plt.xlabel('Type of Dwelling')
        plt.ylabel('Mean Sale Price')
        title = "Mean Sale Price by " + featureName

        plt.title(title)

        file_path = "eda_mean_sale_price_vs_" + featureName
        plt.savefig(file_path)
        return

    def plotNeighborhoodAgainstMeanOfPrices(self, train: pd.DataFrame):

        # take the mean of the SalePrice feature for each label in featureName column 
        mean_sale_price = train.groupby(['Neighborhood'])['SalePrice'].mean().sort_index()

        # make the plot 
        plt.figure(figsize=(14,10))
        mean_sale_price.plot(kind = 'line', marker='o', linestyle='-',color='b')

        plt.grid()
        plt.xticks(ticks=range(0, len(mean_sale_price), 1), labels=mean_sale_price.index, rotation=45)
        #plt.xticks(ticks=range(0, len(mean_sale_price), 2), labels=mean_sale_price.index[::2], rotation=45)
        plt.xlabel('Neighborhood')
        plt.ylabel('Mean Sale Price')
        title = "Mean Sale Price by Neighborhood" 

        plt.title(title)

        file_path = "eda_mean_sale_price_vs_neighborhood"
        plt.savefig(file_path)
        return

    def processOverallQualandOverallCond(self, train: pd.DataFrame):

        overallQual_mode = train["OverallQual"].mode()
        print(overallQual_mode)

        overallCond_mode = train["OverallCond"].mode()
        print(overallCond_mode)

        self.plotAgainstMeanOfPrices(train, "OverallQual")
        self.plotAgainstMeanOfPrices(train, "OverallCond")

        return
    
    def lotAreaAndPriceCategoryGraph(self, train:pd.DataFrame):

        sns.lmplot(x='LotArea', y='SalePrice', data=train, line_kws={"color":"red"}, height=6,aspect=1.5)
        plt.xlabel('Lot Area (sq ft)')
        plt.ylabel('Sale Price')
        plt.title('Scatter Plot of Sale Price vs Lot Area with Trend Line')
        plt.show()

        return
    
    def CondvsQual_heatmap(self, train: pd.DataFrame):

        heatmap_data = train.pivot_table(values='SalePrice', index='OverallQual', columns='OverallCond',aggfunc= 'mean')

        plt.figure(figsize=(10,6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".0f", linewidths=0.5)
        plt.axhline(y=0, color='black', linewidth=1.5)  # Top horizontal line
        plt.axhline(y=heatmap_data.shape[0], color='black', linewidth=1.5)  # Bottom horizontal line
        plt.axvline(x=0, color='black', linewidth=1.5)  # Left vertical line
        plt.axvline(x=heatmap_data.shape[1], color='black', linewidth=1.5) 
        plt.xlabel('Overall Condition')
        plt.ylabel('Overall Quality')
        plt.title('Heatmap of Average Sale Price by Overall Quality and Condition')
        plt.show()

        return