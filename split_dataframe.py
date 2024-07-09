import json
import pandas as pd
from sklearn.model_selection import train_test_split

def main(dataframe):
    dataX = dataframe["Abstract"]
    dataY = dataframe["Keywords generated"]

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)
    # validation is now 15% of the initial data set
    x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    data_dev =  {'Abstract': x_dev, 'Keywords': y_dev}
    data_test =  {'Abstract': x_test, 'Keywords': y_test}
    data_train =  {'Abstract': x_train, 'Keywords': y_train}

    dev_df = pd.DataFrame(data_dev)
    test_df = pd.DataFrame(data_test)
    train_df = pd.DataFrame(data_train)
    
    dev_df.to_csv('dev_df.csv')
    test_df.to_csv('test_df.csv')
    train_df.to_csv('train_df.csv')


    # Export DataFrame to TSV file
    dataframe.to_csv('output.tsv', sep='\t', index=False)
    
    return dev_df.to_csv('dev_df.csv'),test_df.to_csv('test_df.csv'),train_df.to_csv('train_df.csv')

