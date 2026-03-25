#this uses data from HEPData record 93086, Table 5
#dataframe uses mass (GeV), counts, bin width and poisson uncertainty

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('data/data_.csv', skipfooter=65, skiprows=11, delimiter=',') #load data into dataframe, skip unnecessary lines

    #make columns
    df.columns = ['m_center', 'm_lo', 'm_hi', 'counts']

    df['uncertainty'] = np.sqrt(df['counts']) #poisson uncertainty, because no uncertainty specified 
    df['bin_width'] = df['m_hi'] - df['m_lo']

    #print(df)  #print to check



    return (
        df['m_center'].values,
        df['counts'].values,
        df['uncertainty'].values,
        df['bin_width'].values,
        df['m_lo'].values,
        df['m_hi'].values
    ) #return everything into separate columns

# Call the function and unpack the returned values
m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data()

#plotting raw data
#could also plot on linear scale, would remove plt.yscale
if __name__ == '__main__': #make sure that the data is plotted only when this file is run directly
    plt.figure(figsize=(10,6))
    plt.errorbar(m_center, counts, yerr=uncertainty, fmt='o', color='black', ms=4, capsize=2, alpha=0.6)
    plt.xlabel(r'$m_{jj}$ [GeV]', fontsize=11)
    plt.ylabel('Counts', fontsize=11)
    plt.yscale('log')  #log scale
    plt.title('Raw Data')
    plt.savefig('data\data_raw.png', dpi=300)
    plt.show()
