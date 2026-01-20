import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # Set by PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

population_data_path = resource_path("dependencies/Bev-KS-NW-2009-2022.xlsx")
payins_data_path = resource_path("dependencies/71717-05i-KS-2009-2022_Payins Short.xlsx")
payouts_data_path = resource_path("dependencies/71717-15i-KS-2009-2022_Payouts Short.xlsx")

def find_fin_twin(df, year, metric, dftype):
    year = str(year)
    df = df[df['Year'] == year].reset_index(drop=True)
    municipalities = df['Gemeinde']
    
    drop_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde']
    drop_columns.append('EZ-Konto' if dftype == 'payins' else 'AZ-Konto')
    
    financial_df = df.drop(columns=drop_columns, errors='ignore')
    financial_df = financial_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    pairwise_distances = pdist(financial_df, metric=metric)
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, np.inf)
    
    fin_twin_data = []
    for i, municipality in enumerate(municipalities):
        closest_index = distance_matrix[i].argmin()
        closest_municipality = df.loc[closest_index, 'Gemeinde']
        fin_twin_data.append({'Municipality': municipality, 'Financial Twin': closest_municipality})
    
    return pd.DataFrame(fin_twin_data)

def generate_relative_spends_df(df):
    df_relative = df.copy()
    df_relative = df_relative.loc[:, ~df_relative.columns.duplicated()]
    for col in df_relative.columns[8:]:
        df_relative[col] = pd.to_numeric(df_relative[col], errors='coerce').fillna(0).astype(float)
    
    spend_columns = df_relative.columns[7:]
    population_column = df_relative.columns[3]
    
    for col in spend_columns:
        df_relative[col] = df_relative[col] / df_relative[population_column]
        df_relative.rename(columns={col: f"{col}_relative"}, inplace=True)
    
    return df_relative

def find_relative_ratio_df(df_payouts_relative, df_payins_relative):
    df_spends_ratio = df_payouts_relative.copy()
    spend_columns = df_payouts_relative.columns[7:]
    
    for col in spend_columns:
        non_zero_mask = df_payouts_relative[col] != 0.0
        df_spends_ratio[col] = np.nan
        df_spends_ratio.loc[non_zero_mask, col] = (
            df_payins_relative.loc[non_zero_mask, col] / df_payouts_relative.loc[non_zero_mask, col]
        ).astype(float)
        df_spends_ratio.rename(columns={col: f"{col}_ratio"}, inplace=True)
    
    return df_spends_ratio

def find_merged_df(payins_cosine_df, payins_euc_df, payouts_cosine_df, payouts_euc_df):
    payins_cosine_df = payins_cosine_df.rename(columns={'Financial Twin': 'Payins Cosine Twin'})
    payins_euc_df = payins_euc_df.rename(columns={'Financial Twin': 'Payins Euclidean Twin'})
    payouts_cosine_df = payouts_cosine_df.rename(columns={'Financial Twin': 'Payouts Cosine Twin'})
    payouts_euc_df = payouts_euc_df.rename(columns={'Financial Twin': 'Payouts Euclidean Twin'})
    
    merged_df = payins_cosine_df.merge(payins_euc_df, on='Municipality')
    merged_df = merged_df.merge(payouts_cosine_df, on='Municipality')
    merged_df = merged_df.merge(payouts_euc_df, on='Municipality')
    
    return merged_df


def rename_last_14_columns(df):
    # Get the last 14 columns
    last_14_columns = df.columns[-14:]

    # Loop through the last 14 columns to rename them
    new_column_names = []
    for col in last_14_columns:
        # Extract the year from the current column name
        for year in range(2009, 2023):
            if str(year) in col:
                # Create the new name
                new_name = f'Overall Population {year}'
                new_column_names.append(new_name)
                break

    # Rename the columns
    df.rename(columns=dict(zip(last_14_columns, new_column_names)), inplace=True)
    return df

def populate_new_year_col(df):
  # Step 1: Filter out rows where 'AGS' is not NaN and not empty
  # Convert the 'AGS' column to string
  df['AGS'] = df['AGS'].astype(str)

  # Filter rows where 'AGS' is a 4-digit year or an 8-digit number
  df_valid = df[df['AGS'].str.match(r'^\d{4}$|^\d{8}$')]
  year_mask = df_valid['AGS'].str.contains(r'20\d{2}')

  # Step 2: Create a 'Year' column that will hold the corresponding year for each row
  df_valid['Year'] = None

  # Step 3: Populate the 'Year' column based on the 'AGS' values
  current_year = None
  valid_years = [str(year) for year in range(2009, 2023)]
  for index, row in df_valid.iterrows():
      if row['AGS'] in valid_years:
          current_year = df_valid.loc[index, 'AGS']  # Update the current year
      df_valid.loc[index, 'Year'] = current_year  # Set the year in the new 'Year' column

  # Step 4: Move the 'Year' column to the second position
  cols = df_valid.columns.tolist()
  cols.insert(1, cols.pop(cols.index('Year')))
  df_valid = df_valid[cols]

  return df_valid


def modify_payins_payouts_df(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Set the first row as header and remove it from the data
    df_copy.columns = df_copy.iloc[0]
    df_copy = df_copy[1:].reset_index(drop=True)  # Reset index after dropping the first row

    # Assuming populate_new_year_col is a defined function
    df_copy = populate_new_year_col(df_copy)

    # Specify column indices to keep
    columns_to_keep = list(range(4)) + list(range(115, 140))
    
    # Filter the DataFrame to include only the selected columns
    df_copy = df_copy.iloc[:, columns_to_keep]

    # Identify columns to keep based on the presence of non-numeric values
    col_to_keep = []
    for col_num in range(len(df_copy.columns)):
        if not df_copy.iloc[:, col_num].astype(str).str.contains('-').any():
            col_to_keep.append(col_num)

    # Keep only the identified columns
    df_copy = df_copy.iloc[:, col_to_keep]

    # Specify valid years as strings
    valid_years = [str(year) for year in range(2009, 2023)]
    
    # Remove rows where 'AGS' column contains any of the valid years
    df_final = df_copy[~df_copy['AGS'].astype(str).isin(valid_years)]

    return df_final


def generic_ops():

    df = pd.read_excel(population_data_path, sheet_name="2019 (2)")
    df_generic = df_generic = df.iloc[4:, :38].copy()
    df_generic.columns = df_generic.iloc[0]
    df_generic = df_generic.iloc[1:]
    df_generic.reset_index(drop=True, inplace=True)
    df_generic.columns = df_generic.iloc[4]
    df_generic = df_generic.iloc[5:].reset_index(drop=True)
    df_generic = rename_last_14_columns(df_generic)
    df_population_data = df_generic[['AGS_JuAmt','Bezeichnung', 'Overall Population 2009',
       'Overall Population 2010', 'Overall Population 2011',
       'Overall Population 2012', 'Overall Population 2013',
       'Overall Population 2014', 'Overall Population 2015',
       'Overall Population 2016', 'Overall Population 2017',
       'Overall Population 2018', 'Overall Population 2019',
       'Overall Population 2020', 'Overall Population 2021',
       'Overall Population 2022']]
    
    df_population_data.columns=['AGS_JuAmt', 'Bezeichnung1', 'Bezeichnung2', 'Overall Population 2009',
       'Overall Population 2010', 'Overall Population 2011',
       'Overall Population 2012', 'Overall Population 2013',
       'Overall Population 2014', 'Overall Population 2015',
       'Overall Population 2016', 'Overall Population 2017',
       'Overall Population 2018', 'Overall Population 2019',
       'Overall Population 2020', 'Overall Population 2021',
       'Overall Population 2022']
    
    df_pop_data_final = df_population_data[['AGS_JuAmt', 'Bezeichnung2', 'Overall Population 2009',
       'Overall Population 2010', 'Overall Population 2011',
       'Overall Population 2012', 'Overall Population 2013',
       'Overall Population 2014', 'Overall Population 2015',
       'Overall Population 2016', 'Overall Population 2017',
       'Overall Population 2018', 'Overall Population 2019',
       'Overall Population 2020', 'Overall Population 2021',
       'Overall Population 2022']]
    
    # Melt the DataFrame to long format
    df_pop_data_reshaped = pd.melt(df_pop_data_final, 
                   id_vars=['AGS_JuAmt', 'Bezeichnung2'], 
                   var_name='Year', 
                   value_name='Population')

   # Extract the year from the column name
    df_pop_data_reshaped['Year'] = df_pop_data_reshaped['Year'].str.extract(r'(\d{4})')

    # Prefix a '0' to all entries in the 'AGS_JuAmt' column
    df_pop_data_reshaped['AGS_JuAmt'] = '0' + df_pop_data_reshaped['AGS_JuAmt'].astype(str)

    #payouts and payins data transformation
    df_payouts = pd.read_excel(payouts_data_path,header=1)
    df_payouts_final = modify_payins_payouts_df(df_payouts)

    df_payins = pd.read_excel(payins_data_path,header=1)
    df_payins.to_excel('df_payins.xlsx')
    df_payins_final = modify_payins_payouts_df(df_payins)

    df_payins_final.to_excel('df_payins_final.xlsx')

    #merge with population data
    
    merged_payouts = pd.merge(df_pop_data_reshaped, df_payouts_final, left_on=['AGS_JuAmt', 'Year'], right_on=['AGS', 'Year'])
    merged_payins = pd.merge(df_pop_data_reshaped, df_payins_final, left_on=['AGS_JuAmt', 'Year'], right_on=['AGS', 'Year'])

    #generate relative spends df
    df_payouts_relative = generate_relative_spends_df(merged_payouts)
    df_payins_relative = generate_relative_spends_df(merged_payins)

    df_spends_ratio = find_relative_ratio_df(df_payouts_relative, df_payins_relative)

    return merged_payins, merged_payouts, df_payins_relative, df_payouts_relative, df_spends_ratio

def get_financial_twin(municipality, year, analysis_type, include_per_capita):
    merged_payins, merged_payouts, df_payins_relative, df_payouts_relative, df_spends_ratio = generic_ops()
    
    twins = {
        ('payins_cosine', 'No'): find_fin_twin(merged_payins, year, 'cosine', 'payins'),
        ('payins_euclidean', 'No'): find_fin_twin(merged_payins, year, 'euclidean', 'payins'),
        ('payins_cosine', 'Yes'): find_fin_twin(df_payins_relative, year, 'cosine', 'payins'),
        ('payins_euclidean', 'Yes'): find_fin_twin(df_payins_relative, year, 'euclidean', 'payins'),
        ('payouts_cosine', 'No'): find_fin_twin(merged_payouts, year, 'cosine', 'payouts'),
        ('payouts_euclidean', 'No'): find_fin_twin(merged_payouts, year, 'euclidean', 'payouts'),
        ('payouts_cosine', 'Yes'): find_fin_twin(df_payouts_relative, year, 'cosine', 'payouts'),
        ('payouts_euclidean', 'Yes'): find_fin_twin(df_payouts_relative, year, 'euclidean', 'payouts'),
    }
    
    fin_twin_df = twins.get((analysis_type, include_per_capita))
    
    merged_all_df = find_merged_df(
        twins[('payins_cosine', include_per_capita)],
        twins[('payins_euclidean', include_per_capita)],
        twins[('payouts_cosine', include_per_capita)],
        twins[('payouts_euclidean', include_per_capita)]
    )
    
    result = fin_twin_df[fin_twin_df['Municipality'] == municipality]
    
    return result.iloc[0]['Financial Twin'] if not result.empty else "No financial twin found", merged_all_df