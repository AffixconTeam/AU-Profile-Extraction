import streamlit as st
import numpy as np
import pandas as pd
import datetime
import gender_guesser.detector as gender
from kneed import KneeLocator 
d = gender.Detector()
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import ast 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



st.set_page_config(layout='wide')
# st.title(':red[DataZoo Profile Extraction]')
st.markdown("<h1 style='text-align: center; color: blue;'>Lookalike Profile Extraction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: green;'>Contact Plus lookalike Audience Count:</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: orange;'>Below is a summary of the lookalike audience counts available for purchase, based on the current database records for each profiles.</h6>", unsafe_allow_html=True)

#---------------------------------------------Functions --------------------------------------------------------------------
def calculate_match(row, profile):
    total_fields = len(profile)  -2 
    matches = 0
    
    for field in profile:
        if field == 'Profile' or field =='Cluster':
            continue
        if (profile[field] != '' and row[field]!='') and row[field] == profile[field]:
            matches += 1
    return (matches / total_fields) * 100

def multi_calculate_match(row, profile):
    total_fields = len(profile) - 2  
    matches = 0
    for field in profile:
        if field == 'Profile':
            continue

        if profile.get(field) and row.get(field):
            if isinstance(profile[field], str) and ',' in profile[field]:
                profile_values = [val.strip() for val in profile[field].split(', ')]
            else:
                profile_values = [profile[field]] 

            if str(row[field]) in profile_values:
                matches += 1
    return (matches / total_fields) * 100

def age(dob):
    born_year = pd.to_datetime(dob).date().year
    current_year = datetime.datetime.now().year
    age = current_year - born_year
    return age

def predict_gender(row):
    if pd.isna(row['GenderNew']) : 
        for name_column in ['FirstName', 'MiddleName', 'Surname']:
            name = row.get(name_column)
            if pd.notna(name) and len(name)>1: 
                predicted_gender = d.get_gender(name)
                if predicted_gender in ['male', 'female']:
                    return predicted_gender
    return row['GenderTemp']

# @st.cache_data
def preprocss(df):
    
    #------------------------------Age-------------------------------------------
    age_df = df[['Age',"Merged_Age_Range","Age_Merged","Age_S4614","Age_Range_S4614","DOB_Formatted","DOB_Merged"]]
    age_df['AgeNew'] = age_df['DOB_Formatted'].apply(age)
    age_ranges = {
    "0-4": (0, 4),
    "5-9": (5, 9),
    "10-14": (10, 14),
    "15-19": (15, 19),
    "20-24": (20, 24),
    "25-29": (25, 29),
    "30-34": (30, 34),
    "35-39": (35, 39),
    "40-44": (40, 44),
    "45-49": (45, 49),
    "50-54": (50, 54),
    "55-59": (55, 59),
    "60-64": (60, 64),
    "65-69": (65, 69),
    "70-74": (70, 74),
    "75-79": (75, 79),
    "80-84": (80, 84),
    "85-89": (85, 89),
    "90-94": (90, 94),
    "Over 95": (95, float('inf'))}
    def get_age_range(age):
        for label, (lower, upper) in age_ranges.items():
            if lower <= age <= upper:
                return label
        return "" 
    
    age_df.loc[
        (age_df['DOB_Formatted'].isna()) & (age_df['DOB_Merged'].notna()),
        'AgeNew'
    ] = age_df.loc[
        (age_df['DOB_Formatted'].isna()) & (age_df['DOB_Merged'].notna()),
        'DOB_Merged'
    ].apply(age)

    age_df['AgeRangeNew'] = age_df['AgeNew'].apply(get_age_range)

    age_df.loc[
        (age_df['AgeRangeNew'] == 'Unknown') & (age_df['Merged_Age_Range'].notna()),
        'AgeRangeNew'
    ] = age_df['Merged_Age_Range']
    age_df.loc[
        (age_df['AgeRangeNew'] == 'Unknown') & (age_df['Age_Range_S4614'].notna()),
        'AgeRangeNew'
    ] = age_df['Merged_Age_Range']
    

    
    #------------------------------Gender-------------------------------------------
    gender_df = df[["FirstName","MiddleName","Surname",'Gender', "GenderTemp"]]
    gender_df['GenderNew'] = np.nan
    gender_df.loc[
        ((gender_df['Gender'].notna()) & (gender_df['Gender'] != "_")),
        'GenderNew'
    ] = gender_df['GenderTemp']

    gender_df.loc[
        ((gender_df['GenderTemp'].notna()) & (gender_df['GenderTemp'] != "_")),
        'GenderNew'
    ] = gender_df['Gender']

    gender_df['GenderNew'] = gender_df.apply(predict_gender, axis=1)
    gender_df['GenderNew'] = gender_df['GenderNew'].fillna(gender_df['GenderTemp'])
    gender_df['GenderNew'] = gender_df['GenderNew'].replace({'F':'female', 'M':'male'})

    #------------------------------HomeOwner-------------------------------------------
    home_owner_df = df[["HomeOwner","Homeowner_Merged"]]
    gender_df['HomeOwnerNew'] = np.nan

    home_owner_df.loc[
        (home_owner_df['HomeOwner'].notna()),
        'HomeOwnerNew'
    ] = home_owner_df['HomeOwner']

    home_owner_df.loc[
        ((home_owner_df['Homeowner_Merged'].notna())),
        'HomeOwnerNew'
    ] = home_owner_df['Homeowner_Merged']

    home_owner_df['HomeOwnerNew'] = home_owner_df['HomeOwnerNew'].fillna("renters")
    home_owner_df['HomeOwnerNew'] = home_owner_df['HomeOwnerNew'].replace({'Y':"HomeOwner"})
    
    if is_toggled:
        analysis_df = pd.concat(
            [
                age_df[['AgeRangeNew']], 
                gender_df[['GenderNew']], 
                df[['IncomeRange_S4614']], 
                home_owner_df[['HomeOwnerNew']], 
                df[['DZ_Profile']],
                df[addition_features]
            ], 
            axis=1
        )
    else:
        analysis_df = pd.concat(
            [
                age_df[['AgeRangeNew']], 
                gender_df[['GenderNew']], 
                df[['IncomeRange_S4614']], 
                home_owner_df[['HomeOwnerNew']], 
                df[['DZ_Profile']]
            ], 
            axis=1
        )

    # analysis_df = analysis_df.fillna("")
    analysis_df = analysis_df.rename(columns={'AgeRangeNew':'Age','GenderNew':'Gender','IncomeRange_S4614':'Income', 'HomeOwnerNew':'HomeOwner', 'DZ_Profile':'Contact_Plus_Profiles'})

    return analysis_df

# @st.cache_data
def normalization(df, cols_encode):

    encoded_df = preprocss(df).fillna("").copy()
    LE=LabelEncoder()
    Sc = StandardScaler()
    # cols_encode = ['IncomeRange','GenderNew','AgeRangeNew', 'ProfileType', 'DirShr_Category', 'NewHomeOwner', "PropertyValueRange"]
    if is_toggled:
        for i in addition_features:

            # Label encode the column
            encoded_df[i] = LE.fit_transform(encoded_df[i].astype(str))  # Convert to string if needed
            
            # Standard scale the column
            encoded_df[i] = Sc.fit_transform(encoded_df[[i]].values).flatten()

    income_order = {'Unknown': 9, 'Under $20,799': 1, '$20,800 - $41,599': 2, '$41,600 - $64,999':3, '$65,000 - $77,999':4, '$78,000 - $103,999':5, '$104,000 - $155,999':6, '$156,000+':7}
    age_order = {
        '':1,'0-4':2,'5-9': 3,'10-14': 4,'15-19':5,'20-24': 6,'25-29': 7,'30-34': 8,'35-39': 9,'40-44': 10,'45-49': 11,'50-54': 12,'55-59': 13,'60-64': 14,'65-69': 15,
        '70-74': 16,'75-79': 17,'80-84': 18,'>84': 19, '85-89': 20,'90-94': 21,'Over 95':22}
    profile_category_order = {
    '':1,'A1': 2,'A2': 3,'A3': 4,'A4': 5,'A5': 6,'B1': 7,'B2': 8,'B3': 9,'B4':10,'B5': 11, 'B6':12, 'C1': 13,'C2': 14,'C3': 15,'C4': 16,'C5': 17,'D1': 18,'D2': 19,'D3': 20,'D4': 21,
    'E1': 22,'E2': 23,'E3': 24,'E4': 25,'E5': 26,'E6': 27,'F1': 28,'F2': 29}
    home_owner_order = {'':1,'renters':2,'HomeOwner':3}
    gender_order = {'male':3,'female':2,'':1}


    df_encoded= encoded_df.copy()

    df_encoded['Income'] = encoded_df['Income'].map(income_order)
    df_encoded['Age'] = encoded_df['Age'].map(age_order)
    df_encoded['Contact_Plus_Profiles'] = encoded_df['Contact_Plus_Profiles'].map(profile_category_order)
    df_encoded['Gender'] = encoded_df['Gender'].map(gender_order)
    df_encoded['HomeOwner'] = encoded_df['HomeOwner'].map(home_owner_order)

    df_encoded = df_encoded[cols_encode]

    # #Scaling
    # scaler = StandardScaler()
    # scaler.fit(df_encoded)
    # df_encoded = pd.DataFrame(scaler.transform(df_encoded[cols_encode]),columns= df_encoded[cols_encode].columns )
    # # print("All features are now scaled")
    
    return df_encoded

@st.cache_data
def Dimensional_Reduction(scaled_df):
    # scaled_df = normalization()
    pca = PCA(n_components=3)
    pca.fit(scaled_df)
    PCA_df = pd.DataFrame(pca.transform(scaled_df), columns=(["col1","col2", "col3"]))
    
    return PCA_df

def Threshold(column):
    test1 = breakdowns_all[breakdowns_all['Category']==column]
    # val = (test1[f'Group_{i}_Relative_Percentage'].tolist() i for i in range(optimal_clusters))
    # val = [test1[f'Group_{i}_Relative_Percentage'].tolist() for i in range(1, optimal_clusters + 1)]
    # Filter columns ending with '_Relative_Percentage'
    relative_cols = [col for col in test1.columns if col.endswith('_Relative_Percentage')]
    val = test1[relative_cols].values.tolist()  
    thresold = np.mean(val)+0.5*np.std(val)
    return thresold

def remove_empty_elements_from_string(lst_str):
    try:
        lst = ast.literal_eval(lst_str)
        # return str([item for item in lst if not item.startswith("empty_")])
        return str([item for item in lst if not "empty_" in item])

    except (ValueError, SyntaxError):
        return lst_str
#------------------------------------------------------ End of Functions --------------------------------------------------
        
# file = st.sidebar.file_uploader('Choose a File')
# if file is not None:
datazoo_profile_count = {
    "Profile": ['Group 1', 'Group 2', 'Group 3', 'Group 4','Total'],
    "Mailing": ['9,457', '808', '4,844', '6,245','18,777'],
    "Tele Marketing": ['7,582', '374', '4,581', '3,214','16,814'],
    "Mobile": ['722', '104', '303', '74','822'],
    "Phone": ['7,142', '175', '4,033', '3,083','15,379'],
    "Email": ['358', '18', '245', '57','613'],
}


datazoo_profile_df = pd.DataFrame(datazoo_profile_count)
datazoo_profile_df.set_index('Profile', inplace=True)
st.dataframe(datazoo_profile_df, width=2000)

if 1 != 3 :
    usecols = ['Age',"Merged_Age_Range","Age_Merged","Age_S4614","Age_Range_S4614","DOB_Formatted","DOB_Merged",
            "FirstName",	"MiddleName","Surname",'Gender', "GenderTemp","Income", "IncomeRange_S4614",
            "HomeOwner","Homeowner_Merged", 
            "DZ_Profile","DZ_Profile_Description","Suburb","State","Metro"]
    data = pd.read_excel("AU Sample Data.xlsx", usecols=usecols)[usecols]

    # data = pd.read_csv(file, encoding = 'latin1' )
    dilworth = data


#     #-------------------------------------------------------------------------------------------------------------------------
    st.sidebar.write(":green[These are the additional geo features that can analyze for profile extraction.]")
    is_toggled = st.sidebar.checkbox("Geo features Analysis", value=False)

    if is_toggled:
        addition_features = st.sidebar.multiselect("Select Geo Featues to Analyze:",["Suburb","State","Metro"], default=['Suburb'])
        cols_encode = ['Income','Gender','Age', 'Contact_Plus_Profiles', 'HomeOwner'] + (addition_features)
    else:
        cols_encode = ['Income','Gender','Age', 'Contact_Plus_Profiles', 'HomeOwner']
    #-------------------------------------------------------------------------------------------------------------------------
    # st.write(cols_encode)
    scaled_df = normalization(dilworth,cols_encode=cols_encode)
    # st.write("scaled_df",scaled_df)
    PCA_df = Dimensional_Reduction(scaled_df)
    
    from matplotlib import colors
    cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
    corrmat= scaled_df.corr()
    # plt.figure(figsize=(20, 20))
    # heatmap = sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)

    # # Ensure proper handling with Streamlit
    # st.sidebar.pyplot(plt)
    
    Elbow_M = KElbowVisualizer(KMeans(random_state=42), k=10)
    Elbow_M.fit(Dimensional_Reduction(scaled_df))
    Elbow_M.ax.set_title("Affix Method for Optimal Groups")
    Elbow_M.ax.set_xlabel("No of Groups")
    # col1,col2,col3 = st.columns((3))

    # with col11:
    #     with st.expander(":red[Show Feature Correlations]"):
    #         fig1 = plt.figure(figsize=(10, 10))
            
    #         # Create the heatmap with larger annotation text
    #         sns.heatmap(corrmat, annot=True, cmap=cmap, center=0,
    #                     annot_kws={"size": 15})  # Set annotation font size
            
    #         # Adjust the size of the axis labels
    #         plt.xticks(fontsize=30)  # X-axis label font size
    #         plt.yticks(fontsize=30)  # Y-axis label font size
            
    #         # Show the plot
    #         st.pyplot(fig1)
    # with col1:
    #     with st.expander(":red[Show Optimal Group Selection]"):
    #         fig2 = Elbow_M.ax.figure
    #         st.pyplot(fig2)
        
    inertia_list = []
    cluster_range = range(2, 10)  

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(PCA_df)
        
        inertia_list.append(kmeans.inertia_)

    kl = KneeLocator(cluster_range, inertia_list, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow
    # st.write(f"The optimal number of clusters : {optimal_clusters}")


    AC = AgglomerativeClustering(n_clusters=optimal_clusters)
    yhat_AC = AC.fit_predict(PCA_df)
    PCA_df["Clusters"] = yhat_AC+1
    dilworth_preprocessed = preprocss(dilworth)
    dilworth_preprocessed["Clusters"] = yhat_AC+1

    cluster_counts = dilworth_preprocessed['Clusters'].value_counts()

    # Create a mapping from the old cluster labels to new labels based on the rank of counts
    cluster_mapping = {old_label: idx + 1 for idx, old_label in enumerate(cluster_counts.index)}

    # Map the clusters in dilworth_preprocessed to the new labels
    dilworth_preprocessed['Clusters'] = dilworth_preprocessed['Clusters'].map(cluster_mapping)
    PCA_df['Clusters'] = PCA_df['Clusters'].map(cluster_mapping)


    # Check the updated Clusters column
    # st.write(dilworth_preprocessed['Clusters'].value_counts())

    fig = px.scatter_3d(
        PCA_df, 
        x="col1", 
        y="col2", 
        z="col3", 
        color='Clusters', 
        # title="All Groups as clusters"
    )
    fig.update_layout(
    coloraxis_colorbar_title="Groups" 
    )   

    # with col2:
    #     with st.expander(":red[Show 3d Group Separation]"):
    #         st.plotly_chart(fig)

    # with col3:
    #     pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     pl = sns.countplot(x=dilworth_preprocessed["Clusters"], palette=pal, ax=ax)
    #     ax.set_xlabel('Groups')
    #     # pl.set_title("Distribution Of The Clusters")
    #     total = len(dilworth_preprocessed["Clusters"])
    #     for patch in ax.patches:
    #         height = patch.get_height()
    #         percentage = f"{(height / total) * 100:.1f}%"
    #         ax.annotate(percentage, (patch.get_x() + patch.get_width() / 2, height),
    #                     ha="center", va="bottom", fontsize=10)

    #     with st.expander(":red[Show Distribution of Groups]"):
    #         st.pyplot(fig)


    # ----------------------- Sample Breakdown ---------------------------------------
    if is_toggled:
        cols_del = ['Income','Gender','Age', 'Contact_Plus_Profiles', 'HomeOwner'] + addition_features
    else:
        cols_del = ['Income','Gender','Age', 'Contact_Plus_Profiles', 'HomeOwner']
    # st.write(cols_del)
    dfs = []
    for col in cols_del:
        count_df = pd.DataFrame(dilworth_preprocessed[col].value_counts())
        percent_df = pd.DataFrame(dilworth_preprocessed[col].value_counts(normalize=True) * 100)
        combined_df = pd.concat([count_df, percent_df], axis=1)
        combined_df['Category'] = col
        combined_df.columns = ['Count', 'Percentage','Category']
        combined_df.index = combined_df.index.map(lambda x: f"empty_{col}" if x == "" else x)
        if is_toggled and len(addition_features) > 1:
            if col in ['Suburb', 'Ward', 'City']:
                combined_df.index = combined_df.index.map(lambda x: f"{col}_{x}")

        dfs.append(combined_df)

    final_df = pd.concat(dfs, axis=0)
    final_df.columns = ['Sample_Count', 'Sample_Percentage','Category']

    # st.write(final_df)
    # ----------------------- Sample with all groups count Breakdown ---------------------------------------
    dfs_all = []
    for cluster in sorted(dilworth_preprocessed.Clusters.unique().tolist()):
        dfs = []
        profile_df = dilworth_preprocessed[dilworth_preprocessed['Clusters'] == cluster][cols_del]

        for col in cols_del:
            count_df = pd.DataFrame(profile_df[col].value_counts())
            count_df.columns = [f'Group_{cluster}_Count']
            count_df.index = count_df.index.map(lambda x: f"empty_{col}" if x == "" else x)
            if is_toggled and len(addition_features) > 1:
                if col in ["Suburb","State","Metro"]:
                    count_df.index = count_df.index.map(lambda x: f"{col}_{x}")
            dfs.append(count_df)
        profile_df_values = pd.concat(dfs, axis=0)
        profile_df_values = profile_df_values.reindex(final_df.index, fill_value=0)
        dfs_all.append(profile_df_values)
        
    dfs_all = pd.concat(dfs_all,axis=1)
    # st.write(dfs_all)

    breakdowns_all = pd.concat([final_df,dfs_all],axis=1)
    for i in range(1, optimal_clusters+1):
        breakdowns_all[f"Group_{i}_Percentage"] = (breakdowns_all[f"Group_{i}_Count"] / breakdowns_all['Sample_Count']) * 100
        breakdowns_all[f"Group_{i}_Relative_Percentage"] = (breakdowns_all[f"Group_{i}_Percentage"] * breakdowns_all['Sample_Percentage']) /100
    
    # st.write(breakdowns_all)

    # test1 = breakdowns_all[breakdowns_all['Category']=='ProfileType']
    # # val = (test1[f'Group_{i}_Relative_Percentage'].tolist() i for i in range(optimal_clusters))
    # val = [test1[f'Group_{i}_Relative_Percentage'].tolist() for i in range(1, optimal_clusters + 1)]
    profile_thresold = Threshold('Contact_Plus_Profiles')
    # st.write(profile_thresold)

    # -----------------------------------------------Profile Type extraction --------------------------------------
    test = breakdowns_all[breakdowns_all['Category'] == 'Contact_Plus_Profiles']
    relaive_cols = [f'Group_{i}_Relative_Percentage' for i in range(1,optimal_clusters+1) ]
    data = test[relaive_cols][test[relaive_cols]>profile_thresold].dropna(how='all').fillna(0).idxmax(axis=1).sort_values().to_dict()

    group_dict = {}
    for value, group in data.items():
        split_group = group.split('_', 2)
        group_prefix = '_'.join(split_group[:2])
        if group not in group_dict:
            group_dict[group] = [] 
        group_dict[group].append(value)  
    new_dict = {}

    for key, value in group_dict.items():
        new_key = '_'.join(key.split('_', 2)[:2])  
        new_dict[new_key] = value

    # st.write(new_dict)

    income_threshold = Threshold('Income')
    # st.write('profile_threshold previous', profile_thresold)

# ------------------------------------------- select other demog --------------------
    # results = {}
    # for i in range(1, optimal_clusters + 1):
    #     column = f"Group_{i}_Relative_Percentage"
        
    #     # Initialize the dictionary for the current group
    #     results[f"Group_{i}"] = {}

    #     filtered = breakdowns_all.copy()

    #     for col in cols_del:
    #         filtered_range = filtered[(filtered["Category"] == col) & (filtered[column] > Threshold(col))]
            
    #         # Check if filtered_range is not empty
    #         if not filtered_range.empty:
    #             results[f"Group_{i}"][col] = str(filtered_range.index.tolist())

    results = {}
    for i in range(1, optimal_clusters + 1):
        column = f"Group_{i}_Relative_Percentage"
        filtered = breakdowns_all.copy()  
        # st.write(filtered)
        # st.write(Threshold('Income'))
        # st.write(Threshold('Contact_Plus_Profiles'))


        filtered = filtered[
            ((filtered["Category"] == "Income") & (filtered[column] > Threshold('Income'))) | 
            ((filtered["Category"] != "Income") & (filtered[column] > Threshold('Contact_Plus_Profiles')))]
        filtered = filtered.sort_values(by=column, ascending=False)
        grouped = filtered.groupby("Category").apply(lambda x: str(x.index.tolist()))
        results[f"Group_{i}"] = grouped.to_dict()
             
    # st.write(results)

    profiles_selected = pd.DataFrame(results).fillna("")
    # st.dataframe(profiles_selected,width=1500)

    for i in range(1, optimal_clusters + 1):
        profiles_selected.loc['Contact_Plus_Profiles',f'Group_{i}'] = ''
    for i,j in new_dict.items():
        profiles_selected.loc['Contact_Plus_Profiles',i] = str(j)

    profiles_selected = profiles_selected.applymap(remove_empty_elements_from_string)
    profiles_selected = profiles_selected.apply(lambda x: x.replace(r"[\[\]']", '', regex=True))

    # st.dataframe(profiles_selected,width=1500)
    cluster_percentages = np.ceil(dilworth_preprocessed['Clusters'].value_counts(normalize=True) * 100)
    cluster_percentages_df = cluster_percentages.to_frame().T  
    cluster_percentages_df.index = ["Group_percentage"]    
    cluster_percentages_df.columns = [f'Group_{i}' for i in cluster_percentages.index]
    profiles_selected = pd.concat([profiles_selected, cluster_percentages_df], axis=0).T
    # profiles_selected = profiles_selected.T.sort_values('Cluster_percentage', ascending=False)
    # profiles_selected.index = [f'Group_{i}' for i in range(1,len(cluster_percentages)+1)]
    # st.write(profiles_selected.sort_values('Cluster_percentage', ascending=False))

    profiles_selected = profiles_selected.T

    for i in range(1, optimal_clusters + 1):
        ranges = profiles_selected.loc['Age', f'Group_{i}']
        ranges = ", ".join([r for r in ranges.split(", ") if r != 'Over 95'])
        
        if ranges != "": 

            value_ranges = ranges.split(', ')
            values = [int(r.split('-')[0]) for r in value_ranges]
            lowest_value = min(values)
            highest_value = max([int(r.split('-')[1]) for r in value_ranges])
            age_range = (f'{lowest_value} - {highest_value}')
            profiles_selected.loc['Age', f'Group_{i}'] = age_range
            
        else:
            continue
    
    import re
    # def replace_with_k_format(ranges):
    #     # Regex to find ranges like '70,001 - 100,000'
    #     return re.sub(r'(\d{1,3}(?:,\d{3})*) - (\d{1,3}(?:,\d{3})*)', lambda m: f"{int(m.group(1).replace(',', '')) // 1000}K - {int(m.group(2).replace(',', '')) // 1000}K", ranges)

    def replace_with_k_format(ranges):
        """
        Converts income ranges like '70,001 - 100,000' or '156,000+' 
        into a format with 'K' for thousands, e.g., '70K - 100K' or '156K+'.
        """
        # Regex to find patterns in ranges
        def convert_match(match):
            if match.group(2):  # Range case: '70,001 - 100,000'
                start = int(match.group(1).replace(',', '')) // 1000
                end = int(match.group(2).replace(',', '')) // 1000
                return f"{start}K - {end}K"
            else:  # Single case with '+': '156,000+'
                value = int(match.group(1).replace(',', '')) // 1000
                return f"{value}K"

        return re.sub(
            r'(\d{1,3}(?:,\d{3})*)(?: - (\d{1,3}(?:,\d{3})*))?',  # Regex for range or single values
            convert_match,
            ranges
        )
    for i in range(1, optimal_clusters + 1):
        # Extract the ranges for the current group
        income_ranges = profiles_selected.loc['Income', f'Group_{i}'].replace("$", '')  # Remove '$'
        modified_ranges = replace_with_k_format(income_ranges)  # Format with 'K'
        profiles_selected.loc['Income', f'Group_{i}'] = modified_ranges  # Update DataFrame


    def sort_income_ranges(value):
        if value and isinstance(value, str):  # Check for non-empty and string values
            ranges = value.split(", ")
            sorted_ranges = sorted(ranges, key=lambda x: int(''.join(filter(str.isdigit, x.split("K")[0].replace("M", "000").replace("k", "000").replace(" ", "")))))
            return ", ".join(sorted_ranges)
        return value  # Return the original value if it's empty or invalid

    # Function to sort PropertyValueRange values
    def sort_property_value_ranges(value):
        if value and isinstance(value, str):  # Check for non-empty and string values
            ranges = value.split(", ")
            sorted_ranges = sorted(ranges, key=lambda x: int(''.join(filter(str.isdigit, x.split("-")[0].replace("M", "000").replace("k", "000").replace("+", "000").replace(" ", "")))))
            return ", ".join(sorted_ranges)
        return value  # Return the original value if it's empty or invalid

    # Apply sorting to both "IncomeRange" and "PropertyValueRange"
    for idx in ['Income']:
        if idx in profiles_selected.index:
            for col in profiles_selected.columns:
                # Apply sorting only for these specific rows
                if idx == 'Income':
                    profiles_selected.loc[idx, col] = sort_income_ranges(profiles_selected.loc[idx, col])

    cols_rename_new = {'Contact_Plus_Profiles':'Contact Plus Profiles','Income':'Income', 'HomeOwner':'HomeOwner', 'Gender':'Gender', 'Age':'Age'}   
    if is_toggled:
        for feature in addition_features:
            cols_rename_new[feature] = feature  # You can customize the value as needed
    # if "ProfileType" in profiles_selected.index:
    #     cols_to_drop = profiles_selected.columns[profiles_selected.loc["ProfileType"] == ""]
    #     profiles_selected.drop(columns=cols_to_drop, inplace=True

    # st.write(cols_rename_new)
    # st.write(profiles_selected)
# ----------------------commented block   ----------------------------------------------

    # st.dataframe(profiles_selected.rename(cols_rename_new),width=1700)

    # st.markdown(
    #     "<h4 style='text-align: center; color: green;'>Grouped Detailed Breakdowns:</h4>",
    #     unsafe_allow_html=True
    # )
    # # test2 = breakdowns_all[['Category'] + [f'Group_{i}_Relative_Percentage' for i in range(1,optimal_clusters+1)]]
    # # test2_df = test2[test2['Category'] == 'ProfileType']
    # # st.write(test2[test2['Category'] == 'ProfileType'])
    # # st.write(test2_df[test2_df.drop(columns='Category',axis=1)>profile_thresold])
    # test2 = breakdowns_all[['Category']+[f'Group_{i}_Relative_Percentage' for i in range(1,optimal_clusters+1)]]
    # groups = [f'Group_{i}_Relative_Percentage' for i in range(1, optimal_clusters+1)]
    # for group in groups:
    #     group_data = []  # Collect DataFrames for each group

    #     for col in cols_del:
    #         # Filter and compute threshold
    #         demog = test2[test2['Category'] == col][[group]].copy()
    #         demog['Threshold Value'] = np.where(
    #             col != "IncomeRange",
    #             demog[group] - Threshold('ProfileType'),
    #             demog[group] - Threshold('IncomeRange'))
    #         demog = demog.sort_values(by=group, ascending=False)
    #         demog = demog[~demog.index.str.startswith("empty_")]
    #         demog['Category'] = col  # Add category column for clarity
    #         group_data.append(demog)

    #     # Concatenate all DataFrames for the group
    #     concatenated_df = pd.concat(group_data, axis=0)

    #     # Apply styling to the concatenated DataFrame
    #     styled_df = concatenated_df.style.bar(
    #         subset=['Threshold Value'],  
    #         align='mid',
    #         color=['#d65f5f', '#5fba7d']
    #     )

    #     # Convert to HTML and display as one table
    #     full_html = styled_df.to_html()
    #     with st.expander(f":red[**View {' '.join(group.split('_')[:2])}**]"):
    #         col11, col12 = st.columns((0.6,0.4))
    #         with col11:
    #             st.markdown(full_html, unsafe_allow_html=True)
    #         with col12:
    #             st.write("""
    #                 :green[**Threshold graphs are based on the following criteria**]
    #                 - :orange[**Weight Significance:** Percentages of each group attribute calculated to indicate their significance as weight.]
    #                 - :orange[**Distributed Percentages:** Percentages distributed across attributes among the groups.]
    #                 - :orange[**Relative Percentage:** Represents each attribute's significance relative to the entire sample data.]
    #                 - :orange[**Threshold Value:** Derived from the distribution using mean and standard deviation.]
    #                 - :orange[**Positive Difference Selection:** Attributes with positive differences between actual and threshold values are selected.]
    #                 - :orange[**Sorting:** Attributes sorted in descending order of the threshold value, visualized from :green[green] to :red[red.]]

    #                      """)




    #------------------------------------------------------------Profile Selection ----------------------------
    ## for col in breakdowns_all.columns:
    # groups = sorted(st.multiselect("Select Groups", profiles_selected.columns, default=profiles_selected.columns))
    # selected_columns = ['Sample_Count', 'Sample_Percentage','Category'] + [col for col in breakdowns_all.columns if any(group in col for group in groups)]
    if "Contact_Plus_Profiles" in profiles_selected.index:
        groups = profiles_selected.columns[profiles_selected.loc["Contact_Plus_Profiles"] != ""]
        selected_columns = ['Sample_Count', 'Sample_Percentage','Category'] + [col for col in breakdowns_all.columns if any(group in col for group in groups)]


    breakdowns_all = breakdowns_all[selected_columns]

    # st.dataframe(breakdowns_all,width=1500)
    # st.write(breakdowns_all.columns.tolist())
    # groups_rename = [f'Group_{i+1}' for i in range(len(selected_columns))]

    # selected_column_indices = [i for i, col in enumerate(breakdowns_all.columns) if col in selected_columns]
    # new_column_names = [f"Group_{i+1}_" for i in range(len(selected_columns))]
    # for idx, new_col in zip(selected_column_indices, new_column_names):
    #     breakdowns_all.columns.values[idx] = new_col

    

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------Profile Type extraction --------------------------------------
    test = breakdowns_all[breakdowns_all['Category'] == 'Contact_Plus_Profiles']
    # relaive_cols = [f'Group_{i}_Relative_Percentage' for i in range(1,len(selected_columns)+1) ]
    # relative_cols = [i for i in breakdowns_all.columns.str.strip() if i.endswith('_Relative_Percentage')]
    relaive_cols = [col for col in breakdowns_all.columns if col.endswith('_Relative_Percentage')]
    # st.write(test.columns)
    profile_thresold = Threshold('Contact_Plus_Profiles')
    # st.write('profile_threshold new', profile_thresold)
    
    data = test[relaive_cols][test[relaive_cols]>profile_thresold].dropna(how='all').fillna(0).idxmax(axis=1).sort_values().to_dict()
    group_dict = {}
    for value, group in data.items():
        split_group = group.split('_', 2)
        group_prefix = '_'.join(split_group[:2])
        if group not in group_dict:
            group_dict[group] = [] 
        group_dict[group].append(value)  
    new_dict = {}

    for key, value in group_dict.items():
        new_key = '_'.join(key.split('_', 2)[:2])  + '_Relative_Percentage'
        new_dict[new_key] = value

    # st.write(new_dict)

    income_threshold = Threshold('Income')
    # st.write('income_threshold_new',income_threshold)

    results = {}
    for column in relaive_cols:
        # column = f"Group_{i}_Relative_Percentage"
        filtered = breakdowns_all.copy()  
        filtered = filtered[
            ((filtered["Category"] == "Income") & (filtered[column] > Threshold('Income'))) | 
            ((filtered["Category"] != "Income") & (filtered[column] > Threshold('Contact_Plus_Profiles')))]
        filtered = filtered.sort_values(by=column, ascending=False)
        grouped = filtered.groupby("Category").apply(lambda x: str(x.index.tolist()))
        results[column] = grouped.to_dict()
             

    profiles_selected = pd.DataFrame(results).fillna("")
    for i in relaive_cols:
        profiles_selected.loc['Contact_Plus_Profiles',i] = ''
    for i,j in new_dict.items():
        profiles_selected.loc['Contact_Plus_Profiles',i] = str(j)

    profiles_selected = profiles_selected.applymap(remove_empty_elements_from_string)
    profiles_selected = profiles_selected.apply(lambda x: x.replace(r"[\[\]']", '', regex=True))

    new_profiles_selected = profiles_selected[relaive_cols]
    rename_cols = [f'Group_{i}' for i in range(1,len(results)+1)]
    new_profiles_selected.columns = rename_cols
    new_cluster_percentages_df = cluster_percentages_df[groups]
    new_cluster_percentages_df.columns = rename_cols
    new_profiles_selected = pd.concat([new_profiles_selected, new_cluster_percentages_df], axis=0)
    # st.dataframe(new_profiles_selected,width=1500)
    new_profiles_selected = new_profiles_selected.T.sort_values('Group_percentage',ascending=False)
    new_profiles_selected.index = rename_cols
    new_profiles_selected = new_profiles_selected.T
    # for col in breakdowns_all.columns:\

    for i in range(1, len(groups)+1):
        ranges = new_profiles_selected.loc['Age', f'Group_{i}']
        ranges = ", ".join([r for r in ranges.split(", ") if r != 'Over 95'])

        
        if ranges != "": 
            value_ranges = ranges.split(', ')
            values = [int(r.split('-')[0]) for r in value_ranges]
            lowest_value = min(values)
            highest_value = max([int(r.split('-')[1]) for r in value_ranges])
            age_range = (f'{lowest_value} - {highest_value}')
            new_profiles_selected.loc['Age', f'Group_{i}'] = age_range
            
        else:
            continue
    
    import re
    def replace_with_k_format(ranges):
        """
        Converts income ranges like '70,001 - 100,000' or '156,000+' 
        into a format with 'K' for thousands, e.g., '70K - 100K' or '156K+'.
        """
        # Regex to find patterns in ranges
        def convert_match(match):
            if match.group(2):  # Range case: '70,001 - 100,000'
                start = int(match.group(1).replace(',', '')) // 1000
                end = int(match.group(2).replace(',', '')) // 1000
                return f"{start}K - {end}K"
            else:  # Single case with '+': '156,000+'
                value = int(match.group(1).replace(',', '')) // 1000
                return f"{value}K"

        return re.sub(
            r'(\d{1,3}(?:,\d{3})*)(?: - (\d{1,3}(?:,\d{3})*))?',  # Regex for range or single values
            convert_match,
            ranges
        )
    for i in range(1, len(groups)+1):
        # Extract the ranges for the current group
        income_ranges = new_profiles_selected.loc['Income', f'Group_{i}']
        modified_ranges = replace_with_k_format(income_ranges)
        new_profiles_selected.loc['Income', f'Group_{i}'] = modified_ranges

    for idx in ['Income']:
        if idx in new_profiles_selected.index:
            for col in new_profiles_selected.columns:
                # Apply sorting only for these specific rows
                if idx == 'Income':
                    new_profiles_selected.loc[idx, col] = sort_income_ranges(new_profiles_selected.loc[idx, col])

    new_profiles_selected = new_profiles_selected.rename(cols_rename_new)
    # st.write(new_profiles_selected)
    if is_toggled:
        # st.write(new_profiles_selected.index.tolist())

        if addition_features in new_profiles_selected.index.tolist():
            for idx in addition_features:
                new_profiles_selected.loc[idx] = new_profiles_selected.loc[idx].apply(lambda x: x.replace(f"{idx}_", "") if f"{idx}_" in x else x)
    # cols_rename_new = {'ProfileType':'Contact Plus Profiles','IncomeRange':'Income', 'DirShr_Category':'Company Office', 'NewHomeOwner':'HomeOwner', 'GenderNew':'Gender', 'Age':'Age', 'PropertyValueRange':'PropertyValue'}   
    if is_toggled and addition_features in new_profiles_selected.index.tolist():
        new_index_order = ['Contact Plus Profiles','Age','Gender','Income','HomeOwner'] + addition_features + ['Group_percentage']
    else:
        new_index_order = ['Contact Plus Profiles','Age','Gender','Income','HomeOwner','Group_percentage']
    st.markdown(
        "<h4 style='text-align: center; color: green;'>Multivariate Profiles Leading Characteristics:</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h6 style='text-align: center; color: orange;'>These groups are formed based on similar characteristics and highlight the top demographics within each group, providing insights into the defining attributes that distinguish each group.</h6>",
        unsafe_allow_html=True
    )

    profiles_With_description  = pd.read_excel("AU Sample Data.xlsx", usecols=["DZ_Profile","DZ_Profile_Description"])
    profiles_With_description.columns = ['Contact_Plus_Profiles','ProfileDescription']
    profiles_With_description['Profile_with_description'] = profiles_With_description['Contact_Plus_Profiles']+ " - "+ profiles_With_description['ProfileDescription']
    profiles_With_description_list = profiles_With_description['Profile_with_description'].dropna().drop_duplicates().tolist()
    new_profiles_selected.columns = new_profiles_selected.columns.str.replace('_', ' ')
    # profile_dict = {desc.split(' - ')[0]: desc for desc in profiles_With_description_list}

    # def replace_profiles(cell_value):
    #     # Split the string into codes, map to descriptions, and rejoin
    #     return ", ".join(profile_dict.get(code.strip(), code.strip()) for code in cell_value.split(", "))

    # # Apply the replacement only to the "Contact Plus Profiles" row
    # new_profiles_selected.loc["Contact Plus Profiles"] = new_profiles_selected.loc["Contact Plus Profiles"].apply(replace_profiles)
    st.dataframe(new_profiles_selected.reindex(new_index_order), width=2000)
    st.markdown(
        "<h4 style='text-align: center; color: green;'>Grouped Detailed Breakdowns:</h4>",
        unsafe_allow_html=True
    )
    # st.write(breakdowns_all.filter(like='_Relative_Percentage').columns)
    test2 = breakdowns_all[['Category'] + list(breakdowns_all.filter(like='_Relative_Percentage').columns)]
    groups = [f'Group_{i}_Relative_Percentage' for i in range(1,len(groups)+1)]
    test2.columns = ['Category'] + groups

    # st.write(test2.columns)

    # st.write(profiles_With_description_list)
    for group in groups:
        # group_data = []  # Collect DataFrames for each group

        with st.expander(f":red[**View {' '.join(group.split('_')[:2])}**]"):
            st.markdown("""
                <div style="text-align: center;">
                    <p style="color: green;"><strong>Threshold graphs are based on the following criteria</strong></p>
                    <ul style="list-style-type: none; padding-left: 0;">
                        <li style="color: orange;"><strong>Percentage:</strong> Represents each attribute's significance to the entire sample data.</li>
                        <li style="color: orange;"><strong>Index Value:</strong> Derived from the distribution based on statistical analysis.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            cols = st.columns((3))
            for i,col in enumerate(cols_del):
                # Filter and compute threshold
                demog = test2[test2['Category'] == col][[group]].copy()
                demog['Threshold Value'] = np.where(
                    col != "Income",
                    demog[group] - Threshold('Contact_Plus_Profiles'),
                    demog[group] - Threshold('Income'))
                demog = demog.sort_values(by=group, ascending=False)
                demog = demog[~demog.index.str.startswith("empty_")]
                # demog = demog[~demog.index.to_series().str.contains("empty_")]
                # demog['Category'] = col  # Add category column for clarity
                demog.columns = ['Percentage', 'Threshold Value']
                # demog.index.name = cols_rename_new[col]
                demog = demog[['Percentage','Threshold Value']]
                demog['Propotional Percentage'] = (100/(np.where(col != "Income",Threshold('Contact_Plus_Profiles'),demog["Percentage"] - Threshold('Income'))))*demog["Percentage"]
                demog['Index'] = np.ceil(demog['Propotional Percentage'] - 100).astype(int)
                demog = demog.drop(columns=['Percentage', 'Threshold Value'],axis=1)
                demog.columns = ['Percentage', 'Index']

                if col == 'Contact_Plus_Profiles':
                    profile_dict = {desc.split(' - ')[0]: desc for desc in profiles_With_description_list}
                    demog.index = demog.index.map(lambda idx: profile_dict.get(idx, idx))
                # group_data.append(demog)
                styled_df = demog.style.bar(
                    subset=['Index'],  
                    align='mid',
                    color=['#d65f5f', '#5fba7d']
                )

            # Convert to HTML and display as one table
                full_html = styled_df.to_html()
                column_index = i % 3  # This ensures the table will be displayed in one of the 3 columns
                with cols[column_index]:  # Select the column based on the index
                    st.write(f":red[**{cols_rename_new[col]} Analysis**]")
                    st.markdown(f"<div>{full_html}</div>", unsafe_allow_html=True)



        # Concatenate all DataFrames for the group
        # concatenated_df = pd.concat(group_data, axis=0)
        # concatenated_df.index = concatenated_df.index.map(
        #     lambda idx: idx.replace(f"{concatenated_df.loc[idx, 'Category']}_", "")
        #     if concatenated_df.loc[idx, "Category"] in idx else idx
        # )


        # concatenated_df.reset_index(inplace=True)
        # concatenated_df.rename(columns={"index": "Feature"}, inplace=True)
        # concatenated_df['Category'] = concatenated_df['Category'].replace(cols_rename_new)
        # st.write(concatenated_df)

        # Apply styling to the concatenated DataFrame
        # styled_df = concatenated_df.style.bar(
        #     subset=['Threshold Value'],  
        #     align='mid',
        #     color=['#d65f5f', '#5fba7d']
        # )

        # # Convert to HTML and display as one table
        # full_html = styled_df.to_html()
        # with st.expander(f":red[**View {' '.join(group.split('_')[:2])}**]"):
        #     col21, col22 = st.columns((0.6,0.4))
        #     with col21:
        #         st.markdown(full_html, unsafe_allow_html=True)
        #     with col22:
                # st.write("""
                #     :green[**Threshold graphs are based on the following criteria**]
                #     - :orange[**Relative Percentage:** Represents each attribute's significance relative to the entire sample data.]
                #     - :orange[**Threshold Value:** Derived from the distribution using mean and standard deviation.]
                #          """)
    
    st.markdown("<h4 style='text-align: center; color: green;'>Appendixes</h4>", unsafe_allow_html=True)
    # st.markdown("<h6 style='text-align: center; color: orange;'>A group represents similar characteristics across selected features.</h6>", unsafe_allow_html=True)

    with st.expander(":red[View all Appendixes]"):
        col1,col2,col3 = st.columns((3))

        with col3:
            st.write(':orange[**Optimal Group Selection:** Groups are formed by identifying shared similar characteristics within the sample data]')
            fig2 = Elbow_M.ax.figure
            st.pyplot(fig2)
  
        with col1:
                st.write(':orange[**3D Group Separation:** Groups are visually separated in a 3D space, showcasing distinct patterns and shared similarities among different sets of data.]')
            # with st.expander(":red[Show 3d Group Separation]"):
                st.plotly_chart(fig)
        with col2:
            st.write(':orange[**Distributions of Groups:** This represents how different groups are spread out, highlighting their size and percentage]')
            cluster_counts = dilworth_preprocessed["Clusters"].value_counts().reset_index()
            cluster_counts.columns = ["Clusters", "Count"]

            # Calculate percentages
            cluster_counts["Percentage"] = (cluster_counts["Count"] / cluster_counts["Count"].sum()) * 100

            # Create the bar chart
            fig = px.bar(
                cluster_counts,
                x="Clusters",
                y="Count",
                text="Percentage",
                color="Clusters",
                color_discrete_sequence=["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"],
                labels={"Clusters": "Groups", "Count": "Count"},
            )

            # Customize text and layout
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                # title="Distribution of Groups",
                xaxis_title="Groups",
                yaxis_title="Count",
                showlegend=False,
                template="simple_white",
            )

            # Show the plot in Streamlit
            st.plotly_chart(fig)


    