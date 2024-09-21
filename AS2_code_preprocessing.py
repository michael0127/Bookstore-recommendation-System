import pandas as pd
import nltk

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Load datasets
ratings = pd.read_csv("BX-Ratings.csv")
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv('BX-Users.csv')

# Preprocessing Age: drop the 'User-Age' column due to too many missing values
users.drop(columns=['User-Age'], inplace = True)

# Preprocessing User-City: drop the 'User-City' column
users.drop(columns=['User-City'], inplace = True)

# Clean string data by stripping extra quotation marks and spaces
users = users.applymap(lambda x: x.strip(' "').strip() if isinstance(x, str) else x)

# Replace empty and placeholder strings in 'User-State' and 'User-Country' with None
users.replace(["", "N/A", "n/a", "none", "-"], None, inplace=True)

# Remove records with missing 'User-State' or 'User-Country'
users = users.dropna(subset=['User-State', 'User-Country'])

# Filter users to only include those in the USA
users_in_usa = users[users["User-Country"] == 'usa']

# Set of valid US states
US_STATES = set(["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"])

# Convert US states to lowercase for comparison
lowercase_US_STATES = set(map(str.lower, US_STATES))

# Identify and correct invalid state entries
invalid_states = users_in_usa[~users_in_usa["User-State"].isin(lowercase_US_STATES)]
users_in_usa.loc[users_in_usa["User-State"] == "dc", "User-State"] = "district of columbia"
users_in_usa = users_in_usa[users_in_usa['User-State'].isin(lowercase_US_STATES)]

# Merge ratings with user data
merged_ratings = pd.merge(ratings, users_in_usa, on='User-ID', how='inner')

# Normalize publication year range
books["Year-Of-Publication"] = books["Year-Of-Publication"].clip(upper=2024,lower=1950)
merged_data = merged_ratings.merge(books, on='ISBN', how='inner')

# Load user data again for preprocessing
users = pd.read_csv('BX-Users.csv')

# Process 'User-Age' for abnormal values
users['User-Age'] = users['User-Age'].str.strip('"').replace("NaN", pd.NA).astype('Int64')
users = users[users['User-Age'] < 100]  # Remove ages greater than 100
users['User-Age'].fillna(-1, inplace=True)  # Mark missing ages with -1

# Categorize ages into bins
bins = [-1, 0, 20, 40, 60, 80, 100]
labels = [i for i in range(0, 6)]
users['User-Age Encoded'] = pd.cut(users['User-Age'], bins=bins, labels=labels, include_lowest=True)

# Create a dataframe with encoded age and user id
encoded_users = users[['User-ID','User-Age','User-Age Encoded']]

# Merge user ratings with age data
df = pd.merge(merged_data,encoded_users[['User-ID', 'User-Age', 'User-Age Encoded']], on='User-ID', how='inner')

# Drop 'User-Country' as all data is from USA
df.drop(columns=['User-Country'], inplace=True)

# Encode categorical data
df["User-State Encoded"] = label_encoder.fit_transform(df["User-State"])
df['Book-Author'] = df['Book-Author'].str.lower()
df['Book-Publisher'] = df['Book-Publisher'].str.lower()
df["Book-Author Encoded"] = label_encoder.fit_transform(df["Book-Author"])
df["Book-Publisher Encoded"] = label_encoder.fit_transform(df["Book-Publisher"])

# Binning publication year and encoding
bins = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
df["Year-Of-Publication-bin"] = pd.cut(books["Year-Of-Publication"], bins=bins, right=False)
df["Year-Of-Publication encoded"] = label_encoder.fit_transform(df["Year-Of-Publication-bin"])

# Finalize author and publisher formatting
df['Book-Author'] = df['Book-Author'].str.title()
df['Book-Publisher'] = df['Book-Publisher'].str.title()
df.drop(columns=['Year-Of-Publication-bin'], inplace=True)

# Reorder columns and save the final dataset
new_column_order = [
    'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
    'Book-Publisher', 'Book-Author Encoded','Year-Of-Publication encoded', 'Book-Publisher Encoded',
    'Book-Rating','User-ID', 'User-State','User-State Encoded', 'User-Age', 'User-Age Encoded'
]
Final_datas = df[new_column_order]
Final_datas.to_csv('Final_datas.csv',index = False)
