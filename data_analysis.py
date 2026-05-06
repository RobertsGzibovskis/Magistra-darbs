import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Ielādē datu kopu

file_path = "C:\Magistrs_praktiskais/LIME_XAI/data.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)


# Izvēlās iezīmes

numeric_df = df.select_dtypes(include=['number']).dropna()


# Iezīmju korelācija ar iezīmi "Popularitāte"

if "popularity" in numeric_df.columns:
    corr = numeric_df.corr()
    
    print("\nAtribūtu korelācija ar atribūtu 'Popularity':\n")
    print(corr["popularity"].sort_values(ascending=False))


    plt.figure(figsize=(8,5))
    corr["popularity"].sort_values().plot(kind='barh')
    plt.title("Iezīmju korelācija ar iezīmi 'Popularity'")
    plt.tight_layout()
    plt.show()


# Iezīmju iekliede

selected_features = ["mode", "popularity", "tempo", "acousticness"]
colors = ["blue", "green", "orange", "red"]

selected_df = numeric_df[selected_features].dropna()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    axes[i].hist(selected_df[feature], bins=30, color=colors[i])
    axes[i].set_title(feature)

plt.suptitle("Iezīmju sadalījums")
plt.tight_layout()
plt.show()


# Dziesmu skaits
print("Dziesmu skaits:", len(df))

# Dziesmu skaits gadā

if "year" in df.columns:
    songs_per_year = df["year"].value_counts().sort_index()

    plt.figure(figsize=(10,5))
    songs_per_year.plot(kind='line')
    plt.title("Dziesmu skaits pa gadiem")
    plt.xlabel("Gads")
    plt.ylabel("Dziesmu skaits")
    plt.tight_layout()
    plt.show()



# Dziesmu skaits dekādē

if "year" in df.columns:
    df["decade"] = (df["year"] // 10) * 10
    songs_per_decade = df["decade"].value_counts().sort_index()

    plt.figure(figsize=(8,5))
    songs_per_decade.plot(kind='bar')
    plt.title("Dziesmu skaits dekādē")
    plt.xlabel("Dekāde")
    plt.ylabel("Dziesmu skaits")
    plt.tight_layout()
    plt.show()
