import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import seaborn as sns
import streamlit as st

df = pd.read_csv("dashboard/main.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

min_date = df["dteday"].min()
max_date = df["dteday"].max()

with st.sidebar:
    date_range = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

st.header('Bike Sharing Dataset')


filtered_data = df[(df["dteday"] >= pd.to_datetime(start_date)) & (df["dteday"] <= pd.to_datetime(end_date))]
average_temp = filtered_data["temp"].mean() * 41
average_humidity = filtered_data["hum"].mean() * 100
total_casual = filtered_data["casual"].sum()
total_registered = filtered_data["registered"].sum()

st.subheader("Metrics Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Average Temperature (°C)", value=f"{average_temp:.2f}")

with col2:
    st.metric(label="Average Humidity (%)", value=f"{average_humidity:.2f}")

st.subheader('Total Bike Sharing Order')
col3, col4 = st.columns(2)
with col3:
    st.metric(label="Total Casual Sharing", value=f"{total_casual}")
with col4:
    st.metric(label="Total Registered Sharing", value=f"{total_registered}")

st.subheader("Casual vs Registered Users Over Time")

daily_data = filtered_data.groupby("dteday")[["casual", "registered"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

# Plot casual and registered lines
ax.plot(daily_data["dteday"], daily_data["casual"], label="Casual", color="orange", linewidth=2)
ax.plot(daily_data["dteday"], daily_data["registered"], label="Registered", color="blue", linewidth=2)

# Set title and labels
plt.title('Pola Peminjaman Sepeda Per Hari')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Peminjaman Sepeda')
plt.legend(title='Kategori')
plt.grid()
st.pyplot(fig)
# Aggregate data by workingday (using mean of 'cnt')
workingday_summary = (
    filtered_data.groupby("workingday")["cnt"]
    .mean()  # Calculate mean for total rentals
    .reset_index()
)

# Mapping workingday to readable labels
workingday_summary["Type"] = workingday_summary["workingday"].map({0: "Non-Working Day", 1: "Working Day"})

# Create bar plot for average total rentals
st.subheader("Rata-Rata Peminjaman Sepeda per Jam: Hari Kerja vs Hari Libur")

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    workingday_summary["Type"],
    workingday_summary["cnt"],
    color=["orange", "blue"],
    alpha=0.8
)

# Customize the chart
ax.set_title("Rata-Rata Peminjaman Sepeda per Jam Berdasarkan Hari Kerja vs Hari Libur", fontsize=16)
ax.set_xlabel("Tipe Hari", fontsize=12)
ax.set_ylabel("Rata-Rata Peminjaman Sepeda per Jam", fontsize=12)
ax.grid(axis="y", alpha=0.5)

# Show plot in Streamlit
st.pyplot(fig)

pola_peminjaman = df.groupby('workingday')['cnt'].mean()

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
pola_peminjaman.plot(
    kind='bar',
    color=['#D3D3D3', '#90CAF9'],
    ax=ax
)

st.subheader('Pola Bike Sharing dalam Sehari menurut tipe Hari')

pola_peminjaman_jam = df.groupby(['workingday', 'hr'])['cnt'].mean()

# Membuat Streamlit layout dengan dua kolom
col1, col2 = st.columns(2)

# Plot untuk Non-Working Day (Hari Libur)
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    pola_peminjaman_jam.loc[0].plot(kind='line', ax=ax1, color='orange')
    ax1.set_title('Hari Libur', fontsize=12)
    ax1.set_xlabel('Jam', fontsize=10)
    ax1.set_ylabel('Rata-rata Peminjaman Sepeda', fontsize=10)
    ax1.grid()
    st.pyplot(fig1)

# Plot untuk Working Day (Hari Kerja)
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    pola_peminjaman_jam.loc[1].plot(kind='line', ax=ax2, color='blue')
    ax2.set_title('Hari Kerja', fontsize=12)
    ax2.set_xlabel('Jam', fontsize=10)
    ax2.set_ylabel('Rata-rata Peminjaman Sepeda', fontsize=10)
    ax2.grid()
    st.pyplot(fig2)

weather_effect_hour = df.groupby('weathersit')['cnt'].mean().reset_index()

# Membuat Streamlit layout untuk menampilkan visualisasi dan detail
st.subheader("Pengaruh Cuaca terhadap Peminjaman Sepeda")

# Visualisasi bar plot untuk rata-rata peminjaman sepeda berdasarkan cuaca
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(weather_effect_hour['weathersit'], weather_effect_hour['cnt'], color='#90CAF9')
ax.set_title('Rata-rata Peminjaman Sepeda Berdasarkan Cuaca', fontsize=16)
ax.set_xlabel('Cuaca', fontsize=12)
ax.set_ylabel('Rata-rata Peminjaman Sepeda', fontsize=12)
ax.set_xticks([0, 1, 2, 3, 4])  # Asumsi kategori cuaca ada 4 jenis (0, 1, 2, 3)
ax.set_xticklabels(['','Clear', 'Mist', 'Light Rain', 'Heavy Rain'], rotation=0)
ax.grid(axis="y", alpha=0.5)
st.pyplot(fig)

# Penjelasan detail mengenai cuaca
with st.expander("Lihat Detail Cuaca"):
    st.markdown("""
    1. **Clear**: Cuaca cerah dengan sedikit atau tanpa awan, umumnya mendukung peminjaman sepeda yang tinggi.
    2. **Mist**: Cuaca berkabut, mengurangi visibilitas tetapi tidak terlalu mengganggu penggunaan sepeda.
    3. **Light Rain**: Hujan ringan yang membuat beberapa orang ragu untuk menggunakan sepeda, tetapi masih memungkinkan.
    4. **Heavy Rain**: Hujan lebat yang umumnya mengurangi peminjaman sepeda karena ketidaknyamanan.
    """)

# Aggregasi data hourly menjadi daily
day_df = df.groupby("dteday").agg({
    "cnt": "sum",         # Total peminjaman sepeda per hari
    "temp": "mean",       # Rata-rata suhu per hari
    "hum": "mean",        # Rata-rata kelembaban per hari
    "windspeed": "mean",  # Rata-rata kecepatan angin per hari
}).reset_index()

# Menambahkan fitur tambahan jika diperlukan
day_df["day_of_week"] = day_df["dteday"].dt.dayofweek  # Menambahkan hari dalam minggu (0=Senin, 6=Minggu)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data untuk clustering
hour_df = df[["hr", "cnt"]]

# Membuat clustering dengan K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
hour_df["Cluster"] = kmeans.fit_predict(hour_df[["hr", "cnt"]])

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = hour_df[hour_df["Cluster"] == cluster]
    plt.scatter(cluster_data["hr"], cluster_data["cnt"], label=f"Cluster {cluster}")

# Menambahkan centroid ke plot
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c="red", marker="X", label="Centroids")

plt.xlabel("Hour")
plt.ylabel("Count (cnt)")
plt.title("Clustering of Rentals Based on Hour and Count")
plt.legend()
plt.grid()
plt.show()

st.subheader("Clustering Data Hourly")
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in range(k):
    cluster_data = hour_df[hour_df["Cluster"] == cluster]
    ax.scatter(cluster_data["hr"], cluster_data["cnt"], label=f"Cluster {cluster}")

centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c="red", marker="X", label="Centroids")

ax.set_xlabel("Hour")
ax.set_ylabel("Count (cnt)")
ax.set_title("Clustering of Rentals Based on Hour and Count")
ax.legend()
st.pyplot(fig)

# Menampilkan jumlah data di setiap cluster
cluster_counts = hour_df["Cluster"].value_counts()
print("Jumlah data di setiap cluster:")
print(cluster_counts)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Normalisasi fitur untuk clustering
scaler = StandardScaler()
features = day_df[["temp", "hum"]]
features_scaled = scaler.fit_transform(features)

# Mencari jumlah cluster optimal
best_k = None
best_score = -1

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    if score > best_score:
        best_k = k
        best_score = score

print(f"Jumlah cluster optimal: {best_k}")

# Membuat clustering dengan jumlah cluster optimal
kmeans = KMeans(n_clusters=best_k, random_state=42)
day_df["Cluster"] = kmeans.fit_predict(features_scaled)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
for cluster in np.unique(day_df["Cluster"]):
    cluster_data = day_df[day_df["Cluster"] == cluster]
    plt.scatter(cluster_data["temp"], cluster_data["hum"], label=f"Cluster {cluster}", alpha=0.6)

# Menambahkan centroid ke plot
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], s=300, c="red", marker="X", label="Centroids")

plt.title("Clustering of Bike Rentals Based on Temperature and Humidity")
plt.xlabel("Temperature (temp)")
plt.ylabel("Humidity (hum)")
plt.legend()
plt.grid()
plt.show()

# Menampilkan jumlah data di setiap cluster
cluster_counts = day_df["Cluster"].value_counts()
print("Jumlah data di setiap cluster:")
print(cluster_counts)


st.subheader("Clustering Data Daily")
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in np.unique(day_df["Cluster"]):
    cluster_data = day_df[day_df["Cluster"] == cluster]
    ax.scatter(cluster_data["temp"], cluster_data["hum"], label=f"Cluster {cluster}", alpha=0.6)

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], s=300, c="red", marker="X", label="Centroids")

ax.set_title("Clustering of Bike Rentals Based on Temperature and Humidity")
ax.set_xlabel("Temperature (temp)")
ax.set_ylabel("Humidity (hum)")
ax.legend()
st.pyplot(fig)

with st.expander("See Detail"):
    st.markdown("""
    - **Temperature** yang ada pada plot ini telah dibagi dengan **41** (hasil dari pengolahan data).
    - **Humidity** yang ada pada plot ini telah dibagi dengan **100**.
    - Dengan kata lain, untuk memahami nilai yang sesungguhnya, **humidity** harus dikalikan 100 dan **temperature** harus dikalikan 41.
    """)