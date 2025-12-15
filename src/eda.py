import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset (pandas read CSV)
df = pd.read_csv("data/Train.csv")


# Basic info about the dataset
print("BASIC DATASET INFO")
print("Number of rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())


# Check missing values
print("\nMISSING VALUES PER COLUMN")
print(df.isna().sum())


# On-time vs late deliveries
# Reached.on.Time_Y.N: 1 = reached on time; 0 = late
print("\nON-TIME VS LATE DELIVERIES")
print(df["Reached.on.Time_Y.N"].value_counts())


# Plot: On-time vs late deliveries
sns.countplot(data=df, x="Reached.on.Time_Y.N")
plt.title("On-time (1) vs Late (0) Deliveries")
plt.xlabel("Reached on Time (1 = Yes, 0 = Late)")
plt.ylabel("Number of Deliveries")
plt.show()


# Plot: Delivery status by Mode of shipment
sns.countplot(
    data=df,
    x="Mode_of_Shipment",
    hue="Reached.on.Time_Y.N"
)
plt.title("Delivery Status by Mode of shipment")
plt.xlabel("Mode of shipment")
plt.ylabel("Number of deliveries")
plt.legend(title="Reached on time (1 = Yes, 0 = Late)")
plt.show()


# Plot: Weight distribution by delivery status
sns.boxplot(
    data=df,
    x="Reached.on.Time_Y.N",
    y="Weight_in_gms"
)
plt.title("Weight Distribution by Delivery Status")
plt.xlabel("Reached on Time (1 = Yes, 0 = Late)")
plt.ylabel("Weight in grams")
plt.show()
