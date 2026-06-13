import pandas as pd

# Load files
fraud = pd.read_csv("data/Train-1542865627584.csv")
bene = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
inp = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
outp = pd.read_csv("data/Train_Outpatientdata-1542865627584.csv")

# Aggregate inpatient
inp_agg = inp.groupby("Provider").agg({
    "InscClaimAmtReimbursed": "sum",
    "DeductibleAmtPaid": "sum"
}).reset_index()

# Aggregate outpatient
outp_agg = outp.groupby("Provider").agg({
    "InscClaimAmtReimbursed": "sum",
    "DeductibleAmtPaid": "sum"
}).reset_index()

# Merge with fraud labels
df = fraud.merge(inp_agg, on="Provider", how="left")
df = df.merge(
    outp_agg,
    on="Provider",
    how="left",
    suffixes=("_in", "_out")
)

# Fill missing values
df = df.fillna(0)

# Save final dataset
df["PotentialFraud"] = df["PotentialFraud"].replace({
    "Yes": 1,
    "No": 0
})
df.to_csv("data/final_dataset.csv", index=False)

print("Dataset created successfully")
print(df.shape)
print(df.head())