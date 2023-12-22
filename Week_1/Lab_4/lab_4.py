# import packages
import pandas as pd

# define 'y', the outcome of the patient
y = pd.Series([0,0,1,0])
y.name="health"
y

# Define the risk scores for each patient
risk_score = pd.Series([2.2, 3.3, 4.4, 4.4])
risk_score.name='risk score'
risk_score

# Check patients 0 and 1 make a permissible pair.
if y[0] != y[1]:
    print(f"y[0]={y[0]} and y[1]={y[1]} is a permissible pair")
else:
    print(f"y[0]={y[0]} and y[1]={y[1]} is not a permissible pair")

# Check if patients 0 and 2 make a permissible pair
if y[0] != y[2]:
    print(f"y[0]={y[0]} and y[2]={y[2]} is a permissible pair")
else:
    print(f"y[0]={y[0]} and y[2]={y[2]} is NOT permissible pair")

# Check if patients 2 and 3 make a risk tie
if risk_score[2] == risk_score[3]:
    print(f"patient 2 ({risk_score[2]}) and patient 3 ({risk_score[3]}) have a risk tie")
else:
    print(f"patient 2 ({risk_score[2]}) and patient 3 ({risk_score[3]}) DO NOT have a risk tie")

# Check if patient 1 and 2 make a concordant pair
if y[1] == 0 and y[2] == 1:
    if risk_score[1] < risk_score[2]:
        print(f"patient 1 and 2 is a concordant pair")



