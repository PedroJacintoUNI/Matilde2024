import numpy as np

def ascvd (age, choles, HDL, S_BP, input_s, input_d):
    #Coefficiente for Age 
    coef_age = 17.114
    #Coeffiente for Total Cholesterol 
    coef_choles = 0.940
    #Coefficient for HDL
    coef_HDL = -18.920
    #Coefficient for the multiplication of age and HDL
    coef_age_HDL = 4.475
    #Coefficent for untreated systolic BP 
    coef_S_BP = 27.820
    #Coefficient for the multiplication of age and untreated systolic BP 
    coef_age_S_BP = -6.087
    #Coefficient to Smoking
    coef_s = 0.691
    #Coeficient for diabetes
    coef_d = 0.874

     # Calculate the natural logarithm of input variables
    log_age = np.log(age)
    log_choles = np.log(choles)
    log_HDL = np.log(HDL)
    log_S_BP = np.log(S_BP)

    #Multipling factors
    age_HDL = log_age * log_HDL
    age_S_BP  = log_age * log_S_BP

    #Comput Output 
    sum = (log_age * coef_age) + (log_choles * coef_choles) + (age_HDL * coef_age_HDL) +\
    (log_S_BP * coef_S_BP) + (age_S_BP * coef_age_S_BP) + (input_s * coef_s) + \
    (input_d * coef_d)

    #Calculate the risk 
    risk = 1 - (np.power(0.9533, (np.exp(sum - 86.61))))

    return risk

tmp_risk_score = ascvd(55, 213, 50, 120, 0,0)
print(f"\npatient's ascvd risk score is {tmp_risk_score:.2f}")