import numpy as np

def liver_disease_mortality(input_creatine, input_bilirubin, input_inr):
    """
    Calculate the probability of mortality given that the patient has
    liver disease. 
    Parameters:
        Creatine: mg/dL
        Bilirubin: mg/dL
        INR: 
    """
    # Coefficient values
    coef_creatine = 0.957
    coef_bilirubin = 0.378
    coef_inr = 1.12
    intercept = 0.643
    # Calculate the natural logarithm of input variables
    log_cre = np.log(input_creatine)
    log_bil = np.log(input_bilirubin)
    
    # Calculate the natural log of input_inr
    log_inr = np.log(input_inr)
    
    # Compute output
    meld_score = (coef_creatine * log_cre) +\
                 (coef_bilirubin * log_bil ) +\
                 (coef_inr * log_inr) +\
                 intercept
    
    # Multiply meld_score by 10 to get the final risk score
    meld_score = meld_score * 10
    
    return meld_score


tmp_meld_score = liver_disease_mortality(0.8, 1.5, 1.3)
print(f"The patient's MELD score is: {tmp_meld_score:.2f}")