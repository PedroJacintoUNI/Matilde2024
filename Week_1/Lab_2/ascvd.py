import numpy as np

def ascvd(x_age,
          x_cho,
          x_hdl,
          x_sbp,
          x_smo,
          x_dia,
          verbose=False
         ):
    """
    Atherosclerotic Cardiovascular Disease
    (ASCVD) Risk Estimator Plus
    """
    
    # Define the coefficients
    b_age = 17.114
    b_cho = 0.94
    b_hdl = -18.92
    b_age_hdl = 4.475
    b_sbp = 27.82
    b_age_sbp = -6.087
    b_smo = 0.691
    b_dia = 0.874
    
    # Calculate the sum of the products of inputs and coefficients
    sum_prod =  b_age * np.log(x_age) + \
                b_cho * np.log(x_cho) + \
                b_hdl * np.log(x_hdl) + \
                b_age_hdl * np.log(x_age) * np.log(x_hdl) +\
                b_sbp * np.log(x_sbp) +\
                b_age_sbp * np.log(x_age) * np.log(x_sbp) +\
                b_smo * x_smo + \
                b_dia * x_dia
    
    if verbose:
        print(f"np.log(x_age):{np.log(x_age):.2f}") #Prints the natural logarithm of x_age with two decimal places.
        print(f"np.log(x_cho):{np.log(x_cho):.2f}")
        print(f"np.log(x_hdl):{np.log(x_hdl):.2f}")
        #Prints the product of the natural logarithm of x_age and the natural logarithm of x_hdl with two decimal places
        print(f"np.log(x_age) * np.log(x_hdl):{np.log(x_age) * np.log(x_hdl):.2f}")
        print(f"np.log(x_sbp): {np.log(x_sbp):2f}")
        print(f"np.log(x_age) * np.log(x_sbp): {np.log(x_age) * np.log(x_sbp):.2f}")
        print(f"sum_prod {sum_prod:.2f}")
        
    # Risk Score = 1 - (0.9533^( e^(sum_prod - 86.61) ) )
    risk_score = 1 - (np.power(0.9533, (np.exp(sum_prod - 86.61))))
    
    return risk_score


tmp_risk_score = ascvd(x_age=55,
                      x_cho=213,
                      x_hdl=50,
                      x_sbp=120,
                      x_smo=0,
                      x_dia=0, 
                      verbose=True
                      )
print(f"\npatient's ascvd risk score is {tmp_risk_score: .2f}")