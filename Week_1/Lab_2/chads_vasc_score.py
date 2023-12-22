#Calculating Risk Scores 

#Example 1: Chads vasc score - Risk of stroke

def chads_vasc_score(input_c, input_h, input_a2, input_d, input_s2, input_v, input_a, input_sc):
    #Coefficiente for Congestive heart failure 
    coef_c = 1
    #Coeffiente for hypertension
    coef_h = 1
    #Coefficient for Age >= 75 years 
    coef_a2 = 2
    #Coefficient for diabetes mellitus 
    coef_d = 1
    #Coefficent for stroke
    coef_s2 = 2
    #Coefficient for vascular disease
    coef_v = 1
    #Coefficient to Age between 65 and 74 
    coef_a = 1
    #Coeficient for sex category (female)
    coef_sc = 1

    risk_score = (input_c * coef_c) + (input_h *  coef_h) + (input_a2 * coef_a2) +\
    (input_d * coef_d) + (input_s2 * coef_s2) + (input_v * coef_v) + \
    (input_a * coef_a) + (input_sc * coef_sc)
 
    return  'The chads-vasc score is ' + str(risk_score)

# Calculate the patient's Chads-vasc risk score
tmp_c = 0
tmp_h = 1
tmp_a2 = 0
tmp_d = 0
tmp_s2 = 0
tmp_v = 1
tmp_a = 0
tmp_sc = 1

print(f"The chads-vasc score for this patient is",
      f"{chads_vasc_score(tmp_c, tmp_h, tmp_a2, tmp_d, tmp_s2, tmp_v, tmp_a, tmp_sc)}")