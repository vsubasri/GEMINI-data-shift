
DIAGNOSIS_DICT = {
        'Certain infectious and parasitic diseases': ('A00', 'B99'),
        'Neoplasms': ('C00', 'D49'),
        'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': ('D50','D89'),
        'Endocrine, nutritional and metabolic diseases': ('E00', 'E89'),
        'Mental, Behavioral and Neurodevelopmental disorders': ('F01', 'F99'),
        'Diseases of the nervous system': ('G00', 'G99'),
        'Diseases of the eye and adnexa': ('H00', 'H59'),
        'Diseases of the ear and mastoid process': ('H60', 'H95'),
        'Diseases of the circulatory system': ('I00', 'I99'),
        'Diseases of the respiratory system': ('J00', 'J99'),
        'Diseases of the digestive system': ('K00', 'K95'),
        'Diseases of the skin and subcutaneous tissue': ('L00', 'L99'),
        'Diseases of the musculoskeletal system and connective tissue': ('M00', 'M99'),
        'Diseases of the genitourinary system': ('N00', 'N99'),
        'Pregnancy, childbirth and the puerperium': ('O00', 'O99'),
        'Certain conditions originating in the perinatal period': ('P00', 'P96'),
        'Congenital malformations, deformations and chromosomal abnormalities': ('Q00','Q99'),
        'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': ('R00', 'R99'),
        'Injury, poisoning and certain other consequences of external causes': ('S00', 'T88'),
        'External causes of morbidity': ('V00', 'Y99'),
        'COVID19': ('U07', 'U08'),
        'Factors influencing health status and contact with health services': ('Z00', 'Z99')
} 


COMORBIDITIES = {
    'Kidney disease' : {'diagnosis_code':['N18, N19']},
    'Ischemic heart disease' : {'diagnosis_code':['I20','I21', 'I22', 'I23', 'I24','I25']},
    'Other heart disease' : {'diagnosis_code':['I27', 'I28', 'I29', 'I30', 'I31', 'I32' ,'I33' , 'I34','I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51','I52']},
    'Cerebrovascular disease': {'diagnosis_code':['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69']},
    'Hypertension': {'diagnosis_code':['I10','I11','I12','I13','I14','I15']},
    'Diabetes' : {'diagnosis_code':['E10', 'E11', 'E12', 'E13']},
    'Hyperlipidemia': {'diagnosis_code':['E78']},
    'Hypertension' : {'diagnosis_code':['I10']},
    'Congestive heart failure' : {'diagnosis_code':['I50']},
    'Cancer' : {'diagnosis_trajectory':['C00_D49']},
    'Dyspnea': {'diagnosis_code':['R06']},
    'COPD': {'diagnosis_code':['J44']},
    'Asthma': {'diagnosis_code':['J45']},
    'Pulmonary embolism': {'diagnosis_code':['I26']},
    'Connective tissue disease': {'diagnosis_code':['I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36']},
    'Inflammatory bowel disease': {'diagnosis_code':['K50', 'K51']},
    'Osteoarthritis': {'diagnosis_code':['M15', 'M16', 'M17', 'M18', 'M19']},
    'Rheumatoid arthritis': {'diagnosis_code':['M05', 'M06', 'M07', 'M08', 'M09', 'M010', 'M011', 'M012', 'M013', 'M014']},
    'HIV': {'diagnosis_code':['B20', 'B21', 'B22', 'B23', 'B24']},
}

ACADEMIC=["MSH","PMH", "SMH","UHNTW","UHNTG","SBK"]
COMMUNITY=["THPC","THPM"]
HOSPITALS = ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW", "SBK"]
