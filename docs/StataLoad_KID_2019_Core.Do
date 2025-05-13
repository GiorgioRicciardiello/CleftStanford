/*****************************************************************************
* Stataload_KID_2019_Core.Do
* This program will load the KID_2019_Core ASCII File into Stata.
* Because Stata loads the entire file into memory, it may not be possible
* to load every data element for large files.  If necessary, edit this
* program to change the memory size or to load only selected data elements.
* Created on 12/02/2022.
*****************************************************************************/

*** Read data elements from the ASCII file ***
infix long   HOSP_KID                     1-   5   ///
      long   RECNUM                       6-  13   ///
      int    AGE                         14-  16   ///
      byte   AGE_NEONATE                 17-  18   ///
      byte   AMONTH                      19-  20   ///
      byte   AWEEKEND                    21-  22   ///
      byte   DIED                        23-  24   ///
      double DISCWT                      25-  35   ///
      byte   DISPUNIFORM                 36-  37   ///
      byte   DQTR                        38-  39   ///
      int    DRG                         40-  42   ///
      byte   DRGVER                      43-  44   ///
      int    DRG_NoPOA                   45-  47   ///
      byte   ELECTIVE                    48-  49   ///
      byte   FEMALE                      50-  51   ///
      int    HCUP_ED                     52-  54   ///
      byte   HOSP_REGION                 55-  56   ///
      int    I10_BIRTH                   57-  59   ///
      int    I10_DELIVERY                60-  62   ///
      str    I10_DX1                     63-  69   ///
      str    I10_DX2                     70-  76   ///
      str    I10_DX3                     77-  83   ///
      str    I10_DX4                     84-  90   ///
      str    I10_DX5                     91-  97   ///
      str    I10_DX6                     98- 104   ///
      str    I10_DX7                    105- 111   ///
      str    I10_DX8                    112- 118   ///
      str    I10_DX9                    119- 125   ///
      str    I10_DX10                   126- 132   ///
      str    I10_DX11                   133- 139   ///
      str    I10_DX12                   140- 146   ///
      str    I10_DX13                   147- 153   ///
      str    I10_DX14                   154- 160   ///
      str    I10_DX15                   161- 167   ///
      str    I10_DX16                   168- 174   ///
      str    I10_DX17                   175- 181   ///
      str    I10_DX18                   182- 188   ///
      str    I10_DX19                   189- 195   ///
      str    I10_DX20                   196- 202   ///
      str    I10_DX21                   203- 209   ///
      str    I10_DX22                   210- 216   ///
      str    I10_DX23                   217- 223   ///
      str    I10_DX24                   224- 230   ///
      str    I10_DX25                   231- 237   ///
      str    I10_DX26                   238- 244   ///
      str    I10_DX27                   245- 251   ///
      str    I10_DX28                   252- 258   ///
      str    I10_DX29                   259- 265   ///
      str    I10_DX30                   266- 272   ///
      str    I10_DX31                   273- 279   ///
      str    I10_DX32                   280- 286   ///
      str    I10_DX33                   287- 293   ///
      str    I10_DX34                   294- 300   ///
      str    I10_DX35                   301- 307   ///
      str    I10_DX36                   308- 314   ///
      str    I10_DX37                   315- 321   ///
      str    I10_DX38                   322- 328   ///
      str    I10_DX39                   329- 335   ///
      str    I10_DX40                   336- 342   ///
      byte   I10_HOSPBRTH               343- 344   ///
      byte   I10_INJURY                 345- 346   ///
      byte   I10_MULTINJURY             347- 348   ///
      byte   I10_NDX                    349- 350   ///
      byte   I10_NPR                    351- 352   ///
      str    I10_PR1                    353- 359   ///
      str    I10_PR2                    360- 366   ///
      str    I10_PR3                    367- 373   ///
      str    I10_PR4                    374- 380   ///
      str    I10_PR5                    381- 387   ///
      str    I10_PR6                    388- 394   ///
      str    I10_PR7                    395- 401   ///
      str    I10_PR8                    402- 408   ///
      str    I10_PR9                    409- 415   ///
      str    I10_PR10                   416- 422   ///
      str    I10_PR11                   423- 429   ///
      str    I10_PR12                   430- 436   ///
      str    I10_PR13                   437- 443   ///
      str    I10_PR14                   444- 450   ///
      str    I10_PR15                   451- 457   ///
      str    I10_PR16                   458- 464   ///
      str    I10_PR17                   465- 471   ///
      str    I10_PR18                   472- 478   ///
      str    I10_PR19                   479- 485   ///
      str    I10_PR20                   486- 492   ///
      str    I10_PR21                   493- 499   ///
      str    I10_PR22                   500- 506   ///
      str    I10_PR23                   507- 513   ///
      str    I10_PR24                   514- 520   ///
      str    I10_PR25                   521- 527   ///
      int    I10_SERVICELINE            528- 530   ///
      byte   I10_UNCBRTH                531- 532   ///
      int    KID_STRATUM                533- 536   ///
      long   LOS                        537- 541   ///
      byte   MDC                        542- 543   ///
      byte   MDC_NoPOA                  544- 545   ///
      byte   PAY1                       546- 547   ///
      byte   PCLASS_ORPROC              548- 549   ///
      int    PL_NCHS                    550- 552   ///
      int    PRDAY1                     553- 555   ///
      int    PRDAY2                     556- 558   ///
      int    PRDAY3                     559- 561   ///
      int    PRDAY4                     562- 564   ///
      int    PRDAY5                     565- 567   ///
      int    PRDAY6                     568- 570   ///
      int    PRDAY7                     571- 573   ///
      int    PRDAY8                     574- 576   ///
      int    PRDAY9                     577- 579   ///
      int    PRDAY10                    580- 582   ///
      int    PRDAY11                    583- 585   ///
      int    PRDAY12                    586- 588   ///
      int    PRDAY13                    589- 591   ///
      int    PRDAY14                    592- 594   ///
      int    PRDAY15                    595- 597   ///
      int    PRDAY16                    598- 600   ///
      int    PRDAY17                    601- 603   ///
      int    PRDAY18                    604- 606   ///
      int    PRDAY19                    607- 609   ///
      int    PRDAY20                    610- 612   ///
      int    PRDAY21                    613- 615   ///
      int    PRDAY22                    616- 618   ///
      int    PRDAY23                    619- 621   ///
      int    PRDAY24                    622- 624   ///
      int    PRDAY25                    625- 627   ///
      byte   RACE                       628- 629   ///
      double TOTCHG                     630- 639   ///
      byte   TRAN_IN                    640- 641   ///
      byte   TRAN_OUT                   642- 643   ///
      int    YEAR                       644- 647   ///
      byte   ZIPINC_QRTL                648- 649   ///
      using  "KID_2019_Core.ASC"

***  Assign labels to the data elements ***
label var HOSP_KID                 "KID hospital number"
label var RECNUM                   "KID record number"
label var AGE                      "Age in years at admission"
label var AGE_NEONATE              "Neonatal age (first 28 days after birth) indicator"
label var AMONTH                   "Admission month"
label var AWEEKEND                 "Admission day is a weekend"
label var DIED                     "Died during hospitalization"
label var DISCWT                   "KID discharge weight"
label var DISPUNIFORM              "Disposition of patient (uniform)"
label var DQTR                     "Discharge quarter"
label var DRG                      "DRG in effect on discharge date"
label var DRGVER                   "DRG grouper version used on discharge date"
label var DRG_NoPOA                "DRG in use on discharge date, calculated without POA"
label var ELECTIVE                 "Elective versus non-elective admission"
label var FEMALE                   "Indicator of sex"
label var HCUP_ED                  "HCUP Emergency Department service indicator"
label var HOSP_REGION              "Region of hospital"
label var I10_BIRTH                "ICD-10-CM Birth Indicator"
label var I10_DELIVERY             "ICD-10-CM Delivery Indicator"
label var I10_DX1                  "ICD-10-CM Diagnosis 1"
label var I10_DX2                  "ICD-10-CM Diagnosis 2"
label var I10_DX3                  "ICD-10-CM Diagnosis 3"
label var I10_DX4                  "ICD-10-CM Diagnosis 4"
label var I10_DX5                  "ICD-10-CM Diagnosis 5"
label var I10_DX6                  "ICD-10-CM Diagnosis 6"
label var I10_DX7                  "ICD-10-CM Diagnosis 7"
label var I10_DX8                  "ICD-10-CM Diagnosis 8"
label var I10_DX9                  "ICD-10-CM Diagnosis 9"
label var I10_DX10                 "ICD-10-CM Diagnosis 10"
label var I10_DX11                 "ICD-10-CM Diagnosis 11"
label var I10_DX12                 "ICD-10-CM Diagnosis 12"
label var I10_DX13                 "ICD-10-CM Diagnosis 13"
label var I10_DX14                 "ICD-10-CM Diagnosis 14"
label var I10_DX15                 "ICD-10-CM Diagnosis 15"
label var I10_DX16                 "ICD-10-CM Diagnosis 16"
label var I10_DX17                 "ICD-10-CM Diagnosis 17"
label var I10_DX18                 "ICD-10-CM Diagnosis 18"
label var I10_DX19                 "ICD-10-CM Diagnosis 19"
label var I10_DX20                 "ICD-10-CM Diagnosis 20"
label var I10_DX21                 "ICD-10-CM Diagnosis 21"
label var I10_DX22                 "ICD-10-CM Diagnosis 22"
label var I10_DX23                 "ICD-10-CM Diagnosis 23"
label var I10_DX24                 "ICD-10-CM Diagnosis 24"
label var I10_DX25                 "ICD-10-CM Diagnosis 25"
label var I10_DX26                 "ICD-10-CM Diagnosis 26"
label var I10_DX27                 "ICD-10-CM Diagnosis 27"
label var I10_DX28                 "ICD-10-CM Diagnosis 28"
label var I10_DX29                 "ICD-10-CM Diagnosis 29"
label var I10_DX30                 "ICD-10-CM Diagnosis 30"
label var I10_DX31                 "ICD-10-CM Diagnosis 31"
label var I10_DX32                 "ICD-10-CM Diagnosis 32"
label var I10_DX33                 "ICD-10-CM Diagnosis 33"
label var I10_DX34                 "ICD-10-CM Diagnosis 34"
label var I10_DX35                 "ICD-10-CM Diagnosis 35"
label var I10_DX36                 "ICD-10-CM Diagnosis 36"
label var I10_DX37                 "ICD-10-CM Diagnosis 37"
label var I10_DX38                 "ICD-10-CM Diagnosis 38"
label var I10_DX39                 "ICD-10-CM Diagnosis 39"
label var I10_DX40                 "ICD-10-CM Diagnosis 40"
label var I10_HOSPBRTH             "ICD-10-CM Indicator of birth in this hospital"
label var I10_INJURY               "Injury ICD-10-CM diagnosis reported on record (1: First-listed injury; 2: Other than first-listed injury; 0: No injury)"
label var I10_MULTINJURY           "Multiple ICD-10-CM injuries reported on record"
label var I10_NDX                  "ICD-10-CM Number of diagnoses on this record"
label var I10_NPR                  "ICD-10-PCS Number of procedures on this record"
label var I10_PR1                  "ICD-10-PCS Procedure 1"
label var I10_PR2                  "ICD-10-PCS Procedure 2"
label var I10_PR3                  "ICD-10-PCS Procedure 3"
label var I10_PR4                  "ICD-10-PCS Procedure 4"
label var I10_PR5                  "ICD-10-PCS Procedure 5"
label var I10_PR6                  "ICD-10-PCS Procedure 6"
label var I10_PR7                  "ICD-10-PCS Procedure 7"
label var I10_PR8                  "ICD-10-PCS Procedure 8"
label var I10_PR9                  "ICD-10-PCS Procedure 9"
label var I10_PR10                 "ICD-10-PCS Procedure 10"
label var I10_PR11                 "ICD-10-PCS Procedure 11"
label var I10_PR12                 "ICD-10-PCS Procedure 12"
label var I10_PR13                 "ICD-10-PCS Procedure 13"
label var I10_PR14                 "ICD-10-PCS Procedure 14"
label var I10_PR15                 "ICD-10-PCS Procedure 15"
label var I10_PR16                 "ICD-10-PCS Procedure 16"
label var I10_PR17                 "ICD-10-PCS Procedure 17"
label var I10_PR18                 "ICD-10-PCS Procedure 18"
label var I10_PR19                 "ICD-10-PCS Procedure 19"
label var I10_PR20                 "ICD-10-PCS Procedure 20"
label var I10_PR21                 "ICD-10-PCS Procedure 21"
label var I10_PR22                 "ICD-10-PCS Procedure 22"
label var I10_PR23                 "ICD-10-PCS Procedure 23"
label var I10_PR24                 "ICD-10-PCS Procedure 24"
label var I10_PR25                 "ICD-10-PCS Procedure 25"
label var I10_SERVICELINE          "ICD-10-CM/PCS Hospital Service Line"
label var I10_UNCBRTH              "ICD-10-CM Indicator of normal newborn born in hospital"
label var KID_STRATUM              "Stratum used to sample hospital"
label var LOS                      "Length of stay (cleaned)"
label var MDC                      "MDC in effect on discharge date"
label var MDC_NoPOA                "MDC in use on discharge date, calculated without POA"
label var PAY1                     "Primary expected payer (uniform)"
label var PCLASS_ORPROC            "Indicates operating room (major diagnostic or therapeutic) procedure on the record"
label var PL_NCHS                  "Patient Location: NCHS Urban-Rural Code"
label var PRDAY1                   "Number of days from admission to I10_PR1"
label var PRDAY2                   "Number of days from admission to I10_PR2"
label var PRDAY3                   "Number of days from admission to I10_PR3"
label var PRDAY4                   "Number of days from admission to I10_PR4"
label var PRDAY5                   "Number of days from admission to I10_PR5"
label var PRDAY6                   "Number of days from admission to I10_PR6"
label var PRDAY7                   "Number of days from admission to I10_PR7"
label var PRDAY8                   "Number of days from admission to I10_PR8"
label var PRDAY9                   "Number of days from admission to I10_PR9"
label var PRDAY10                  "Number of days from admission to I10_PR10"
label var PRDAY11                  "Number of days from admission to I10_PR11"
label var PRDAY12                  "Number of days from admission to I10_PR12"
label var PRDAY13                  "Number of days from admission to I10_PR13"
label var PRDAY14                  "Number of days from admission to I10_PR14"
label var PRDAY15                  "Number of days from admission to I10_PR15"
label var PRDAY16                  "Number of days from admission to I10_PR16"
label var PRDAY17                  "Number of days from admission to I10_PR17"
label var PRDAY18                  "Number of days from admission to I10_PR18"
label var PRDAY19                  "Number of days from admission to I10_PR19"
label var PRDAY20                  "Number of days from admission to I10_PR20"
label var PRDAY21                  "Number of days from admission to I10_PR21"
label var PRDAY22                  "Number of days from admission to I10_PR22"
label var PRDAY23                  "Number of days from admission to I10_PR23"
label var PRDAY24                  "Number of days from admission to I10_PR24"
label var PRDAY25                  "Number of days from admission to I10_PR25"
label var RACE                     "Race (uniform)"
label var TOTCHG                   "Total charges (cleaned)"
label var TRAN_IN                  "Transfer in indicator"
label var TRAN_OUT                 "Transfer out indicator"
label var YEAR                     "Calendar year"
label var ZIPINC_QRTL              "Median household income national quartile for patient ZIP Code"

*** Convert special values to missing values ***
recode HOSP_KID                  (-9999 -8888 -6666=.)
recode RECNUM                    (-9999999 -8888888 -6666666=.)
recode AGE                       (-99 -88 -66=.)
recode AGE_NEONATE               (-9 -8 -6 -5=.)
recode AMONTH                    (-9 -8 -6 -5=.)
recode AWEEKEND                  (-9 -8 -6 -5=.)
recode DIED                      (-9 -8 -6 -5=.)
recode DISCWT                    (-99.9999999 -88.8888888 -66.6666666=.)
recode DISPUNIFORM               (-9 -8 -6 -5=.)
recode DQTR                      (-9 -8 -6 -5=.)
recode DRG                       (-99 -88 -66=.)
recode DRGVER                    (-9 -8 -6 -5=.)
recode DRG_NoPOA                 (-99 -88 -66=.)
recode ELECTIVE                  (-9 -8 -6 -5=.)
recode FEMALE                    (-9 -8 -6 -5=.)
recode HCUP_ED                   (-99 -88 -66=.)
recode HOSP_REGION               (-9 -8 -6 -5=.)
recode I10_BIRTH                 (-99 -88 -66=.)
recode I10_DELIVERY              (-99 -88 -66=.)
recode I10_HOSPBRTH              (-9 -8 -6 -5=.)
recode I10_INJURY                (-9 -8 -6 -5=.)
recode I10_MULTINJURY            (-9 -8 -6 -5=.)
recode I10_NDX                   (-9 -8 -6 -5=.)
recode I10_NPR                   (-9 -8 -6 -5=.)
recode I10_SERVICELINE           (-99 -88 -66=.)
recode I10_UNCBRTH               (-9 -8 -6 -5=.)
recode KID_STRATUM               (-999 -888 -666=.)
recode LOS                       (-9999 -8888 -6666=.)
recode MDC                       (-9 -8 -6 -5=.)
recode MDC_NoPOA                 (-9 -8 -6 -5=.)
recode PAY1                      (-9 -8 -6 -5=.)
recode PCLASS_ORPROC             (-9 -8 -6 -5=.)
recode PL_NCHS                   (-99 -88 -66=.)
recode PRDAY1                    (-99 -88 -66=.)
recode PRDAY2                    (-99 -88 -66=.)
recode PRDAY3                    (-99 -88 -66=.)
recode PRDAY4                    (-99 -88 -66=.)
recode PRDAY5                    (-99 -88 -66=.)
recode PRDAY6                    (-99 -88 -66=.)
recode PRDAY7                    (-99 -88 -66=.)
recode PRDAY8                    (-99 -88 -66=.)
recode PRDAY9                    (-99 -88 -66=.)
recode PRDAY10                   (-99 -88 -66=.)
recode PRDAY11                   (-99 -88 -66=.)
recode PRDAY12                   (-99 -88 -66=.)
recode PRDAY13                   (-99 -88 -66=.)
recode PRDAY14                   (-99 -88 -66=.)
recode PRDAY15                   (-99 -88 -66=.)
recode PRDAY16                   (-99 -88 -66=.)
recode PRDAY17                   (-99 -88 -66=.)
recode PRDAY18                   (-99 -88 -66=.)
recode PRDAY19                   (-99 -88 -66=.)
recode PRDAY20                   (-99 -88 -66=.)
recode PRDAY21                   (-99 -88 -66=.)
recode PRDAY22                   (-99 -88 -66=.)
recode PRDAY23                   (-99 -88 -66=.)
recode PRDAY24                   (-99 -88 -66=.)
recode PRDAY25                   (-99 -88 -66=.)
recode RACE                      (-9 -8 -6 -5=.)
recode TOTCHG                    (-999999999 -888888888 -666666666=.)
recode TRAN_IN                   (-9 -8 -6 -5=.)
recode TRAN_OUT                  (-9 -8 -6 -5=.)
recode YEAR                      (-999 -888 -666=.)
recode ZIPINC_QRTL               (-9 -8 -6 -5=.)

save "KID_2019_Core.dta", replace
