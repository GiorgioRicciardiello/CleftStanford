/*****************************************************************************
* Stataload_KID_2016_Core.Do
* This program will load the KID_2016_Core ASCII File into Stata.
* Because Stata loads the entire file into memory, it may not be possible
* to load every data element for large files.  If necessary, edit this
* program to change the memory size or to load only selected data elements.
*****************************************************************************/

*** Set available memory size ***
set mem 500m

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
      byte   DXVER                       48-  49   ///
      byte   ELECTIVE                    50-  51   ///
      byte   FEMALE                      52-  53   ///
      int    HCUP_ED                     54-  56   ///
      byte   HOSP_REGION                 57-  58   ///
      str    I10_DX1                     59-  65   ///
      str    I10_DX2                     66-  72   ///
      str    I10_DX3                     73-  79   ///
      str    I10_DX4                     80-  86   ///
      str    I10_DX5                     87-  93   ///
      str    I10_DX6                     94- 100   ///
      str    I10_DX7                    101- 107   ///
      str    I10_DX8                    108- 114   ///
      str    I10_DX9                    115- 121   ///
      str    I10_DX10                   122- 128   ///
      str    I10_DX11                   129- 135   ///
      str    I10_DX12                   136- 142   ///
      str    I10_DX13                   143- 149   ///
      str    I10_DX14                   150- 156   ///
      str    I10_DX15                   157- 163   ///
      str    I10_DX16                   164- 170   ///
      str    I10_DX17                   171- 177   ///
      str    I10_DX18                   178- 184   ///
      str    I10_DX19                   185- 191   ///
      str    I10_DX20                   192- 198   ///
      str    I10_DX21                   199- 205   ///
      str    I10_DX22                   206- 212   ///
      str    I10_DX23                   213- 219   ///
      str    I10_DX24                   220- 226   ///
      str    I10_DX25                   227- 233   ///
      str    I10_DX26                   234- 240   ///
      str    I10_DX27                   241- 247   ///
      str    I10_DX28                   248- 254   ///
      str    I10_DX29                   255- 261   ///
      str    I10_DX30                   262- 268   ///
      str    I10_ECAUSE1                269- 275   ///
      str    I10_ECAUSE2                276- 282   ///
      str    I10_ECAUSE3                283- 289   ///
      str    I10_ECAUSE4                290- 296   ///
      byte   I10_HOSPBRTH               297- 298   ///
      byte   I10_NDX                    299- 300   ///
      int    I10_NECAUSE                301- 303   ///
      byte   I10_NPR                    304- 305   ///
      str    I10_PR1                    306- 312   ///
      str    I10_PR2                    313- 319   ///
      str    I10_PR3                    320- 326   ///
      str    I10_PR4                    327- 333   ///
      str    I10_PR5                    334- 340   ///
      str    I10_PR6                    341- 347   ///
      str    I10_PR7                    348- 354   ///
      str    I10_PR8                    355- 361   ///
      str    I10_PR9                    362- 368   ///
      str    I10_PR10                   369- 375   ///
      str    I10_PR11                   376- 382   ///
      str    I10_PR12                   383- 389   ///
      str    I10_PR13                   390- 396   ///
      str    I10_PR14                   397- 403   ///
      str    I10_PR15                   404- 410   ///
      byte   I10_UNCBRTH                411- 412   ///
      int    KID_STRATUM                413- 416   ///
      long   LOS                        417- 421   ///
      byte   MDC                        422- 423   ///
      byte   MDC_NoPOA                  424- 425   ///
      byte   PAY1                       426- 427   ///
      int    PL_NCHS                    428- 430   ///
      int    PRDAY1                     431- 433   ///
      int    PRDAY2                     434- 436   ///
      int    PRDAY3                     437- 439   ///
      int    PRDAY4                     440- 442   ///
      int    PRDAY5                     443- 445   ///
      int    PRDAY6                     446- 448   ///
      int    PRDAY7                     449- 451   ///
      int    PRDAY8                     452- 454   ///
      int    PRDAY9                     455- 457   ///
      int    PRDAY10                    458- 460   ///
      int    PRDAY11                    461- 463   ///
      int    PRDAY12                    464- 466   ///
      int    PRDAY13                    467- 469   ///
      int    PRDAY14                    470- 472   ///
      int    PRDAY15                    473- 475   ///
      byte   PRVER                      476- 477   ///
      byte   RACE                       478- 479   ///
      double TOTCHG                     480- 489   ///
      byte   TRAN_IN                    490- 491   ///
      byte   TRAN_OUT                   492- 493   ///
      int    YEAR                       494- 497   ///
      byte   ZIPINC_QRTL                498- 499   ///
      using  "KID_2016_Core.ASC"

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
label var DXVER                    "Diagnosis Version"
label var ELECTIVE                 "Elective versus non-elective admission"
label var FEMALE                   "Indicator of sex"
label var HCUP_ED                  "HCUP Emergency Department service indicator"
label var HOSP_REGION              "Region of hospital"
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
label var I10_ECAUSE1              "ICD-10-CM External cause 1"
label var I10_ECAUSE2              "ICD-10-CM External cause 2"
label var I10_ECAUSE3              "ICD-10-CM External cause 3"
label var I10_ECAUSE4              "ICD-10-CM External cause 4"
label var I10_HOSPBRTH             "ICD-10-CM Indicator of birth in this hospital"
label var I10_NDX                  "ICD-10-CM Number of diagnoses on this record"
label var I10_NECAUSE              "ICD-10-CM Number of External cause codes on this record"
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
label var I10_UNCBRTH              "Normal uncomplicated birth in hospital"
label var KID_STRATUM              "Stratum used to sample hospital"
label var LOS                      "Length of stay (cleaned)"
label var MDC                      "MDC in effect on discharge date"
label var MDC_NoPOA                "MDC in use on discharge date, calculated without POA"
label var PAY1                     "Primary expected payer (uniform)"
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
label var PRVER                    "Procedure Version"
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
recode DXVER                     (-9 -8 -6 -5=.)
recode ELECTIVE                  (-9 -8 -6 -5=.)
recode FEMALE                    (-9 -8 -6 -5=.)
recode HCUP_ED                   (-99 -88 -66=.)
recode HOSP_REGION               (-9 -8 -6 -5=.)
recode I10_HOSPBRTH              (-9 -8 -6 -5=.)
recode I10_NDX                   (-9 -8 -6 -5=.)
recode I10_NECAUSE               (-99 -88 -66=.)
recode I10_NPR                   (-9 -8 -6 -5=.)
recode I10_UNCBRTH               (-9 -8 -6 -5=.)
recode KID_STRATUM               (-999 -888 -666=.)
recode LOS                       (-9999 -8888 -6666=.)
recode MDC                       (-9 -8 -6 -5=.)
recode MDC_NoPOA                 (-9 -8 -6 -5=.)
recode PAY1                      (-9 -8 -6 -5=.)
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
recode PRVER                     (-9 -8 -6 -5=.)
recode RACE                      (-9 -8 -6 -5=.)
recode TOTCHG                    (-999999999 -888888888 -666666666=.)
recode TRAN_IN                   (-9 -8 -6 -5=.)
recode TRAN_OUT                  (-9 -8 -6 -5=.)
recode YEAR                      (-999 -888 -666=.)
recode ZIPINC_QRTL               (-9 -8 -6 -5=.)

save "KID_2016_Core.dta", replace
