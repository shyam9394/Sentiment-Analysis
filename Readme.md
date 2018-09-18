# Spam or Ham
## Introduction 
The dataset for this project was taken from Kaggle.com. This code helps in detecting whether a text is spam or ham. Several operations are performed on the dataset and a bag of words is created (Sparse Matrix). Random forest classifier is used here to classify if a text is spam or ham. With the help of confusion matrix, an accuracy of 97.85% is obtained.

## Implementation
### Importing libraries

`import pandas as pd`  
Used for importing dataset

`import re`   
Used for text cleaning

`import nltk`  
Library used for natural language processing

`from nltk.corpus import stopwords`  
Used for importing stop words

`from nltk.stem.porter import PorterStemmer`  
Used for stemming words
  
`from sklearn.feature_extraction.text import CountVectorizer`   
Creates bag of words 

`from sklearn.preprocessing import LabelEncoder`  
Encodes categorical data

`from sklearn.model_selection import train_test_split`  
Used for splitting dataset

`from sklearn.ensemble import RandomForestClassifier`  
Random forest classifier

`from sklearn.metrics import confusion_matrix`  
Used for creating confusion matrix

`from sklearn.model_selection import GridSearchCV`  
Used for performance tuning

### Importing dataset  
`dataset = pd.read_csv('spam.csv', sep=',', engine='python')`

![](https://lh3.googleusercontent.com/AwClWINb6fBGgY4hRBnVucpm4M2ghhWr-jx36WHX174e4OInOOeQXo3yEq70ab8o-hz7RMbJIX2hcr0LwjnFkTWzj3V9596k42ONR23LTmd6bPWgH89hR_iqnCQG8ZgBUD6Net3WM06L2QV58okWkeAdSI7F-r8W7potXE0XmdANR2M-Blrm0oZ3sWcax40eEYzL6JK8i_Z3RMb1duf2JNUo4vaivZhTmdKkIdTvZOc5Od2v_6djobSBgbDfLSXQbCcdY7_6h-70TT-qocE-_T1fx1_PZcA7bmTitIY8W8WLdcNLIQPz6FxgmbT9Fs0sRLKKLe48iaj1NGDlHBx7jg7SzsgOoy4HCwpcgy0jr2LDAFnwr_x0Ey93KyzgCwzH5DQGAG8Cjt9EV9uWlfBSq8cln-_WASJ6Lrkr3BowQon3yy8cW1g8DCd-J2udxTeXxWFPGBl_1SYogOzAW7AJyU5_BIa6qJ51-eGefJqsSxUO132mJjioisTFKsoF6-kRLWk4DBQIQ6pUIei4vyjSI9lH727yNY2elUznUfwidXpOh9jm3sVz-fOL0MjzxgDHTna29ew6LeLkKzuKdtbiVbQMNcu43gheNpHptUrFI8RzfbWqGSNciTKw0A-QCj0=w614-h540-no)

The dataset contains 5 columns v1, v2, unnamed:2, unnamed:3, unnamed:4. Only the columns v1 and v2 are used, where v1 is the dependent variable and v2 is the independent variable.

### Text cleaning 
`corpus = []`

![](https://lh3.googleusercontent.com/ukBEEl7z3MeyVd0rSV8PFY2lgoXTc7UnmiZLaN09je2WHlIB-NOL40_4A3NI_3SeSNxjWCarOQX1JW7ZUgXvZpw-uPRBDlCEwdSNMJpzSdZZ4tHUMYH64mN5nCO27aRW_aL5_ygjgg8mcp0JEXAEg_TxzR0XTj4ttf85zDNASWd8YyOH6XmFKji60DIpeqsbtjUeFJ6-acBP-tZfpUBlJ1d8VbDZpcmMpPvhUDwGj70jNexYjXCSz6VVRe8mEestaa2QqICmncVFf_kLxLBJswqGFliSwCdcGlwKCy6yS92IWEkv0cpLn63b20z680ETWKp_h4WXo8y6WPnOkLpfZNrduiN5ucWunjVej_2Qa6U2jbN8u6z4ZJR1eUfmnjesD5Xx58TI7g5LXqfd-8IDCuT1_x0GI-IaUmmdDTpwLRxYPC6ickV4taqYk8MRjnHHDZH-deMMHVjKoV9kRaTAaQFw2PiNMc50-ve0k7pB1zK56n6jcFLgxLu7aI1aYAd8OR1YhY7x0Zxm6-_Hv6JawEeBSg11pbQZAMV1X3jwa0PspjGZb9R9BWlgD04pVO9wCn6U4E65kiDDU7qZaFwxKHqyWLdviPAmoPpEppYsbiH7Jbk2saaGteH3BVKbFOQ=w514-h180-no)

A corpus is created which is a list that is initially empty. After performing text cleaning, the corpus contains a huge list of text without anything unnecessary.

`for i in range(0,5572):`  

Several operations are performed on text dataset. A for loop is applied, hence these operations are applied on the entire list

![](https://lh3.googleusercontent.com/seLb03aJ48hHG3b5RaLS-BCWknS7z46KMd2UzSlujlWAIRuZchlBWeswBpeb-L1qJJSdI_RUOe3vqEaB1SVATifqyMabgqYAbozuxdhSMI6AYlJ1nMqTP1OrlZsfcTnooxblyKriNxm5yWBRGReYOZ19j17uxdo6CY1hTMw-YfBZmw_SsfZ_obWdf9rv4LV-SLAjFgiJSQQhmCK9HGQp4gEFUEOcuCcDkNe_jihNr0N_P4ZZfqRYLs5KgMaEwpjobOWU2IVgq6FXcj7J_AOEdSdT3untRcrLMrcohORbIw5Jh9QD5qXJ0q8jsq1tNDoZhoqqdVBqWi-AMtoq32OFDm7SocqlRoptkBi-FW8-xvqC-PrMFzVMlOmRGeZLxId309VBORGdRpHBmAqc_N8afu7BWOBKhcwwKw80TTrqq1HvsZPVd81JKqViVKO4UYYiXPr0eO9ddvPgWheRTHe0KPcwmkff0JPvaubi4b-q2RXNyqcemmkHB32c3fXiCBXHvBxOa9Uds2QMW2OieiZXXzqeF2KApcvXyKb_1L4nHdkJ_1iQdaKjWb3lafp-0GA_fTMlAcqDkfNALjxEt2Tpr_wvuf3A-zX6UPlGcZtyoxk5S729Y3RSeMbdWrakYRk=w394-h55-no)


The last element of dataset initially looks like above

`review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])` 

![](https://lh3.googleusercontent.com/0rCTHY65DmmqJymb69_tWlbZANdUattCUiRf9HdUvlzJh4eF5z-2Rmsk00vFz23FKous1N88X71f6kD5r-ASibbv5y7oljXxwT_O-4kf7jnY_LBDdfnzSmPobjtOuLntELoAdNIdptIBS5727XSuAdIa8LaKZHjFMEGVnG4cLjuiSd7wMJv2bajAk2fD1jPw7G3BDHa1naKDj0X3hPF2fNs4824flGBULbcL3BeHuNxepe1iKDvze9c-lwZv2Qa9dlDnY65qWJlMjdqKpOwIEiL5hEDSDrsn-0HpGg4kDyOqaENaTKXOnkLSnd-wKhg-KrLlw7o9cAJac_42HnnxBnu5psVZH5RbMdmboyxhXPpnkA-NiyXzKM8C5sKrPdSWSpC0Zr8JLobvTf2VsfLOx5_3__pbtFHN76CrywfrSz-__1ia7mWRWUfOOL4DCcr_3Bbb6lXaa0e6klq6mfswYaGlf7NB_pQfKT_Klfo9aIooTH407LvF4kcUaMuma9Iff7g_sIM4EWw8rIDqjobCX-xRp4k_vYSYuBzH3gttq9iqQJF2l_NCEQ4N-aGbhFHJ0tpuBhHXmsoTA0hE02l-AORx_qrWRDl1y3-JjjMUlmBICzbHdV3CbUyOh2glp6U=w387-h50-no)

Removing all the elements but the alphabets ( lower and upper case).


`review = review.lower()`  
![](https://lh3.googleusercontent.com/FxLcTZ90ObrzVveXzrC9Fojq60jmDwOKSd66cWTSSyf2AtRzUFEY5LFypBKs6I4clgyy8shRYo89mLypFe7oKyyLIlA8HLMDitKbIErO5-xjlr7FGKSFHH6M9oe3FkKKTgBVJhhEi4jw_C4nTp_JyjISBm5Xzl-U46q8BugBdUnLkr5DqEAAX3B88GvJLvmBxKrO6MOX7puZJNcpODTEhFSUj6h8qGOMWlD7FggmXep23BfykA9HIgc6YI9CIqcFoZ7dzCjQNe1cdCpchYGRwp5qcKnWRocnnge1rmYvPejf6gg74wURXLWPWHyvBJwx0mbPsSLuH1z9lrV0atLF53dKwUpTbR9feM6DrhJZuHJybNLKCJEvNIxsYvnR4ufvr62agdfVDcUGkF2UzUXVYIwIJcnKPVrfVpyeU7F3ubMsq5Ipm0hjfXK2syqigyQ0SOnUE3p6P0mhKtPdiJ49GFrkAvcAuvNhod1Dsy33pugmBHvxFEK_rk7tKJbzDDtDdBl1HKVUeCga4ZgnsvdYRAt_yP2OiBR5Cpvp3CuED59iPEMQ4AMS9fe0zZvmfxGeRGy1gZu-ifDOcnva-_-Y5LKwQMb4q-w_YNbmSZaQKgfBk3auE9i52vwsDC5qPYI=w404-h54-no)

Changing text to lower case.

 
`review = review.split()`  

![](https://lh3.googleusercontent.com/jyMarc4bRejUNQRTAqVPLXmPybZmprYPSwsNEuglLoSp9F3XyGpvHEVM_meh__s6g56wc4g42SWPOpFrpTm3McTGzBa0bhkoHHhAVFi2OAxDG-igl1hdePzpjwLDvkfOeGeez0tDDRv7nf3hO5hCI_94KMNgbEvX0C-RppEM5SfB43TCZAbMnyk6pgFzDikwpi-GrvWZoYhI2_eV-9YvIMoqJ7sZ7uqk58nBcm8WkjOO464qaigDPlfOLb-s74q_4cgYgTuMEniNtd6vYw7WyWNke1wIlACUyYf2dgg9UPWSPjGSrorB14_afqYrzJfn0gCiwl1Nl694HD7x4W7RnC6bHzyGKAAJwXEtrvgFxoHuExyq8pwKxlFHWgR7XS_vMCy-_V19py7MiptKvUfjhoLt1Is1-yxoduUGGpfoZgrZKDxmqy72S8nLBCSziptdUDASK-c1O4d4EnglDf-APrvAaP9VoQa3N3Qa4PhjzP-ciECx2kMRUyPIMMaBV22lF1Y_5mCMab4W5D0Tv--kWn0uCwfJTqFIpgdd9LsE5kYql1QHIlNqs2ZZfD12schaQ_K0mR2Vxjc4Uxwb-D7CHcNzjjQJ4WJalELQUAeOd-pTl5lZkmCdrRPWaT63Frw=w562-h58-no)

Splitting sentence into seperate words so, unnecessary words can be removed easily

`ps = PorterStemmer()`  
initializing an object ps using PorterStemmer class

`review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]`   

![](https://lh3.googleusercontent.com/nyJVRCwFXePn4LZEsY0k2y9sC0PTZ4B3wofzVSZRICRgrtwxPlWDa3mchfobbbJ1-vNCqY_lyZAbGewrNuKtVzDeh0tOGe1h4xIhi7TP3y54o0Uux7R3_IuxNrz68-vqDVdxopXRKhSI4NXb6uaiVjgLlVhTv0p14A4wZg4LxpsZTD1dkp00FDfhZFFF8kvuoomg8OGUhRsiArGPLq47GDpGmgVBJ_p2yodXsCkhVZgUvE_aEGwIa4u7GrLqcji71CENfm6ipkL2sUN0hGe4fVVbhJ5d3edkx6g5CdIIV4N0e3os5g-_3YZsNxZR7hs85Ps_hS5MTKTVUIzmB9w8aG8YRjpxdP1veC4396IArxvZUA4tfy6cWYmHGuFVvXqiKbLmMKgUjKZna-T_wVXWeLke1WTAJY2CZyx83c2DuO6Uq4-3CtgnVZQR0hJF5nch03IXtI5LvVnhy8v7kbvhLvA9iyBAiPCZtJBihlvUQnV3Hzeasyvx-lWjobqQoRwkg5jjscEI1VxdvzUbqVmCGpoWvr_PH2UENMc5tomGTvEwXfeqlEMqwsG_AINHTj8rN4dcZv93OmfOS89c8kwoPnFJ_S4WkkIbm6jK8otaA0VuVAEjRTTUoh5AQBiecuk=w364-h64-no)


Removing all the words that are in stopwords list


`review = ' '.join(review)`   

![](https://lh3.googleusercontent.com/PPV9YMGZXIWt6PDuGDkUo_RMepPG0Y0uI58IFSXkeDLLDiG36Uz7DqRKvJB18owhxlO1FRDHqS3G7J2ptBM0mS4PnOBD6z1UdAzRrt0mi4mf1P257g1SwupSwdKoRn7syQ2mlQk8YoYwwJ2suDOzG5L295fUKVHZBJxyyP_WpHJ4dOxKT-SGP_3p7i8xNVFUiLXzW3htd46mKWbvG-JWnqeoo9Eac-0mivlEpS26j3qJjKydGndHnN2M9ZS_iT5VOESpvXZT8hy5MZfygwnR7ghbN9_msYnaC8fAz5c7c7lZJW0muE_XjzhmQoZ-G5CyYButABZT3IxQK7w4VMJKFscN_r40trfVJXf59zBX0vwhcAFa3W4MdWGYI5So2xoXmduQ8TzxqYX2PNfqCLAf64y-rUiLLPDzLcO-7oGhsQ0LXfF0LcTLUlMEvCA_GvDdOe41GQXh0vWdm_gbGgtS6A6Be0Of1r0ZX9aAF8ib-Ih1AStX01qAqtHVjWgnKJemj25xpJDT2gYcUritdBD1UmE8BxIv_27QFqhBtCHI2cz0t8LDUh70bNO9pr2xBS1OJuH5p9POfGshnBpa8LIZ3D88iphzm2ilMpQXnhkCtnCYgBfg6RB07xYVJepG8fc=w281-h47-no)

Joining the words to form a string of cleaned words.

`corpus.append(review)`

![](https://lh3.googleusercontent.com/g4uddV_z3wH25KyBwOvNG0m7GLUsvWtPdlNjbQk3E_8RE8FvZL6tGs2jG1j1jQVQCbsVp--8vm6lQJYm8slbfQXUefiRXfQ-9Juue23oWvomTYRnETVAZfrh9EKqiYVdY9qPvBlNo0rIK08O2jtkwORivrQyRTfBVXj-Zu-4t6_Wo8U6_zNVXn9yoxivp6oVdBAH_q_kzoS7eD02QB57EUbVgfowO2pMgVijzidQ3sWCmzqhLe5vaH5vCtsNvR4s_egO1u7ozaT9JxU4Lt991VsxJ72lOGFulLvnUrkQvtOKByPdFFvC301u__lrGQoiOBos0eYDROhUF7k82APw4ddWXfAhB1noA4nVOIgViULfdBYjUUlb27Dw_oLf3gzlx-Oc0M1LxoEZbV-u6_DJlh1fkNCj_sZGCyBS___NLVfszQUqpwSyikWkzveChJouyFe2_lgf5T5sjsDIEkkZnN-B3Au9TAZVBMPhLNEZhSY5ziflNoPOEsgx3FaQw0TKnRVxE00TuwULtqTkqtPJzyT4iUs_bZiXzUuonv5nhOnuoUzhS6a2aIfuQSebGtkJhVNlMPSlkTr0jMz6QPGhcv9N4Jnz3CW0PMMdsCngAhkb_pbmFReI8BOCZiEfMSE=w518-h628-no)


Appending the object review to corpus list. 


### Creating a bag of words  
`cv=CountVectorizer(max_features = 2500)`  
`X=cv.fit_transform(corpus).toarray()`  
`y=dataset.iloc[:,0].values`

![](https://lh3.googleusercontent.com/Q-DXmLVjQtPrX3N72I6alrNZ479ExP5_pa1ZlMi4FThFB-rBvwMC0z4SlNMkjdN0noP7cP_kOssS_eBeNnL4teZI6YvQu8JJkxehoLuN-fbxdIycDnr4HAqktbuoq7DqvrUAEqBm4K1gtHo8p62giAmOP2yeNMm0mG5vH1of92pkrMRY_aA6bGqAFGEstiGM7v-8XezsyaHicOHA4m_FOR9vbI6sQTJjC5s0kkIwrPR5_X_XVuGXSDJ30g47N5lsywhyFeaMlfQH4VNv4VoYbc7fk4ni6tUDRKlQ17IwAQPIb8TgAzbAOMG-yXyyqNurYOgh4ZauxrLl3wZGPE5yQ6g2EJVDs0yV9XVaUVEv8_xCGCuy22XjYW6QKAAz16SyCiPckMnGkBXANfjroOM6iHyc9jyTphl-pQ_JHFdd0h0uo7yZl3vw24YMWEMSCJxiRA4487Ps1nr-0jdCUdDDp_68IroNxcMCIhlZyTyGt8plVq7psbRy32cb8ZvejfvQExM4X9Avo_TZ8AoPc0rc2APa7YBApvyJOBqEmvf06MbuK-sd2qYwb6apNaqFKxXhA7FmOhnTwNU4oWLchnUYaa_OSgMUrCTRSGFSZCLDzplS6L5WI8sPq84iyY0QF6g=w752-h30-no)

Creating a bag of words (Sparse matrix)

### Encoding Categorical data  
`labelencoder=LabelEncoder()`  

![](https://lh3.googleusercontent.com/61_jFDpnFx5gfyInTjKEw5nI6xTRSfRRtwWe19hf05tlvkphxR2njbQB8okUrIMi-hcmzvIRljW5V1ECTrd6zilTLvaQdJJTRHNbRuUJVBKkERduE6ynq22alm7tw7OSsWkB86pLoBVkVDpvvWG8_BCqFVTFYJfMcadbGI-yXnvyS7ptIqoFFXbF_MLmci4OPBGodqorB2QIV_aDtT95xUwLLovzs7kC8_ISuZrvp_jP5h73HBEuDM3lGmin6q9qfSrxVaGZt6CPQtjB6t60nF5CajMSb3tdPj-MdBozjGnYEmGR20MT7buk5ihvFZEPB7bpN8XJt3XQvBypzuM8qx7ekuYoqK7uCAYBf59ZnYA3gOTslxWewLYwPtYnOL1Q3v5IsBCKHWh8BSrdZzE-sFH_7S2KlmOlS1WUtWmbbYrQnbpnSwUH6fwbm1fz9HR_u0K-iFREDb9QoEcX5E4V3Mwy5fbYAPT-5F67uMqqm7cG3ETVL-o-sjaPGVeu2faLc6LPajYohJMVVM4iEEkyli49VMLPWtXoYaiiVvFTTUZGPmSyP8Y1pSfAaWG2LqmgiCTZaDu2wpFXsWF3SNPPPA0SwX8FfWwKa6C4DlyPndteCXIKPJhu5UqmFDKCa8s=w896-h27-no)

It is important to encode categorical data. There are two categories in dependent variable, Ham and Spam.


`y=labelencoder.fit_transform(y)`

![](https://lh3.googleusercontent.com/bFGfbuzAV-X8-5d7lyxR033MHcAWhoWTaqgmP1iIauuJMcCHxGZ2yZbcnfHVnB4geJ9FLuJKcuv_oJ6_opbVGKjK2rhuj2s8SSkb3EwMByrB_8zmW7VFTwwC4FBYAzqJh3z7WAdkwbyDMzPiweU_YYj-oJwzwfBPWXuriOXGgGxO8wIIQuw5e7Ohcklwgv1GMEsskdUigpp9N-7vSsPx4QSbexXV99mBkRWPbbKn9ZIVxf0OJ5Cg2cf3WfJlKPp8MQfrsGbh09jlbZiKVfeBk4RlA6e3AdbQUeqw8RKNkL0_TqAqnZucjvFxSSQYkKR_vDzZc6yBdzSRbOQU_lpz4VPpmvkPmz8WTwfYA0xjqCNyao4kl-8yjUeMyPpXZ068dLdSeEYp4foiG4HjTgH0j7HvYvq1laOJJY8RlOXZ3Kbl6COAFFJMVX--SvnmUF9h82nderHEo4nnIumcQ1-FkJrHKS3dCcypTQ72bkAS6QKIACrhE2R-Al_5Xv0rZDq8dDiwywb1H4yGNzWDcH9LMJ73T2J1tQ_RNOg2DM6WUHXeesPGGFELFmPyIX0LISHxKUZL5-4cb2yIzk8jhqJ3p1uqGETbDB5fDBs53xI8hi_86SH78xlGx9-j6C41Yr4=w612-h537-no)

The label encoder encodes ham and spam to 0 and 1

### Splitting dataset into training and test set
`X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state =0)`

![](https://lh3.googleusercontent.com/HRCwxIeHH7y6A5LupYoVP_MctG52xfCIQwABPAtmNqaf_zfj_rw9ct1Ct03XayfIAD0vFZchf1rEpS0en35sSLHyISJErbT3pWRwmgADJJ6_lEIgVxRZb4hZ1K2NCoCFW7bDNiYeT7D7MEoeLIovhzDPVzcwlLZrPB8xurTeXWpQnQ14UuNrSNujuPRvl5S6VATbHWs2bIT0B86_sZNgHSK8OxclASqp-U6wis0zS3gd8YHW6ljgE3z2Ngy2Yidbg35zl4GEo6St85tr3ex2UO57zm-tLEWqKyifJn3bBXNOq2afEPY0u__FzboIORx0Q-eNFE4RuLiMhtTZHzw-TtAOjGRXxI_BhD7X7OXtqsj14Wxs-i1zw_JyzLABly7moybjx2xr_l8z1o0FHlGx3ndX8UeA5t3kjqgxpG4-lvR_0VCWovIGYMAgNF-6hFhZgLOrLJrZpLO3B-5zp9X5fJiTi44hnjdVeokd0K3vfwAhsiAzOcGA_TpgTjTUjAE5MVdfwUyvImA3la-eCZjOFZeyuSdI5qqTkcEdt4lrra_hd5TfeNwy0uYzYdsgZbvW3iAitRYw5xBH5j1YPoRGk_WDW_ZiqNa1WAVjkQxtEXa9NaaMOaOlP_vPpX8-vY=w904-h61-no)

Splitting X and y into X_train, X_test, y_train, y_test with test size = 20% 

### Applying Random forest classifier  

`classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs = -1)` 
`classifier.fit(X_train, y_train)`

![](https://lh3.googleusercontent.com/j2WP-w0edx6g-ZQSgL_xGvLtyuJ7JY3H-FxZWJe7Yjy9g2WfUw-wB8t8q6utrdQmLXLfBDLBpq5--bWi8gP9KFUNNodmAY1NM2adCvxZj2SrnD3KvhK7RyJZrGfO-4s-RTnJYPn4GNoqOi3Bqpww7An7SNa9NVC1iHM_mQHA7BI0fRwxMlqefGti4tueawlQyZ0culX-wcsworg5xywkOkMPiBffnWyZlJ1g1cFI1sbAf8JBi3y3VCXDyQOp1hYxk8M3M5wlaOx6eikvOB4W1x8QWBsSOJSzXyB0OEJJdebVR9pls2FFCfSyYoXpeMwSZxXHhra3o0dwd2tYDqJ_VJdol1VD1dFFvpW3nkjWLthvAeyCuV-p62ipWqR_8yZlWicTjo5mK8zFW75BpRDunECEwF5bPso6Muj0DsUb0vD9KGAo9d3Qrf2samp5D4DHhSndsCfrfOiga8LKov5xP45X29YkxVPioQ_BDQM8uUC4KbQwy2GL2EoxsAtSCDNE-606o5MggsBpN-JZQrInvLcoBzSMGwCp3vVM6aCpfTDpENxVOmbt155XG4npLR2TzdVrnf2yh3IswmXb4h3q-cKoD5XjToCS1fogQGPA2ZzsA-wLVfaEOt6czhkjND4=w824-h172-no)

n_estimators = 100 is similar to having 100 individual decision trees, entropy criterion is used and all the processor cores are used (n_jobs = -1)

### Predicting the test set results  

`y_pred = classifier.predict(X_test)`

![](https://lh3.googleusercontent.com/mc7iKosHgGfXpfvjMoZiXoYeubJUO6Gck0ch-e_FKZW9fguD2jnZb8U73usJv0e1igvJqJso9H_qc_WQm0P9Hd5Kijx6xZIa-sq902bKRbBI6OOMzCg7Z5L74IiEipoSWkprHAWn_EcKu7ouCbwPqXFYSY_gp03OOgNdwQhNblHeLfqgQI8CmJ4n6qntw7Xaes6iZuB1Fuvw-5EHOftMCoXQ6oAHumg4M0QoDiP2asrDw6NomCLgN9LCUyfh9b6-Pb3lR3l3mdExwc2ge1Q8baiqjfj8zXmR7xwdTg5DXiqufYHb_OfS9h43TjlCrNIZC3AmY5jmmGn_Frh-OYNuwXallSEz2-kWi4paM57DfFYW3CGZcHkaQqubJd-4k0VcDXb7k_cQE9DwDsfakdBCWgvxCwJduPczi_i1Eo1T6ARN_CX3OKgr5BWa5yW-jgP85r6PPC09bYxefET872s_6DBrsNCa3GB6OSkv9iB2ecA2pzc0EfMkLQDsAYtOCxSfOU85LbcxKRdYqwN8iWlVLjYNpLQBhdFAZlus6zeysMaqAlGQEQcR1axAF9hhbeooPE6Z4R8xoJWDgDQwYydZ5mx4DMCQ-TDVP3HSI3gGNthw0FCtU-q6Y3l3qLemg1Q=w612-h537-no)

Predictions of test set



### confusion matrix  
`cm = confusion_matrix(y_pred,y_test)`

![](https://lh3.googleusercontent.com/XZL8Mj9B9xjWskl7iubTlYURdWSGuwA8UnupwDiB-e0oWOGSWn-3JFPs2pyisUjNiONJ0OCZr_qghKCtxfpfq8E2UM7OpBdep7009hWOkeLWdpZX2RroguR5uzVtj_oVMsHTMLM5sUQqkL85l6HSOi_UahbCAvvgcMZRCxTTV19YHcYOO6BuaNpmw2uFeCJ5iuBFB-AAXLHJZibLa_cZ7nHSx3VEdPwEyIAhyqwce9KakVFZo-YoQ1_q3DK2eladj9bkPzyJaczRhSH6DW8PoA6TJTLW88HYOLm2ZcQ-Kuescb486l7UMTr5Mx_2lDhw2mCdcn8CViZ3M1jm832YtyEyzXouzAHawdApwm-gFWcr2-mysNm0ePNul2ED_7XhElSk1ZcMDM7keQ6M609ajSjdr-ab3u6U5gf7ostTYIDwoceYK-BjloL2MZoQ4tVMZhuT__c2FXvp0fo6F5KD-wh_Aox3a5cuGfztpTUCna3ahyX-H9s7Z1g8PiV2l1jCPPEzgKqZFcvISGrSK7nZYh9HSsTvAfvaYModtQ3rlnH-3v_i0jTNk9cHM4JdsT-kftwX4FWytEWgWW16maf8wB9h4Qu3DLYWYFi2eHy-7AGxjtlEn-hMMLZCp7QDt60=w608-h531-no)

The confusion matrix is used to calculate the accuracy of the predicted categories. It contains 948 correct and 1 incorrect Ham predictions, 142 correct and 24 incorrect spam predictions. An accuracy of 97.85% is obtained.

![](https://lh3.googleusercontent.com/3CvxPEJMgxm38GNYE-bxxXZJmfNAtHeh-Dk-fHKJkeYTg56MmRO4cECswYw31quq2qaw51xsC24LWal8BbqS2jv4X5jqmzkQyHbKhaGx5mzFjXAL2jYWAPJHOGhHQyAX3QvZrB-iwJIRVJbVtuV0PHK-2qd9WDRF82uszCM0jONknXYbNYiFtKKRbR6oD1TTlJK-9CiepnqaWwXIX_W77y4x2lNG2qpEFSg-6G6XHJqd1FtuSI84j8RieNH08wfefBxVaE8mEz44Apuae5hVzAczIpvTCiDeNuEZM45uo1ZLVZVOqXyFhjCAF7e8EfuXUCk8FLO2FlLV4OPG49aMA-tgLbTNXGSQWowYu9e6Y6QQ5hxMySLzDT8AP0zEsiBalVyqhNyUq12Epew-tbk9mG7u4B35TnCS4eHzAtrFWEiO3odSP_hGdHjbCrR2BVDH6VXbKTnFGzNXUGIXVOU1IXcz4ehZ6SFoG6fMT9hJOiMq_Bo_jEujsRe1hQKrgjyRI__k4ReSOh3GFF8kHuvVgYE4DJMYrJ4opVdwH1fm8Bx2n7KlEiO7R_u3qqnLJ6kvfJzUUE4wnaEgQ6ZhfW4rhK0DQskLGaAZsx4V9vI5OXFKVv6NFEopd_SBAKXYXaI=w306-h45-no)

The accuracy is calculated by (sum of correct predictions)/ total no.of predictions.

### Parameter tuning   
`parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}] `   
Using different parameter values of random forest classifier

`grid_search = GridSearchCV(estimator = classifier,`  
                           `param_grid = parameters,`  
                           `scoring = 'accuracy',`  
                           `cv = 10,`  
                           `n_jobs = -1)`  

Using accuracy as the primary factor for evaluation.

`grid_search = grid_search.fit(X_train, y_train)`  

Fitting grid search object to X_train and y_train

`best_accuracy = grid_search.best_score_`  
`best_parameters = grid_search.best_params_`

![](https://lh3.googleusercontent.com/Ergj8Pvq5RLM_3HmatFHixtO39nkAovDar5FsXB4v0rj1ApGmowSAhrL97agg4IMh0-nja9yYyHjrwlE655bBJiLAdy1tQOl_zPOEjSsjBUNsqWf-cvvJqEyfVgNFKj2GV-veHcA8hdsSNGxsEp1z87X9EEzzBDq7mcpnCBy9gOzPFdpCLz3BEXV7I5KsbC_kULx9DgUSgU-vcU7dg2SU8-YyL8I_Izzuhue71cd5HwwvHNnbpzUNF9TxEHxoRXwiLAuaSe4jpn4MzMy5RE3O5VVw5BVfS6GumoZ96yCYS5Tco9tHk8kERmut6FXuJUhIAhq19bLIDGWlim7rAMHbeH0xxZ_EtMOmdLBCoYuNJEyuXVqVq3wf9pCPDg0eQfX4I0vJNDSciQq2sCXkquvN9G40v_5SPef8HeXGRx8fBqJiLqVS8HZXTqKOFSJrLfYy5e9SXvKbEtddZKKOhfbOkXA0hOuoq4SYv1ZCztGOUM3brglzfzfafH9zIRWbZ5JjJGhk5Owqr_f7_XMtex-tJuA-T5awUOLz-cDlhXXk73wisV329QQiLjFEYhOX_i1GPnXkdAMKrotw610j7-PjiGcnGfqS3Gz8EFXwG29OVDt2I9s4SGdKZQnp5NYXyw=w911-h55-no)

best_accuracy gives the best accuracy with the given parameters, best parameters give the best values for different parameters.


