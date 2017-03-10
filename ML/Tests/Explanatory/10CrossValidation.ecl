#option('outputLimit',100);
IMPORT ML;
IMPORT ML.Tests.Explanatory as TE;
//Medium Large dataset for tests
indep_data:= TABLE(TE.AdultDS.Train_Data,{id, Age, WorkClass, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country});
dep_data:= TABLE(TE.AdultDS.Train_Data,{id, Outcome});

ML.ToField(indep_data, pr_indep);
indepData := ML.Discretize.ByRounding(pr_indep);
ML.ToField(dep_data, pr_dep);
depData := ML.Discretize.ByRounding(pr_dep);
learner := ML.Classify.RandomForest(100, 5, 1.0, 35, FALSE);

cv:= ML.NFoldCrossValidation(indepData, depData, learner, 10);
OUTPUT(cv.CrossAssignments, NAMED('CA_part'), ALL);
OUTPUT(cv.RecallByClass, NAMED('RecallByClass'));
OUTPUT(cv.PrecisionByClass, NAMED('PrecByClass'));
OUTPUT(cv.FP_Rate_ByClass, NAMED('FP_Rate_ByClass'));
OUTPUT(cv.Accuracy, NAMED('Accuracy'));
OUTPUT(cv.AUC_scores, NAMED('AUC_scores'));
