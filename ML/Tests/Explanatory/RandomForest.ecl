//RandomForest.ecl
IMPORT * FROM ML;
IMPORT ML.Tests.Explanatory as TE;
/* 
//Tiny dataset for tests
weatherRecord := RECORD
	Types.t_RecordID id;
	Types.t_FieldNumber outlook;
	Types.t_FieldNumber temperature;
	Types.t_FieldNumber humidity;
	Types.t_FieldNumber windy;
	Types.t_FieldNumber play;
END;
weather_Data := DATASET([
{1,0,0,1,0,0},
{2,0,0,1,1,0},
{3,1,0,1,0,1},
{4,2,1,1,0,1},
{5,2,2,0,0,1},
{6,2,2,0,1,0},
{7,1,2,0,1,1},
{8,0,1,1,0,0},
{9,0,2,0,0,1},
{10,2,1,0,0,1},
{11,0,1,0,1,1},
{12,1,1,1,1,1},
{13,1,0,0,0,1},
{14,2,1,1,1,0}],
weatherRecord);
OUTPUT(weather_Data, NAMED('weather_Data'));
indep_Data:= TABLE(weather_Data,{id, outlook, temperature, humidity, windy});
dep_Data:= TABLE(weather_Data,{id, play});

//Medium dataset for tests
indep_data:= TABLE(TE.MonkDS.Train_Data,{id, a1, a2, a3, a4, a5, a6});
dep_data:= TABLE(TE.MonkDS.Train_Data,{id, class});
*/

//Medium Large dataset for tests
indep_data:= TABLE(TE.AdultDS.Train_Data,{id, Age, WorkClass, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country});
dep_data:= TABLE(TE.AdultDS.Train_Data,{id, Outcome});

ToField(indep_data, pr_indep);
indepData := ML.Discretize.ByRounding(pr_indep);
ToField(dep_data, pr_dep);
depData := ML.Discretize.ByRounding(pr_dep);

/* 
// Wont work with the largest dataset, delete " , ALL"
// As well as further commented lines will ", ALL"

// Using a small dataset to facilitate understanding of algorithm
OUTPUT(indepData, NAMED('indepData'), ALL);
OUTPUT(depData, NAMED('depData'), ALL);
*/

// Generating a random forest of 100 trees selecting 7 features for splits using impurity:=1.0 and max depth:= 100. 
//learner := Classify.RandomForest(100, 7, 1.0, 100);         // GiniSplit = TRUE (default) uses Gini Impurity as split criteria
learner := Classify.RandomForest(51, 5, 1.0, 35, TRUE);  // GiniSplit = FALSE uses Info Gain Ratio as split criteria
result := learner.LearnD(IndepData, DepData); // model to use when classifying
// OUTPUT(result,NAMED('learnd_output'), ALL); // group_id represent number of tree
model:= learner.model(result);  // transforming model to a easier way to read it
// Showing only the first 100 records ("result limit" is 100 by default)
OUTPUT(SORT(model, group_id, node_id, value), NAMED('model_ouput') );
//OUTPUT(SORT(model, group_id, node_id, value), NAMED('model_ouput_all'), ALL);
// To review the whole model use following line instead:
//OUTPUT(SORT(model, group_id, node_id, value),, '~user::rdnforest_model', OVERWRITE); // stored in cluster

indep_tdata:= TABLE(TE.AdultDS.Test_Data,{id, Age, WorkClass, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country});
dep_tdata:= TABLE(TE.AdultDS.Test_Data,{id, Outcome});

ToField(indep_tdata, pr_tindep);
indeptData := ML.Discretize.ByRounding(pr_tindep);
ToField(dep_tdata, pr_tdep);
deptData := ML.Discretize.ByRounding(pr_tdep);
//Class distribution for each Instance
ClassDist:= learner.ClassProbDistribD(indeptData, result);
OUTPUT(ClassDist, NAMED('ClassDist'), ALL);
class:= learner.classifyD(indeptData, result); // classifying
OUTPUT(class, NAMED('class_result'), ALL); // conf show voting percentage

//Measuring Performance of Classifier
performance:= Classify.Compare(deptData, class);
OUTPUT(performance.CrossAssignments, NAMED('CrossAssig'));
OUTPUT(performance.RecallByClass, NAMED('RecallByClass'));
OUTPUT(performance.PrecisionByClass, NAMED('PrecisionByClass'));
OUTPUT(performance.FP_Rate_ByClass, NAMED('FP_Rate_ByClass'));
OUTPUT(performance.Accuracy, NAMED('Accuracy'));
//AUC_ROC returns all the ROC points and the value of the Area under the curve in the LAST_RECORD(AUC FIELD)
AUC0:= Classify.AUC_ROC(ClassDist, 0, depData); //Area under ROC Curve for class "0"
OUTPUT(AUC0, ALL, NAMED('AUC_0'));
AUC1:= Classify.AUC_ROC(ClassDist, 1, depData); //Area under ROC Curve for class "1"
OUTPUT(AUC1, ALL, NAMED('AUC_1'));
