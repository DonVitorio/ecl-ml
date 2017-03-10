//RandomForest.ecl
IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Tests.Explanatory as TE;

//Medium Large dataset for tests
ML.ToField(TE.AdultDS.Train_Data, dMatrix);
featBuck  := ML.Discretize.ByBucketing(dMatrix(number in [1, 3, 11, 12]), 20);
featAsIs  := ML.Discretize.ByRounding (dMatrix(number NOT IN [1,3, 11, 12]));
all_data:= featBuck + featAsIs;
attNum := 15; // Outcome is the last column (15th) of Adult dataset
indepData := all_data(number < attNum);
depData   := PROJECT(all_data(number = attNum), TRANSFORM(Types.DiscreteField, SELF.number:=1, SELF:= LEFT));

minClass := JOIN(all_data, depData(value = 2), LEFT.id = RIGHT.id, TRANSFORM(LEFT));
majClass := JOIN(all_data, depData(value = 2), LEFT.id = RIGHT.id, TRANSFORM(LEFT), LEFT ONLY);

// Folding dataset

IndepFolds:= ML.Sampling.NFoldDiscrete(majClass, 3);
// OUTPUT(IndepFolds.NFoldList, ALL);
train_ds1:= IndepFolds.FoldNDS(1) + minClass;
train_ds2:= IndepFolds.FoldNDS(2) + minClass;
train_ds3:= IndepFolds.FoldNDS(3) + minClass;

// Random Forest parameters
numTrees      := 25;
numFeatSelect := 5;
Purity        := 1.0;
maxTreeLevel  := 35;
GiniSplit     := FALSE;
learner_1 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);
learner_2 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);
learner_3 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);

mod_1 := learner_1.LearnD(train_ds1(number < attNum), train_ds1(number = attNum)); // model to use when classifying
mod_2 := learner_2.LearnD(train_ds2(number < attNum), train_ds2(number = attNum)); // model to use when classifying
mod_3 := learner_3.LearnD(train_ds3(number < attNum), train_ds3(number = attNum)); // model to use when classifying


ClassDist_1:= learner_1.ClassProbDistribD(IndepData, mod_1);
ClassDist_2:= learner_2.ClassProbDistribD(IndepData, mod_2);
ClassDist_3:= learner_3.ClassProbDistribD(IndepData, mod_3);
ClassDist_All := DISTRIBUTE(ClassDist_1 + ClassDist_2 + ClassDist_3, HASH32(id));
ClassDist_Acc := TABLE(ClassDist_All, {id, value, sumConf:= SUM(GROUP, conf)}, id, value, LOCAL);
ClassDist_Acc;
ClassDist_Avg := PROJECT(ClassDist_Acc, TRANSFORM(Types.l_result, SELF.number:=1, SELF.conf:= LEFT.sumConf/3.0, SELF:=LEFT), LOCAL);
ClassDist_Avg;
sClass      := SORT(ClassDist_Avg, id, -conf, LOCAL);
finalClass  := DEDUP(sClass, id, LOCAL);
//Measuring Performance of Classifier
performance:= ML.Classify.Compare(depData, finalClass);
OUTPUT(performance.CrossAssignments, NAMED('CrossAssig'));
OUTPUT(performance.RecallByClass, NAMED('RecallByClass'));
OUTPUT(performance.PrecisionByClass, NAMED('PrecisionByClass'));
OUTPUT(performance.FP_Rate_ByClass, NAMED('FP_Rate_ByClass'));
OUTPUT(performance.Accuracy, NAMED('Accuracy'));