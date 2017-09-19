//RandomForest.ecl
#option('outputLimit',1000);
IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Tests.Explanatory as TE;

// Reading and pre-processing the original Data (bucketing fields with too many values)
ML.ToField(TE.AdultDS.Train_Data, dMatrix);
featBuck  := ML.Discretize.ByBucketing(dMatrix(number in [1, 3, 11, 12]), 20);
featAsIs  := ML.Discretize.ByRounding (dMatrix(number NOT IN [1,3, 11, 12]));
all_data:= featBuck + featAsIs;

// Spliting the data in Independent and Dependent datasets
attNum := 15; // Outcome is the last column (15th) of Adult dataset
indepData := all_data(number < attNum);
depData   := PROJECT(all_data(number = attNum), TRANSFORM(Types.DiscreteField, SELF.number:=1, SELF:= LEFT));

// Undersampling the data to alleviate the Class Inbalance
// Spliting the data in minority and majority class data 
minClass := JOIN(all_data, depData(value = 2), LEFT.id = RIGHT.id, TRANSFORM(LEFT));
majClass := JOIN(all_data, depData(value = 2), LEFT.id = RIGHT.id, TRANSFORM(LEFT), LEFT ONLY);

// Folding the Majority Class dataset
IndepFolds:= ML.Sampling.NFoldDiscrete(majClass, 3);
// OUTPUT(IndepFolds.NFoldList, ALL);
// Creating Undersampled Training datasets by appending the minority to every Majority Fold
train_ds1:= IndepFolds.FoldNDS(1) + minClass;
train_ds2:= IndepFolds.FoldNDS(2) + minClass;
train_ds3:= IndepFolds.FoldNDS(3) + minClass;

// Random Forest parameters
numTrees      := 25;
numFeatSelect := 5;
Purity        := 1.0;
maxTreeLevel  := 35;
GiniSplit     := FALSE;

// Defining one learner per Undersampled Training
learner_1 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);
learner_2 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);
learner_3 := ML.Classify.RandomForest(numTrees, numFeatSelect, Purity, maxTreeLevel, GiniSplit);

// Building one RF model for each Undersampled Training (RF sub-model)
mod_1 := learner_1.LearnD(train_ds1(number < attNum), train_ds1(number = attNum)); // sub-model to use when classifying
mod_2 := learner_2.LearnD(train_ds2(number < attNum), train_ds2(number = attNum)); // sub-model to use when classifying
mod_3 := learner_3.LearnD(train_ds3(number < attNum), train_ds3(number = attNum)); // sub-model to use when classifying

// Calculating Class Probability Distribution of Testing data on each RF sub-model
ClassDist_1:= learner_1.ClassProbDistribD(IndepData, mod_1);
ClassDist_2:= learner_2.ClassProbDistribD(IndepData, mod_2);
ClassDist_3:= learner_3.ClassProbDistribD(IndepData, mod_3);
// Aggregating sub-model results
ClassDist_All := DISTRIBUTE(ClassDist_1 + ClassDist_2 + ClassDist_3, HASH32(id));
OUTPUT(SORT(ClassDist_All, id, value), LOCAL, NAMED('CPD_SubModels'));
ClassDist_Acc := TABLE(ClassDist_All, {id, value, sumConf:= SUM(GROUP, conf)}, id, value, LOCAL);
// Averaging sub-model results
ClassDist_Avg := PROJECT(ClassDist_Acc, TRANSFORM(Types.l_result, SELF.number:=1, SELF.conf:= LEFT.sumConf/3.0, SELF:=LEFT), LOCAL);
OUTPUT(SORT(ClassDist_Avg, id, value), LOCAL, NAMED('CPD_WgtAvg'));

// Add from here

LiftResponse(DATASET(Types.DiscreteField) classData, DATASET(Types.l_result) ClassProbData, INTEGER numOfFolds = 10, INTEGER classOfInterest = 1) := FUNCTION
  LiftRec := RECORD
    Types.t_RecordID            id;  // Instance ID
    Types.t_FieldNumber    fold:=0;  // Num of Fold
    Types.t_Discrete      c_actual;  // The original instance's class
    Types.t_Discrete     c_modeled;  // The predicted instance's class
    Types.t_FieldReal   conf_score;  // Score allocated by classifier
  END;
  LiftAccRec := RECORD
    Types.t_FieldNumber   fold:=0;  // Num of Fold
    Types.t_FieldNumber   trueCount;
    Types.t_FieldNumber   accTrueCount:= 0;
    Types.t_FieldNumber   foldCount;
    Types.t_FieldNumber   accFoldCount:= 0;
    Types.t_FieldReal     poprate;
    Types.t_FieldReal     foldRate:=0;
    Types.t_FieldReal     cumFoldRate:= 0;
    Types.t_FieldReal     Lift:=0;
    Types.t_FieldReal     cumLift:=0;
  END;
  dataForLift := ClassProbData(value = classOfInterest);
  FLRec   := JOIN(classData, dataForLift, LEFT.id = RIGHT.id, TRANSFORM(LiftRec, SELF.id:= LEFT.id, SELF.c_actual:= LEFT.value, SELF.c_modeled:= classOfInterest, SELF.conf_score:= RIGHT.conf), LEFT OUTER);
  totInstances    := COUNT(FLRec);
  totPositives    := COUNT(classData(value=classOfInterest));
  REAL pop_rate   := totPositives/totInstances;
  instPerFold     := totInstances DIV numOfFolds;
  FLSort  := SORT(FLRec, -conf_score);
  FLFold  := PROJECT(FLSort, TRANSFORM(LiftRec, SELF.fold:= MIN(numOfFolds, 1 + ((COUNTER - 1) / instPerFold)), SELF:= LEFT));
  aggFold := TABLE(FLFold, {fold, trueCount:= SUM(GROUP,IF(c_actual=c_modeled,1,0)), foldCount:= COUNT(GROUP), popRate:= pop_rate}, fold);
  accFold := PROJECT(aggFold, TRANSFORM(LiftAccRec, SELF:= LEFT));
  LiftAccRec cumLift(LiftAccRec l, LiftAccRec r) := TRANSFORM
    SELF.accTrueCount := l.accTrueCount + r.trueCount;
    SELF.accFoldCount := l.accFoldCount + r.foldCount;
    SELF:= r;
  END;
  itxFold := ITERATE(accFold, cumLift(LEFT, RIGHT));
  //SELF.foldRate:= LEFT.trueCount/LEFT.foldCount, SELF.Lift:= (LEFT.trueCount/LEFT.foldCount)/LEFT.popRate,
  RETURN PROJECT(itxFold, TRANSFORM(LiftAccRec, SELF.foldRate:= LEFT.trueCount/LEFT.foldCount, SELF.Lift:= (LEFT.trueCount/LEFT.foldCount)/LEFT.popRate,
                                     SELF.cumFoldRate:= LEFT.accTrueCount/LEFT.accFoldCount, SELF.cumLift:= (LEFT.accTrueCount/LEFT.accFoldCount)/LEFT.popRate,
                                     SELF:= LEFT));
END;

// To Here

LiftResults := LiftResponse(depData, ClassDist_Avg, 20, 2);
OUTPUT(LiftResults, NAMED('LiftResults'));

 /*
// Selecting final class per instance (Majority voting)
sClass      := SORT(ClassDist_Avg, id, -conf, LOCAL);
finalClass  := DEDUP(sClass, id, LOCAL);

//Measuring Performance of Classifier
performance:= ML.Classify.Compare(depData, finalClass);
OUTPUT(performance.Instances_OrigPredited, NAMED('InstacesResults'), ALL);
OUTPUT(performance.CrossAssignments, NAMED('CrossAssig'));
OUTPUT(performance.RecallByClass, NAMED('RecallByClass'));
OUTPUT(performance.PrecisionByClass, NAMED('PrecisionByClass'));
OUTPUT(performance.FP_Rate_ByClass, NAMED('FP_Rate_ByClass'));
OUTPUT(performance.Accuracy, NAMED('Accuracy'));
*/