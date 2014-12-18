IMPORT * FROM ML;
IMPORT * FROM $;
//NaiveBayes classifier
trainer:= ML.Classify.NaiveBayes;

//Monk Dataset
MonkData:= MonkDS.Train_Data;
OUTPUT(MonkData, NAMED('MonkData'), ALL);
ML.ToField(MonkData, fullmds, id);
full_mds:=PROJECT(fullmds, TRANSFORM(Types.DiscreteField, SELF:= LEFT));
indepDataD:= full_mds(number>1);
depDataD := full_mds(number=1);
// Learning Phase
D_Model:= trainer.LearnD(indepDataD, depDataD);
dmodel:= trainer.Model(D_model);
OUTPUT(SORT(dmodel, id), ALL, NAMED('DiscModel'));
//Classification Phase
D_classDist:= trainer.ClassProbDistribD(indepDataD, D_Model);
D_results:= trainer.ClassifyD(indepDataD, D_Model);
D_compare:= Classify.Compare(depDataD, D_results);
AUC_D0:= Classify.AUC_ROC(D_classDist, 0, depDataD); //Area under ROC Curve for class "1"
AUC_D1:= Classify.AUC_ROC(D_classDist, 1, depDataD); //Area under ROC Curve for class "2"
OUTPUT(D_classDist, ALL, NAMED('DisClassDist'));
OUTPUT(D_results, NAMED('DiscClassifResults'), ALL);
OUTPUT(SORT(D_compare.CrossAssignments, c_actual, c_modeled), NAMED('DiscCrossAssig'), ALL);
OUTPUT(AUC_D0, ALL, NAMED('AUC_D0'));
OUTPUT(AUC_D1, ALL, NAMED('AUC_D1'));

//Lymphoma Dataset
lymphomaData:= lymphomaDS.DS;
OUTPUT(lymphomaData, NAMED('lymphomaData'), ALL);
ML.ToField(lymphomaData, full_lds);
//OUTPUT(full_lds_Map,ALL, NAMED('DatasetFieldMap'));
indepDataC:= full_lds(number<4027);
depDataC:= ML.Discretize.ByRounding(full_lds(number=4027));
// Learning Phase
C_Model:= trainer.LearnC(indepDataC, depDataC);
cmodel:= trainer.ModelC(C_model);
OUTPUT(SORT(cmodel, id), ALL, NAMED('ContModel'));
//Classification Phase
C_results:= trainer.ClassifyC(indepDataC, C_Model);
OUTPUT(C_results, NAMED('ContClassifResults'), ALL);
C_compare:= Classify.Compare(depDataC, C_results);
OUTPUT(SORT(C_compare.CrossAssignments, c_actual, c_modeled), NAMED('ContCrossAssig'), ALL);